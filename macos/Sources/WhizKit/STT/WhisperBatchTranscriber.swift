import CWhisper
import Foundation

/// Batch transcription profile over the same whisper.cpp the dictation engine
/// uses — deliberately a *separate actor with its own context*, not a method
/// on `WhisperEngine`.
///
/// Two reasons, both about interference:
///
/// - An actor serializes its calls. `whisper_full` over a whole file runs for
///   minutes, so a batch job on the dictation engine would hold the actor and
///   leave the dictation hotkey dead until the file finished — the app's
///   primary feature blocked by its secondary one. Separate contexts cost one
///   extra resident model, which is the cheap side of the trade.
/// - Keeping every batch concern in this file leaves `WhisperEngine.swift`
///   untouched, so the dictation path cannot regress by construction.
///
/// The decode profile mirrors what `whiz transcribe` runs through the vendored
/// `whisper-cli` — verified against the vendored sources, not upstream lore,
/// and pinned field-by-field in `WhisperBatchTranscriberTests`. The chain:
/// cli.cpp:45 takes its `beam_size` default from
/// `whisper_full_default_params(BEAM_SEARCH).beam_search.beam_size`, which is
/// **5** — the strategy-specific `switch` at whisper.cpp:6020 overrides the
/// `-1` in the initializer list, so never trust that list alone — and
/// cli.cpp:1213 then picks beam search whenever `beam_size > 1`. whiz never
/// passes `-bs`/`-bo`, so the effective Python-pipeline profile is beam
/// search, beam 5. What actually differs from the dictation profile:
///
/// - `strategy` beam search — dictation decodes greedy; batch pays the slower
///   decode for the accuracy whiz's file pipeline gets.
/// - `no_timestamps` false — batch output is timed segments; dictation throws
///   timestamps away.
/// - `suppress_nst` false — cli's default (cli.cpp:81); the dictation path
///   sets it true to strip `[MUSIC]`-style tokens from short utterances.
/// - VAD on — `whiz/config.py` defaults `vad = True`, so `whiz transcribe`
///   normally runs whisper-cli's built-in Silero VAD (`--vad -vt 0.5`). The
///   caller passes the VAD model path per transcription; without it the
///   profile decodes un-VAD-ed, which is a documented divergence to use only
///   when no VAD model is available.
///
/// Threading: the whisper context is not thread-safe for same-context calls,
/// and `whisper_full` must not block a cooperative-pool thread for minutes the
/// way short dictation calls can. Every context-touching operation therefore
/// runs on one dedicated serial queue — the queue *is* the single-thread
/// guarantee the dictation engine gets from actor isolation — while the actor
/// serializes the Swift-level entry points and owns the state. Cancellation is
/// a lock-guarded box read by the C abort callback on whisper's thread.
actor WhisperBatchTranscriber {

    /// One recognized stretch of speech. Timestamps in seconds — the C API
    /// reports centiseconds (`whisper.cpp`'s `to_timestamp` does `msec = t*10`),
    /// so the conversion is `t / 100`.
    struct Segment: Sendable, Equatable {
        let start: Double
        let end: Double
        let text: String
    }

    /// Progress as a 0…1 fraction, reported from whisper's internal thread —
    /// handlers must be cheap and hop if they touch UI.
    typealias OnProgress = @Sendable (Double) -> Void

    private static let queue = DispatchQueue(label: "whiz.whisper.batch", qos: .userInitiated)

    private var handle: ContextHandle?
    private let modelURL: URL
    private var activeCancellation: BatchBox?

    init(modelURL: URL) {
        self.modelURL = modelURL
    }

    var isLoaded: Bool { handle != nil }

    // MARK: - Lifecycle

    func load() async throws {
        guard handle == nil else { return }
        GGMLBackends.registerOnce()

        let path = modelURL.path
        let name = modelURL.lastPathComponent
        let loaded = try await Self.onQueue {
            var params = whisper_context_default_params()
            // Metal, matching the dictation engine's context setup.
            params.use_gpu = true
            params.flash_attn = true

            guard let ctx = whisper_init_from_file_with_params(path, params) else {
                throw WhisperBatchError.modelLoadFailed(name)
            }
            return ContextHandle(pointer: ctx)
        }
        handle = loaded
    }

    func unload() async {
        guard let handle else { return }
        self.handle = nil
        await Self.onQueue {
            whisper_free(handle.pointer)
        }
    }

    // MARK: - Transcription

    /// Transcribe a whole file's PCM into timestamped segments.
    ///
    /// - Parameters:
    ///   - samples: 16 kHz mono samples, as `AudioFileDecoder` produces.
    ///   - language: BCP-47 code, or nil for auto-detection (whiz passes
    ///     `config.language` through `-l`, default "auto").
    ///   - vadModelPath: path to `ggml-silero-vad.bin`. `whiz transcribe`
    ///     runs VAD by default (`config.vad = True`), so callers that want
    ///     pipeline parity must supply this.
    ///   - vadThreshold: speech probability threshold (`config.vad_threshold`,
    ///     default 0.5).
    ///   - threads: decode threads; nil mirrors whiz's `_auto_threads()`.
    ///   - onProgress: optional 0…1 progress callback.
    ///   - onSegment: optional per-segment callback, invoked as whisper
    ///     recognizes each new segment — the live transcript feed.
    func transcribe(
        samples: [Float],
        language: String? = nil,
        vadModelPath: URL? = nil,
        vadThreshold: Float = 0.5,
        threads: Int? = nil,
        onProgress: OnProgress? = nil,
        onSegment: (@Sendable (Segment) -> Void)? = nil
    ) async throws -> [Segment] {
        guard let handle else { throw WhisperBatchError.notLoaded }
        guard !samples.isEmpty else { return [] }

        let box = BatchBox(onProgress: onProgress)
        activeCancellation = box
        defer { activeCancellation = nil }

        let request = Request(
            samples: samples,
            language: language ?? "auto",
            vadModelPath: vadModelPath?.path,
            vadThreshold: vadThreshold,
            threads: threads ?? Self.autoThreads,
            onSegment: onSegment)

        return try await Self.onQueue {
            try Self.run(handle: handle, request: request, box: box)
        }
    }

    /// Stop the in-flight transcription. `whisper_full` checks the abort
    /// callback between ggml computations, so cancellation lands within one
    /// encoder/decoder step, and the call returns `.cancelled`.
    func cancel() {
        activeCancellation?.cancel()
    }

    // MARK: - The C run

    private struct Request: Sendable {
        let samples: [Float]
        let language: String
        let vadModelPath: String?
        let vadThreshold: Float
        let threads: Int
        let onSegment: (@Sendable (Segment) -> Void)?
    }

    /// The whisper.cpp decode profile `whiz transcribe` runs through
    /// whisper-cli — pinned field-by-field in `WhisperBatchTranscriberTests`
    /// so drift here is a test failure, the same discipline NS-1 applies to
    /// the segmentation constants. Nonisolated and parameter-free: string and
    /// callback pointers are attached at call time inside `run`.
    nonisolated static func profileParams() -> whisper_full_params {
        // Beam search with beam 5: cli.cpp's own beam_size default comes from
        // the strategy-specific switch (whisper.cpp:6020) — the initializer
        // list's -1 is a decoy — and any beam_size > 1 selects beam search
        // (cli.cpp:1213). whiz never overrides it.
        var params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
        // cli.cpp:1242 also carries best_of through even under beam search,
        // where it is inert — matched for fidelity, and load-bearing the moment
        // anyone flips the strategy.
        params.greedy.best_of = 5

        // The console-print flags are whisper-cli conveniences; we observe
        // through callbacks instead.
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
        params.print_special = false

        // Everything else is deliberately whisper.cpp's own default, because
        // that is what cli.cpp keeps unless a flag overrides it: no_context
        // true, no_timestamps false, suppress_blank true, suppress_nst false,
        // temperature 0.0 with 0.2 increments and the entropy/logprob/
        // no-speech fallback thresholds. See whisper.cpp:5929 and cli.cpp:1208.
        return params
    }

    /// Mirror of whiz's `_auto_threads()` (`min(8, cpu_count)`).
    nonisolated static var autoThreads: Int {
        min(8, ProcessInfo.processInfo.activeProcessorCount)
    }

    /// whisper.cpp segment timestamps are centiseconds; seconds = t/100.
    nonisolated static func seconds(fromCentiseconds t: Int64) -> Double {
        Double(t) / 100.0
    }

    private static func run(handle: ContextHandle, request: Request, box: BatchBox) throws -> [Segment] {
        let ctx = handle.pointer
        var params = profileParams()
        params.n_threads = Int32(request.threads)

        // The user-data pointer must stay valid while whisper_full runs; the
        // box is retained by the actor for exactly that long (unretained
        // bridging is therefore correct, not an optimization).
        params.progress_callback = { _, _, progress, data in
            guard let data else { return }
            Unmanaged<BatchBox>.fromOpaque(data).takeUnretainedValue()
                .reportProgress(Double(progress) / 100.0)
        }
        params.progress_callback_user_data = Unmanaged.passUnretained(box).toOpaque()
        params.abort_callback = { data in
            guard let data else { return false }
            return Unmanaged<BatchBox>.fromOpaque(data).takeUnretainedValue().isCancelled
        }
        params.abort_callback_user_data = Unmanaged.passUnretained(box).toOpaque()

        // whisper.cpp hands new segments to this callback as they are
        // recognized; the API contract says the whisper_full_* getters are
        // legal to call from it, and the new ones are the last nNew of the
        // context's segments.
        params.new_segment_callback = { ctx, _, nNew, data in
            guard let ctx, let data, nNew > 0 else { return }
            let box = Unmanaged<BatchBox>.fromOpaque(data).takeUnretainedValue()
            let total = whisper_full_n_segments(ctx)
            let first = total - nNew
            guard first >= 0 else { return }
            for index in first..<total {
                // Inline conversions, not calls into Self: a @convention(c)
                // pointer must capture nothing, and a static-method call
                // captures the metatype.
                let start = Double(whisper_full_get_segment_t0(ctx, index)) / 100.0
                let end = Double(whisper_full_get_segment_t1(ctx, index)) / 100.0
                let text = whisper_full_get_segment_text(ctx, index).map {
                    String(cString: $0).trimmingCharacters(in: .whitespacesAndNewlines)
                } ?? ""
                guard !text.isEmpty else { continue }
                box.reportSegment(Segment(start: start, end: end, text: text))
            }
        }
        params.new_segment_callback_user_data = Unmanaged.passUnretained(box).toOpaque()

        // whisper.cpp does not copy these C strings; nesting the buffers keeps
        // them alive for exactly the scope of the call.
        try request.language.withCString { languagePtr in
            var vadParams = whisper_vad_default_params()
            vadParams.threshold = request.vadThreshold

            func launch(_ vadPtr: UnsafePointer<CChar>?) throws {
                params.language = languagePtr
                params.vad = vadPtr != nil
                params.vad_model_path = vadPtr
                params.vad_params = vadParams

                let status = request.samples.withUnsafeBufferPointer { buffer in
                    whisper_full(ctx, params, buffer.baseAddress, Int32(buffer.count))
                }
                guard status == 0 else {
                    throw box.isCancelled
                        ? WhisperBatchError.cancelled
                        : WhisperBatchError.transcriptionFailed(Int(status))
                }
            }

            if let vadPath = request.vadModelPath {
                try vadPath.withCString { vadPtr in
                    try launch(vadPtr)
                }
            } else {
                try launch(nil)
            }
        }

        var segments: [Segment] = []
        for index in 0..<whisper_full_n_segments(ctx) {
            let start = seconds(fromCentiseconds: whisper_full_get_segment_t0(ctx, index))
            let end = seconds(fromCentiseconds: whisper_full_get_segment_t1(ctx, index))
            let text = whisper_full_get_segment_text(ctx, index).map { String(cString: $0) } ?? ""
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                segments.append(Segment(start: start, end: end, text: trimmed))
            }
        }
        return segments
    }

    /// Run `work` on the context's serial queue and await its result. The
    /// queue is the single-thread guarantee for the whisper context (see the
    /// type comment); crossing it with Sendable-only inputs is what makes that
    /// sound under Swift 6 concurrency.
    private static func onQueue<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Non-throwing variant for work that cannot fail (freeing a context).
    private static func onQueue<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: work())
            }
        }
    }
}

/// Sendable wrapper for the whisper context pointer. `@unchecked` is sound
/// because the pointer is only ever dereferenced on the transcriber's one
/// serial queue — the queue is the single-thread guarantee; the wrapper just
/// gets the type across the queue boundary.
private final class ContextHandle: @unchecked Sendable {
    let pointer: OpaquePointer

    init(pointer: OpaquePointer) {
        self.pointer = pointer
    }
}

/// Cancellation + progress state shared with whisper.cpp's C callbacks, which
/// run on whisper's own thread during `whisper_full`.
final class BatchBox: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private let onProgress: (@Sendable (Double) -> Void)?
    private let onSegment: (@Sendable (WhisperBatchTranscriber.Segment) -> Void)?

    init(
        onProgress: (@Sendable (Double) -> Void)?,
        onSegment: (@Sendable (WhisperBatchTranscriber.Segment) -> Void)? = nil
    ) {
        self.onProgress = onProgress
        self.onSegment = onSegment
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
    }

    // Both handlers are immutable `let`s, so reading them is safe without the
    // lock — only `cancelled` is mutable state, and invoking user code while
    // holding a lock is how you get re-entrant deadlocks.
    func reportProgress(_ fraction: Double) {
        onProgress?(fraction)
    }

    func reportSegment(_ segment: WhisperBatchTranscriber.Segment) {
        onSegment?(segment)
    }
}

enum WhisperBatchError: LocalizedError, Sendable, Equatable {
    case notLoaded
    case modelLoadFailed(String)
    case cancelled
    case transcriptionFailed(Int)

    var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Whisper model is not loaded."
        case .modelLoadFailed(let name):
            return "Could not load Whisper model '\(name)'."
        case .cancelled:
            return "Transcription was cancelled."
        case .transcriptionFailed(let code):
            return "Whisper transcription failed (status \(code))."
        }
    }
}