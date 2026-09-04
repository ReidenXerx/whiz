import CSherpa
import Foundation

/// Locates the diarization models — mirrors `whiz/diarize.py`'s
/// `find_segmentation_model` / `find_embedding_model` candidate-for-candidate,
/// so a machine that has run `whiz models download-diarization` needs nothing
/// else, and a config pointing at explicit paths wins the same way.
enum DiarizationModel {

    static let segmentationDirectoryName = "sherpa-onnx-pyannote-segmentation-3-0"
    static let segmentationModelName = "model.int8.onnx"
    static let embeddingModelName = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    /// diarize.py:_default_diarization_dir.
    static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/whiz/diarization")
    }

    /// The pyannote segmentation model: `.int8.onnx` preferred, `.onnx`
    /// fallback, the same six candidate shapes Python walks.
    static func findSegmentationModel(
        explicit: String,
        searchDirectories: [URL]
    ) -> URL? {
        if !explicit.isEmpty {
            let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        let segDir = defaultDirectory.appendingPathComponent(segmentationDirectoryName)
        var candidates: [URL] = [
            segDir.appendingPathComponent(segmentationModelName),
            segDir.appendingPathComponent("model.onnx"),
        ]
        for directory in searchDirectories {
            candidates.append(directory
                .appendingPathComponent(segmentationDirectoryName)
                .appendingPathComponent(segmentationModelName))
            candidates.append(directory
                .appendingPathComponent(segmentationDirectoryName)
                .appendingPathComponent("model.onnx"))
            candidates.append(directory.appendingPathComponent(segmentationModelName))
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// The 3D-Speaker embedding extractor.
    static func findEmbeddingModel(
        explicit: String,
        searchDirectories: [URL]
    ) -> URL? {
        if !explicit.isEmpty {
            let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        var candidates = [defaultDirectory.appendingPathComponent(embeddingModelName)]
        for directory in searchDirectories {
            candidates.append(directory.appendingPathComponent(embeddingModelName))
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Both models present? — the Python `_diarization_available` gate, used to
    /// decide between "diarize" and "skip with a hint".
    static func isAvailable(settings: BatchSettings) -> Bool {
        findSegmentationModel(
            explicit: settings.diarizationSegmentationModel,
            searchDirectories: WhisperModel.searchDirectories) != nil
            && findEmbeddingModel(
                explicit: settings.diarizationEmbeddingModel,
                searchDirectories: WhisperModel.searchDirectories) != nil
    }
}

/// Native speaker diarization over the sherpa-onnx C API — the vendored
/// dylibs are the same binaries the Python pipeline runs, so cluster
/// boundaries agree by construction.
///
/// Mirrors `diarize.py:run_diarization`'s parameters exactly: pyannote
/// segmentation + eres2net embedding + fast clustering (known speaker count
/// or threshold-based auto), `min_duration_on` 0.3 / `min_duration_off` 0.5,
/// 16 kHz mono samples. The C progress callback reports per-chunk progress
/// but ignores its return value — mid-run cancellation is not available
/// here, exactly like the Python side; the surrounding Task cancels between
/// phases.
///
/// The process call blocks for minutes on long audio (the embedding pass is
/// the expensive part diarize.py's cache exists to skip), so it runs on a
/// dedicated serial queue rather than the cooperative pool.
enum Diarization {

    private static let queue = DispatchQueue(label: "whiz.diarization", qos: .userInitiated)

    /// Run diarization over decoded 16 kHz mono samples, returning segments
    /// sorted by start time (the C API's ResultSortByStartTime — the same
    /// ordering Python's sort_by_start_time produces).
    static func run(
        samples: [Float],
        segmentationModel: URL,
        embeddingModel: URL,
        numSpeakers: Int = 0,
        threshold: Float = 0.9,
        threads: Int? = nil,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> [DiarSegment] {
        guard !samples.isEmpty else { return [] }

        let request = Request(
            samples: samples,
            segmentationModel: segmentationModel.path,
            embeddingModel: embeddingModel.path,
            numSpeakers: numSpeakers,
            threshold: threshold,
            threads: threads ?? min(4, ProcessInfo.processInfo.activeProcessorCount),
            onProgress: onProgress)

        let box = BatchBox(onProgress: onProgress)
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    continuation.resume(returning: try Self.runOnQueue(request, box: box))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct Request: Sendable {
        let samples: [Float]
        let segmentationModel: String
        let embeddingModel: String
        let numSpeakers: Int
        let threshold: Float
        let threads: Int
        let onProgress: (@Sendable (Double) -> Void)?
    }

    private static func runOnQueue(_ request: Request, box: BatchBox) throws -> [DiarSegment] {
        try request.segmentationModel.withCString { segPath in
            try request.embeddingModel.withCString { embPath in
                try "cpu".withCString { provider in
                    var config = SherpaOnnxOfflineSpeakerDiarizationConfig()
                    config.segmentation.pyannote.model = segPath
                    config.segmentation.num_threads = Int32(request.threads)
                    config.segmentation.provider = provider
                    config.embedding.model = embPath
                    config.embedding.num_threads = Int32(request.threads)
                    config.embedding.provider = provider
                    config.clustering.num_clusters = request.numSpeakers > 0
                        ? Int32(request.numSpeakers)
                        : -1   // auto-detect via threshold (diarize.py:257-260)
                    config.clustering.threshold = request.threshold
                    config.min_duration_on = 0.3
                    config.min_duration_off = 0.5

                    guard let sd = SherpaOnnxCreateOfflineSpeakerDiarization(&config) else {
                        throw DiarizationError.setupFailed(
                            "sherpa-onnx config validation failed; check the model paths")
                    }
                    let sampleRate = SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd)
                    let expected = Int(WhisperEngine.sampleRate)
                    if sampleRate != expected {
                        SherpaOnnxDestroyOfflineSpeakerDiarization(sd)
                        throw DiarizationError.wrongSampleRate(got: Int(sampleRate), expected: expected)
                    }

                    // The user-data pointer is the box, retained by `run` for
                    // exactly the duration of this call.
                    let progressCallback: SherpaOnnxOfflineSpeakerDiarizationProgressCallback = {
                        done, total, arg in
                        guard let arg, total > 0 else { return 0 }
                        let box = Unmanaged<BatchBox>.fromOpaque(arg).takeUnretainedValue()
                        box.reportProgress(Double(done) / Double(total))
                        return 0
                    }

                    let result = request.samples.withUnsafeBufferPointer { buffer in
                        SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                            sd, buffer.baseAddress, Int32(buffer.count),
                            progressCallback,
                            Unmanaged.passUnretained(box).toOpaque())
                    }
                    guard let result else {
                        SherpaOnnxDestroyOfflineSpeakerDiarization(sd)
                        throw DiarizationError.processFailed
                    }

                    let count = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result)
                    let segments = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result)
                    var out: [DiarSegment] = []
                    if let segments {
                        if count > 0 {
                            out.reserveCapacity(Int(count))
                            for index in 0..<Int(count) {
                                out.append(DiarSegment(
                                    start: Double(segments[index].start),
                                    end: Double(segments[index].end),
                                    speaker: Int(segments[index].speaker)))
                            }
                        }
                        SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments)
                    }
                    SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result)
                    SherpaOnnxDestroyOfflineSpeakerDiarization(sd)
                    return out
                }
            }
        }
    }
}

enum DiarizationError: LocalizedError, Sendable, Equatable {
    case setupFailed(String)
    case wrongSampleRate(got: Int, expected: Int)
    case processFailed

    var errorDescription: String? {
        switch self {
        case .setupFailed(let detail):
            return "Diarization could not start: \(detail)."
        case .wrongSampleRate(let got, let expected):
            return "Diarization expected \(expected) Hz audio, got \(got) Hz."
        case .processFailed:
            return "Diarization processing failed."
        }
    }
}