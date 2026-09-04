import Testing
import Foundation
import CWhisper
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// No model file is loaded here: what this suite pins is the *decode
// contract* — the parameters `whiz transcribe` effectively runs through the
// vendored whisper-cli — the same discipline NS-1 applies to the segmentation
// constants. The values are verified against the vendored sources, and the
// tests make any drift here a failure rather than a quiet behavior change.

@Suite("Whisper batch transcriber")
struct WhisperBatchTranscriberTests {

    @Test("decode profile mirrors the vendored whisper-cli baseline")
    func profileParamsMirrorWhisperCLI() {
        let params = WhisperBatchTranscriber.profileParams()

        // The strategy question — settled the hard way: the designated
        // initializer in whisper.cpp:5929 lists best_of/beam_size as -1, but
        // the strategy switch at whisper.cpp:6020 overrides them to 5, so
        // cli.cpp's beam_size default is 5 and any value > 1 selects beam
        // search (cli.cpp:1213). whiz never passes -bs/-bo, so the effective
        // Python-pipeline profile is BEAM SEARCH, beam 5 — the slow, accurate
        // end of whisper.cpp, unlike dictation's greedy.
        #expect(params.strategy == WHISPER_SAMPLING_BEAM_SEARCH)
        #expect(params.beam_search.beam_size == 5)
        #expect(params.greedy.best_of == 5)

        // The fields where batch deliberately differs from the dictation
        // profile, pinned so they can only change deliberately:
        // - no_timestamps false — batch output IS timestamps.
        // - suppress_nst false — cli.cpp:81's default; dictation sets true.
        // - no_context true — happens to also be the library default, so
        //   each 30 s window decodes independently, same as dictation.
        #expect(params.no_timestamps == false)
        #expect(params.suppress_nst == false)
        #expect(params.suppress_blank == true)
        #expect(params.no_context == true)

        // Temperature fallback exactly as whisper-cli runs it (cli.cpp:46-53):
        // 0.0 initial, 0.2 increment, `--no-fallback` never set by whiz.
        #expect(params.temperature == 0.0)
        #expect(params.temperature_inc == 0.2)
        #expect(params.entropy_thold == 2.4)
        #expect(params.logprob_thold == -1.0)
        #expect(params.no_speech_thold == 0.6)

        // Console printing is whisper-cli's business; the app observes via
        // callbacks instead.
        #expect(params.print_progress == false)
        #expect(params.print_realtime == false)
        #expect(params.print_timestamps == false)
        #expect(params.print_special == false)

        #expect(params.translate == false)
        #expect(params.single_segment == false)
        #expect(params.token_timestamps == false)
        #expect(params.tdrz_enable == false)
        #expect(params.n_max_text_ctx == 16384)
        #expect(params.initial_prompt == nil)

        // VAD is per-call wiring: `whiz/config.py` defaults `vad = True`, so
        // callers pass the Silero model path per transcription; the profile
        // itself stays neutral until then.
        #expect(params.vad == false)
    }

    @Test("segment timestamps convert centiseconds to seconds")
    func timestampsConvertCentiseconds() {
        // whisper.cpp reports t0/t1 in units of 10 ms (`to_timestamp` does
        // `msec = t*10`), so 300 ticks = 3.0 s — a wrong factor of 10 here
        // shifts every segment boundary.
        #expect(WhisperBatchTranscriber.seconds(fromCentiseconds: 0) == 0)
        #expect(WhisperBatchTranscriber.seconds(fromCentiseconds: 300) == 3)
        #expect(abs(WhisperBatchTranscriber.seconds(fromCentiseconds: 12_345) - 123.45) < 1e-9)
        // Two hours of centiseconds — the long-file sanity check.
        #expect(WhisperBatchTranscriber.seconds(fromCentiseconds: 720_000_000) == 7_200_000)
    }

    @Test("thread default mirrors whiz's _auto_threads()")
    func autoThreadsMirrorPython() {
        #expect(WhisperBatchTranscriber.autoThreads == min(8, ProcessInfo.processInfo.activeProcessorCount))
    }

    @Test("transcribing before load throws notLoaded")
    func unloadedThrows() async {
        let transcriber = WhisperBatchTranscriber(
            modelURL: URL(fileURLWithPath: "/nonexistent/model.bin"))
        await #expect(throws: WhisperBatchError.notLoaded) {
            try await transcriber.transcribe(samples: [0.1, 0.2])
        }
    }

    @Test("cancellation box starts clear, cancels, and forwards progress")
    func boxCancellationAndProgress() {
        let collector = ValueCollector()
        let box = BatchBox(onProgress: collector.record)
        #expect(box.isCancelled == false)

        box.cancel()
        #expect(box.isCancelled == true)

        box.reportProgress(0.5)
        #expect(collector.values == [0.5])
    }
}

/// Lock-guarded collection point for `@Sendable` progress callbacks in tests.
private final class ValueCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [Double] = []

    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func record(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(value)
    }
}