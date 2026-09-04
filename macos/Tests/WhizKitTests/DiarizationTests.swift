import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// Model discovery is pinned candidate-for-candidate against
// whiz/diarize.py:find_segmentation_model / find_embedding_model, and the
// engine is exercised against the real vendored dylibs + the user's downloaded
// models — gated like the whisper e2e, because without the models there is
// nothing to diarize with. The vendored dylibs must exist for the test binary
// to load at all; scripts/build-sherpa.sh is now a build prerequisite, the
// same deal scripts/build-whisper.sh always was.

@Suite("Diarization")
struct DiarizationTests {

    // MARK: - Settings (config.py diarization keys)

    @Test("diarization settings parse from the shared config, defaults mirror config.py")
    func diarizationSettingsParseAndDefaults() {
        // config.py:48-55 — diarize is OFF by default (video auto-enables),
        // speakers auto-detect via the 0.9 cluster threshold, models
        // auto-discovered.
        let defaults = BatchSettings.from([:])
        #expect(defaults.diarize == false)
        #expect(defaults.numSpeakers == 0)
        #expect(defaults.clusterThreshold == 0.9)
        #expect(defaults.diarizationSegmentationModel.isEmpty)
        #expect(defaults.diarizationEmbeddingModel.isEmpty)

        let parsed = BatchSettings.from(FlatTOML.parse("""
            diarize = true
            num_speakers = 3
            cluster_threshold = 0.85
            diarization_segmentation_model = "~/models/seg.onnx"
            diarization_embedding_model = "~/models/emb.onnx"
            """))
        #expect(parsed.diarize == true)
        #expect(parsed.numSpeakers == 3)
        #expect(parsed.clusterThreshold == 0.85)
        #expect(parsed.diarizationSegmentationModel == "~/models/seg.onnx")
        #expect(parsed.diarizationEmbeddingModel == "~/models/emb.onnx")
    }

    // MARK: - Model discovery (diarize.py candidate order)

    @Test("explicit segmentation model paths win; missing ones fall through to discovery")
    func explicitSegmentationModelWins() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let explicit = dir.appendingPathComponent("my-seg.onnx")
        try Data("x".utf8).write(to: explicit)

        #expect(DiarizationModel.findSegmentationModel(
            explicit: explicit.path, searchDirectories: []) == explicit)

        // Python parity (diarize.py:65-67): a configured-but-missing path does
        // NOT fail — it falls back to the auto-discovered candidates. On a
        // machine with downloaded models that is the real model; on a clean
        // machine both are nil. Either way, fall-through == plain discovery.
        let fallback = DiarizationModel.findSegmentationModel(
            explicit: "/nonexistent/seg.onnx", searchDirectories: [])
        let discovered = DiarizationModel.findSegmentationModel(
            explicit: "", searchDirectories: [])
        #expect(fallback == discovered)
    }

    @Test("int8 segmentation is preferred over model.onnx, per diarize.py order")
    func int8SegmentationPreferred() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let segDir = dir.appendingPathComponent(DiarizationModel.segmentationDirectoryName)
        try FileManager.default.createDirectory(at: segDir, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: segDir.appendingPathComponent("model.onnx"))
        try Data("x".utf8).write(to: segDir.appendingPathComponent("model.int8.onnx"))

        let found = DiarizationModel.findSegmentationModel(explicit: "", searchDirectories: [dir])
        #expect(found?.lastPathComponent == "model.int8.onnx")
    }

    @Test("the embedding model is found in the search directories")
    func embeddingModelSearch() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("x".utf8).write(to: dir.appendingPathComponent(DiarizationModel.embeddingModelName))

        #expect(DiarizationModel.findEmbeddingModel(
            explicit: "", searchDirectories: [dir])?.lastPathComponent
            == DiarizationModel.embeddingModelName)
    }

    // MARK: - Engine (real dylibs + real models, gated like the whisper e2e)

    /// The corpus WAV is synthetic — pyannote may find no speech at all, so
    /// what is pinned is that the full native path runs: dylib loads, models
    /// load, the C pipeline executes, and whatever comes back is sorted by
    /// start time with valid speaker ids. Real multi-speaker audio is the
    /// user's click-through.
    @Test(.disabled(if: !DiarizationModel.isAvailable(settings: BatchSettings())))
    func runsThroughTheRealPipeline() async throws {
        let segModel = DiarizationModel.findSegmentationModel(
            explicit: "", searchDirectories: WhisperModel.searchDirectories)
        let embModel = DiarizationModel.findEmbeddingModel(
            explicit: "", searchDirectories: WhisperModel.searchDirectories)
        #expect(segModel != nil)
        #expect(embModel != nil)

        // The golden corpus's quiet fixture: 5.2 s of deterministic audio.
        let wav = repoRoot.appendingPathComponent("tuning/golden/quiet_two_utterances.wav")
        let samples = try await AudioFileDecoder.extractSamples(at: wav)

        let progress = ProgressRecorder()
        let segments = try await Diarization.run(
            samples: samples.samples,
            segmentationModel: segModel!,
            embeddingModel: embModel!,
            numSpeakers: 2,
            onProgress: { progress.record($0) })

        let sorted = segments.map(\.start)
        #expect(sorted == sorted.sorted())
        #expect(segments.allSatisfy { $0.end >= $0.start && $0.speaker >= 0 })
        #expect(progress.latest.isEmpty || progress.latest.last == 1.0)
        print("DIAR E2E: \(segments.count) segments from the synthetic corpus")
    }

    // MARK: - Helpers

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // WhizKitTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // macos
            .deletingLastPathComponent()   // repo root
    }

    private func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-diar-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// Lock-guarded collection point for the `@Sendable` progress callback in tests.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []

    func record(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }

    var latest: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}