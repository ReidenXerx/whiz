import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The setup dialog's contract is the overlay it produces: the config
// snapshot with the dialog's choices applied, never written back to disk.
// The view itself has no test host (consistent with the other UI files).

@Suite("Transcription setup")
struct TranscriptionSetupTests {

    private func makeSettings(
        numSpeakers: Int = 0,
        ocr: Bool = false,
        aiModel: String = ""
    ) -> BatchSettings {
        var s = BatchSettings()
        s.numSpeakers = numSpeakers
        s.ocr = ocr
        s.aiModel = aiModel
        s.aiBaseURL = "http://example.test/v1"
        s.clusterThreshold = 0.85
        return s
    }

    // MARK: - Seeding from config

    @Test("the dialog seeds from the config snapshot")
    @MainActor
    func seedsFromConfig() {
        let model = TranscriptionSetupModel(settings: makeSettings(
            numSpeakers: 3, ocr: true, aiModel: "qwen2.5-vl"))
        #expect(!model.speakersAuto)
        #expect(model.speakerCount == 3)
        #expect(model.ocrEnabled)
        #expect(model.analyzeEnabled)
        #expect(model.aiModel == "qwen2.5-vl")

        let defaults = TranscriptionSetupModel(settings: makeSettings())
        #expect(defaults.speakersAuto)          // numSpeakers 0 => auto
        #expect(!defaults.analyzeEnabled)       // no ai_model => off
    }

    // MARK: - The resolved overlay

    @Test("auto-detect resolves to 0; an explicit count wins when auto is off")
    @MainActor
    func speakerCountResolves() {
        let auto = TranscriptionSetupModel(settings: makeSettings(numSpeakers: 3))
        auto.speakersAuto = true
        auto.speakerCount = 7
        #expect(auto.resolvedSettings().numSpeakers == 0)

        let explicit = TranscriptionSetupModel(settings: makeSettings())
        explicit.speakersAuto = false
        explicit.speakerCount = 5
        #expect(explicit.resolvedSettings().numSpeakers == 5)
    }

    @Test("the OCR toggle passes straight through")
    @MainActor
    func ocrPassesThrough() {
        let model = TranscriptionSetupModel(settings: makeSettings(ocr: false))
        model.ocrEnabled = true
        #expect(model.resolvedSettings().ocr)
        model.ocrEnabled = false
        #expect(!model.resolvedSettings().ocr)
    }

    @Test("analyze-on keeps the chosen model; analyze-off skips it for this run only")
    @MainActor
    func analyzeOverlay() {
        // On with the dialog's choice.
        let on = TranscriptionSetupModel(settings: makeSettings(aiModel: "qwen2.5-vl"))
        on.aiModel = "llava:13b"
        let resolvedOn = on.resolvedSettings()
        #expect(resolvedOn.aiModel == "llava:13b")

        // Off: the run skips analysis even though the config names a model —
        // and the config snapshot itself is untouched.
        let off = TranscriptionSetupModel(settings: makeSettings(aiModel: "qwen2.5-vl"))
        off.analyzeEnabled = false
        let resolvedOff = off.resolvedSettings()
        #expect(resolvedOff.aiModel.isEmpty)
        #expect(off.aiModel == "qwen2.5-vl")   // the @Published choice survives
    }

    @Test("untouched keys come from the config snapshot")
    @MainActor
    func otherKeysPreserved() {
        let model = TranscriptionSetupModel(settings: makeSettings())
        let resolved = model.resolvedSettings()
        #expect(resolved.aiBaseURL == "http://example.test/v1")
        #expect(resolved.clusterThreshold == 0.85)
        #expect(resolved.language == "auto")
    }

    // MARK: - The path field

    @Test("a real file is a valid input; anything else is not")
    @MainActor
    func selectedInputValidation() throws {
        let model = TranscriptionSetupModel(settings: makeSettings())

        #expect(model.selectedInput == nil)
        model.pathText = "/definitely/not/a/file.mp4"
        #expect(model.selectedInput == nil)
        #expect(!model.canRun)

        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-setup-\(UUID().uuidString).mp4")
        try Data("stub".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        model.pathText = "  \(file.path)  "   // pasted paths carry whitespace
        #expect(model.selectedInput == file)
        #expect(model.canRun)
    }

    // MARK: - The flow state machine

    @Test("Run without a path does nothing")
    @MainActor
    func runNeedsAnInput() {
        let flow = TranscriptionFlowModel()
        flow.startRun(backend: SleepingBackend())
        #expect(flow.run == nil)
        #expect(!flow.hasActiveRun)
    }

    @Test("Run swaps the flow to the active running phase")
    @MainActor
    func runEntersProgressPhase() throws {
        let file = try makeStubFile()
        let flow = TranscriptionFlowModel()
        flow.setup.pathText = file.path
        flow.startRun(backend: SleepingBackend())

        #expect(flow.run != nil)
        #expect(flow.hasActiveRun)

        // Cancel clears the phase — this is what windowWillClose / restart do.
        flow.restart()
        #expect(flow.run == nil)
        #expect(!flow.hasActiveRun)
        // And the setup phase comes back fresh, seeded from a new snapshot.
        #expect(flow.setup.pathText.isEmpty)
    }

    @Test("a finished or failed run is not active — starting over is allowed")
    @MainActor
    func finishedRunsAreNotActive() async throws {
        let file = try makeStubFile()
        let flow = TranscriptionFlowModel()
        flow.setup.pathText = file.path
        flow.startRun(backend: FailingBackend())

        // Let the run's task land its failure before asserting the phase.
        for _ in 0..<50 where flow.hasActiveRun {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(flow.run?.stage != .running)
        #expect(!flow.hasActiveRun)

        // A new run replaces the finished one.
        flow.setup.pathText = file.path
        flow.startRun(backend: SleepingBackend())
        #expect(flow.hasActiveRun)
    }

    private func makeStubFile() throws -> URL {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-flow-\(UUID().uuidString).mp4")
        try Data("stub".utf8).write(to: file)
        return file
    }
}

/// Test backend that idles until cancelled — an in-flight run stand-in.
private struct SleepingBackend: TranscriptionBackend {
    func transcribe(
        input: URL,
        outputDirectory: URL,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws -> URL {
        while !Task.isCancelled {
            onEvent(.progress(0.5))
            try await Task.sleep(for: .milliseconds(50))
        }
        throw CancellationError()
    }
}

/// Test backend that fails immediately — a completed-not-running stand-in.
private struct FailingBackend: TranscriptionBackend {
    func transcribe(
        input: URL,
        outputDirectory: URL,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws -> URL {
        throw WhisperBatchError.notLoaded
    }
}