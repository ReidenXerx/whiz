import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The orchestration suite runs against a scriptable ChatClient — the same
// monkeypatch pattern tests/test_ai.py uses, made type-safe — so chunking,
// rolling context, prompt routing and the Essentials augmentation are pinned
// without any network. The HTTP suite (serialized, because URLProtocol
// mocking is process-global) pins _post_chat's retry contract against a
// stubbed server.

// MARK: - Test double

private actor MockChatClient: AnalysisEngine.ChatClient {
    private let responses: [String]
    private let failures: [Error]
    private var recorded: [(prompt: String, frames: [URL], maxFrames: Int)] = []

    init(responses: [String] = [], failures: [Error] = []) {
        self.responses = responses
        self.failures = failures
    }

    func complete(prompt: String, frames: [URL], maxFrames: Int) async throws -> String {
        recorded.append((prompt, frames, maxFrames))
        let index = recorded.count - 1
        if index < failures.count {
            throw failures[index]
        }
        if responses.isEmpty { return "" }
        return index < responses.count ? responses[index] : responses.last!
    }

    var calls: [(prompt: String, frames: [URL], maxFrames: Int)] {
        recorded
    }
}

// MARK: - Orchestration

@Suite("AI analysis orchestration")
struct AIAnalysisTests {

    private func entry(_ start: Double, _ speaker: String = "Speaker A",
                       text: String = "hello world", frame: String = "") -> FrameExtractor.Entry {
        FrameExtractor.Entry(index: 1, start: start, end: start + 2, speaker: speaker,
                             text: text, frame: frame, ocr: "")
    }

    // MARK: transcript_text

    @Test("transcript text renders clock timestamps and speaker labels")
    func transcriptTextRenders() {
        let entries = [
            entry(83.456, "Speaker A", text: "hello"),
            FrameExtractor.Entry(index: 2, start: 100.0, end: 102, speaker: "Alice",
                                 text: "hi there", frame: "seg0002.jpg", ocr: ""),
        ]
        #expect(AnalysisEngine.transcriptText(entries: entries)
                == "[00:01:23] Speaker A: hello\n[00:01:40] Alice: hi there")

        let labeled = [LabeledSegment(
            segment: WhisperBatchTranscriber.Segment(start: 5.4, end: 7, text: "  hi  "),
            speaker: "Bob")]
        #expect(AnalysisEngine.transcriptText(labeled: labeled) == "[00:00:05] Bob: hi")
    }

    // MARK: subsample (test_ai.py fixtures)

    @Test("subsample under the cap returns everything")
    func subsampleUnderCap() {
        #expect(AnalysisEngine.subsample([1, 2, 3], maxFrames: 5) == [1, 2, 3])
        #expect(AnalysisEngine.subsample([1, 2, 3], maxFrames: 0) == [1, 2, 3])
        #expect(AnalysisEngine.subsample([1, 2, 3], maxFrames: -1) == [1, 2, 3])
    }

    @Test("subsample over the cap spreads evenly")
    func subsampleEvenSpread() {
        #expect(AnalysisEngine.subsample(Array(0..<10), maxFrames: 3) == [0, 3, 6])
    }

    @Test("subsample of one returns the middle")
    func subsampleOneMiddle() {
        #expect(AnalysisEngine.subsample([0, 1, 2, 3, 4], maxFrames: 1) == [2])
    }

    // MARK: chunking (test_ai.py fixtures)

    @Test("chunk_entries splits to size with a floor of one")
    func chunkEntriesSplits() {
        #expect(AnalysisEngine.chunkEntries([1, 2, 3, 4, 5], chunkSize: 2)
                == [[1, 2], [3, 4], [5]])
        #expect(AnalysisEngine.chunkEntries([1, 2], chunkSize: 8) == [[1, 2]])
        let noEntries: [Int] = []
        let noChunks: [[Int]] = []
        #expect(AnalysisEngine.chunkEntries(noEntries, chunkSize: 8) == noChunks)
        #expect(AnalysisEngine.chunkEntries([1, 2], chunkSize: 0) == [[1], [2]])
    }

    @Test("chunk_text splits long transcripts on line boundaries")
    func chunkTextSplits() {
        let short = "one line"
        #expect(AnalysisEngine.chunkText(short, targetChars: 6000) == [short])
        #expect(AnalysisEngine.chunkText("", targetChars: 6000) == [])
        #expect(AnalysisEngine.chunkText(short, targetChars: 0) == [short])

        let lines = (0..<6).map { String(repeating: "x", count: 99) + "\($0)" }
        let long = lines.joined(separator: "\n")
        let chunks = AnalysisEngine.chunkText(long, targetChars: 300)
        // Each line is ~101 bytes incl. the rejoined newline: 3 lines exceed
        // 300, so chunks are pairs — 3 of them.
        #expect(chunks.count == 3)
        #expect(chunks.allSatisfy { $0.components(separatedBy: "\n").count == 2 })
        #expect(chunks[0].components(separatedBy: "\n").first == lines[0])
        #expect(chunks[2].components(separatedBy: "\n").last == lines[5])
    }

    // MARK: prompt content markers (test_ai.py:58-115)

    @Test("the Essentials augmentation carries its load-bearing markers")
    func essentialsMarkers() {
        let augmented = AnalysisPrompts.augmentPromptEssentials(AnalysisPrompts.summary)
        #expect(augmented.contains("## Essentials"))
        #expect(augmented.contains("FUN:"))
        #expect(augmented.contains("REJECTED:"))
        #expect(augmented.contains("OPEN:"))
        // Inserted after the transcript so the model reads it last.
        #expect(augmented.contains("{transcript}\n\n---"))
    }

    @Test("the plan prompt has every required section and the classify prompt its tokens")
    func promptSections() {
        for section in ["## Overview", "## Goal", "## Proposed approach", "## Steps",
                        "## Risks", "## Open questions", "## Acceptance criteria"] {
            #expect(AnalysisPrompts.plan.contains(section), "plan is missing \(section)")
        }
        for token in ["MEETING", "PLAN", "WALKTHROUGH"] {
            #expect(AnalysisPrompts.classify.contains(token))
        }
        #expect(AnalysisPrompts.synthPrompt.contains("Deduplicate the Open questions"))
        // The plan task label names the speaker rule and the dedup rule.
        #expect(AnalysisPrompts.AIPrompt.plan.taskLabel.contains("named speaker"))
        #expect(AnalysisPrompts.AIPrompt.plan.taskLabel.contains("deduplicated"))
    }

    @Test("task labels for built-ins and the custom fallback")
    func taskLabels() {
        #expect(AnalysisPrompts.AIPrompt.summaryAndActions.isBuiltIn)
        #expect(!AnalysisPrompts.AIPrompt.custom("Analyze this").isBuiltIn)
        #expect(AnalysisPrompts.AIPrompt.custom("Analyze this").taskLabel.contains("verbatim"))
    }

    // MARK: auto-detection (test_ai.py:379-470)

    @Test("auto-detection routes by the classifier's token")
    func autoDetectRoutes() async {
        let plan = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(responses: ["PLAN"]))
        #expect(plan.prompt == .plan)
        #expect(plan.mode == "plan")

        let walkthrough = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(responses: ["WALKTHROUGH"]))
        #expect(walkthrough.prompt.template == AnalysisPrompts.walkthrough)
        #expect(walkthrough.mode == "walkthrough")

        let meeting = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(responses: ["MEETING"]))
        #expect(meeting.prompt == .summaryAndActions)

        // Lowercase and garbled replies both route to the safe default, and
        // "WALKTHROUGH" is not misread as containing "PLAN".
        let lowercase = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(responses: ["plan"]))
        #expect(lowercase.mode == "plan")
        let garbled = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(responses: ["I am not sure"]))
        #expect(garbled.mode == "meeting")
        let failing = await AnalysisEngine.resolvePromptAuto(
            transcript: "t", client: MockChatClient(failures: [AnalysisError.noChoices("x")]))
        #expect(failing.prompt == .summaryAndActions)
        #expect(failing.mode == "meeting (fallback)")
    }

    // MARK: analyze orchestration

    @Test("a short text transcript is one call with the Essentials instruction")
    func analyzeShortSingleCall() async throws {
        let client = MockChatClient(responses: ["SUMMARY"])
        let response = try await AnalysisEngine.analyze(
            prompt: .summaryAndActions, transcript: "hello world",
            client: client)
        let calls = await client.calls
        #expect(response == "SUMMARY")
        #expect(calls.count == 1)
        #expect(calls[0].frames.isEmpty)
        #expect(calls[0].prompt.contains("hello world"))
        #expect(calls[0].prompt.contains("## Essentials"))
        #expect(!calls[0].prompt.contains("{transcript}"))
    }

    @Test("a long text transcript runs rolling-context map-reduce")
    func analyzeLongMapReduce() async throws {
        // 12 lines × ~148 chars, target 600 ⇒ 3 chunks of 4 lines, exactly.
        let lines = (0..<12).map { String(repeating: "x", count: 140) + " line \($0)" }
        let long = lines.joined(separator: "\n")
        let client = MockChatClient(responses: ["C1", "C2", "C3", "FINAL"])
        let response = try await AnalysisEngine.analyze(
            prompt: .summaryAndActions, transcript: long, chunkChars: 600, client: client)
        let calls = await client.calls

        #expect(response == "FINAL")
        #expect(calls.count == 4)
        // Map prompts are wrapped: chunk k of n.
        #expect(calls[0].prompt.contains("chunk 1 of 3"))
        #expect(calls[2].prompt.contains("chunk 3 of 3"))
        // Rolling context: chunk 2 sees chunk 1's partial.
        #expect(!calls[0].prompt.contains("C1"))
        #expect(calls[1].prompt.contains("C1"))
        // The synth receives all parts and the task label.
        #expect(calls[3].prompt.contains("### Part 1 of 3\nC1"))
        #expect(calls[3].prompt.contains("### Part 3 of 3\nC3"))
        #expect(calls[3].prompt.contains("combining 3 partial analyses"))
    }

    @Test("the rolling context window caps at contextTurns")
    func rollingContextWindow() async throws {
        // 12 lines at target 600 ⇒ 3 chunks; window of 1 keeps chunk 3 seeing
        // C2 but not C1.
        let lines = (0..<12).map { String(repeating: "y", count: 140) + " line \($0)" }
        let client = MockChatClient(responses: ["C1", "C2", "C3", "F"])
        _ = try? await AnalysisEngine.analyze(
            prompt: .summaryAndActions, transcript: lines.joined(separator: "\n"),
            chunkChars: 600, contextTurns: 1, client: client)
        let calls = await client.calls
        // Window of 1: chunk 3 sees C2 but not C1.
        #expect(calls[2].prompt.contains("C2"))
        #expect(!calls[2].prompt.contains("C1\n### Part 2"))
    }

    @Test("custom prompts apply verbatim per chunk with the generic reduce")
    func analyzeCustomRollingContext() async throws {
        let lines = (0..<12).map { String(repeating: "z", count: 140) + " \($0)" }
        let client = MockChatClient(responses: ["A1", "A2", "MERGED"])
        let response = try await AnalysisEngine.analyze(
            prompt: .custom("What is the mood of {transcript}?"),
            transcript: lines.joined(separator: "\n"), chunkChars: 600, client: client)
        let calls = await client.calls

        #expect(response == "MERGED")
        #expect(calls.count == 4)   // 3 maps + the synth
        // No MAP_PROMPT wrapper for custom prompts — the text is verbatim.
        #expect(calls[0].prompt.contains("What is the mood of"))
        #expect(!calls[0].prompt.contains("analyzing chunk"))
        #expect(calls[1].prompt.contains("A1"))
        // The reduce is the custom one plus the Essentials merge instruction.
        #expect(calls[3].prompt.contains("Merge ALL of those Essentials bullets"))
    }

    @Test("contextTurns zero disables the rolling context")
    func analyzeNoContextWhenDisabled() async throws {
        let lines = (0..<12).map { String(repeating: "w", count: 140) + " \($0)" }
        let client = MockChatClient(responses: ["A1", "A2", "M"])
        _ = try? await AnalysisEngine.analyze(
            prompt: .custom("Sum {transcript}"),
            transcript: lines.joined(separator: "\n"), chunkChars: 600,
            contextTurns: 0, client: client)
        let calls = await client.calls
        #expect(!calls[1].prompt.contains("A1"))
        #expect(!calls[1].prompt.contains("Running context"))
    }

    // MARK: frames + vision

    @Test("the frame manifest labels frames in reading order")
    func frameManifestLabels() {
        let entries = [
            entry(1.0, "Speaker A", text: "first", frame: "seg0001.jpg"),
            entry(4.0, "Speaker B", text: "no frame"),
            entry(9.0, "Alice", text: "second", frame: "seg0003.jpg"),
        ]
        let manifest = AnalysisEngine.frameManifest(entries)
        #expect(manifest.hasPrefix("Frame timeline (2 frames, in time order):\n"))
        #expect(manifest.contains("  Frame 1: [00:00:01] Speaker A\n"))
        #expect(manifest.contains("  Frame 2: [00:00:09] Alice\n"))
        // Single-frame grammar.
        let single = AnalysisEngine.frameManifest([entries[0]])
        #expect(single.contains("(1 frame,"))
        #expect(AnalysisEngine.frameManifest([entries[1]]).isEmpty)
    }

    @Test("a short vision call prefixes the frame timeline and sends the frames")
    func analyzeShortVisionSingleCall() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-ai-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let entries = [
            entry(1.0, "Speaker A", text: "hello", frame: "seg0001.jpg"),
            entry(4.0, "Speaker B", text: "again", frame: "seg0002.jpg"),
        ]
        for name in ["seg0001.jpg", "seg0002.jpg"] {
            try Data([0xFF, 0xD8]).write(to: dir.appendingPathComponent(name))
        }
        let client = MockChatClient(responses: ["VISION"])

        let response = try await AnalysisEngine.analyze(
            prompt: .summaryAndActions, transcript: "hello\nagain",
            entries: entries, framesDir: dir, useVision: true, client: client)
        let calls = await client.calls

        #expect(response == "VISION")
        #expect(calls.count == 1)
        #expect(calls[0].prompt.hasPrefix("Frame timeline (2 frames,"))
        #expect(calls[0].frames.map(\.lastPathComponent) == ["seg0001.jpg", "seg0002.jpg"])
        #expect(calls[0].prompt.contains("hello"))
    }

    @Test("a long vision run maps chunks with only their own frames")
    func analyzeLongVisionPerChunk() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-ai-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        var entries: [FrameExtractor.Entry] = []
        for i in 0..<6 {
            let name = String(format: "seg%04d.jpg", i + 1)
            try Data([0xFF, 0xD8]).write(to: dir.appendingPathComponent(name))
            entries.append(FrameExtractor.Entry(
                index: i + 1, start: Double(i) * 10, end: Double(i) * 10 + 5,
                speaker: "Speaker A", text: "seg \(i)", frame: name, ocr: ""))
        }
        let client = MockChatClient(responses: ["P1", "P2", "P3", "FINAL"])
        let response = try await AnalysisEngine.analyze(
            prompt: .plan, transcript: AnalysisEngine.transcriptText(entries: entries),
            entries: entries, framesDir: dir, useVision: true,
            chunkSize: 2, client: client)
        let calls = await client.calls

        #expect(response == "FINAL")
        #expect(calls.count == 4)
        // Each map call carries only its chunk's frames.
        #expect(calls[0].frames.count == 2)
        #expect(calls[0].frames.map(\.lastPathComponent) == ["seg0001.jpg", "seg0002.jpg"])
        #expect(calls[1].frames.map(\.lastPathComponent) == ["seg0003.jpg", "seg0004.jpg"])
        #expect(calls[2].frames.map(\.lastPathComponent) == ["seg0005.jpg", "seg0006.jpg"])
        // Manifest is per-chunk.
        #expect(calls[1].prompt.contains("chunk 2 of 3"))
        #expect(!calls[1].prompt.contains("Frame 3"))
        // The synth is text-only.
        #expect(calls[3].frames.isEmpty)
    }

    @Test("progress messages fire for maps and the synth")
    func progressInvoked() async throws {
        let lines = (0..<12).map { String(repeating: "p", count: 140) + " \($0)" }
        let recorder = ProgressRecorder()
        let client = MockChatClient(responses: ["C1", "C2", "C3", "F"])
        _ = try? await AnalysisEngine.analyze(
            prompt: .summary, transcript: lines.joined(separator: "\n"),
            chunkChars: 600, client: client,
            onProgress: { recorder.record($0) })
        let messages = recorder.latest
        #expect(messages.contains("analyzing chunk 1/3"))
        #expect(messages.contains("analyzing chunk 2/3"))
        #expect(messages.contains("synthesizing 3 partial analyses"))
    }

    // MARK: vision gate + settings + report

    @Test("the vision heuristic matches the Python token list")
    func visionHeuristic() {
        #expect(AnalysisEngine.looksVisionCapable("llava:13b"))
        #expect(AnalysisEngine.looksVisionCapable("qwen2.5-vl"))
        #expect(!AnalysisEngine.looksVisionCapable("mistral-small"))

        let on = AnalysisEngine.resolveVision(hasFrames: true, model: "llava")
        #expect(on.useVision)
        let off = AnalysisEngine.resolveVision(hasFrames: true, model: "mistral-small")
        #expect(!off.useVision)
        #expect(off.message.contains("text-only"))
        let none = AnalysisEngine.resolveVision(hasFrames: false, model: "llava")
        #expect(!none.useVision && none.message.isEmpty)
    }

    @Test("AI settings parse from the shared config, defaults mirror config.py")
    func aiSettingsParseAndDefaults() {
        let defaults = BatchSettings.from([:])
        #expect(defaults.aiBaseURL == "http://localhost:11434/v1")
        #expect(defaults.aiModel.isEmpty)
        #expect(defaults.aiAPIKey.isEmpty)
        #expect(defaults.aiMaxFrames == 50)

        let parsed = BatchSettings.from(FlatTOML.parse("""
            ai_base_url = "https://api.example.com/v1"
            ai_model = "qwen2.5-vl"
            ai_api_key = "sk-test"
            ai_max_frames = 10
            """))
        #expect(parsed.aiBaseURL == "https://api.example.com/v1")
        #expect(parsed.aiModel == "qwen2.5-vl")
        #expect(parsed.aiAPIKey == "sk-test")
        #expect(parsed.aiMaxFrames == 10)
    }

    @Test("the report artifact mirrors cmd_analyze's format")
    func reportMarkdown() {
        let report = AnalysisEngine.reportMarkdown(
            inputName: "clip.mp4", model: "llava", vision: true, mode: "meeting",
            promptTemplate: AnalysisPrompts.summary, response: "It was a meeting.")
        #expect(report.hasPrefix("# whiz analysis — clip.mp4\n\n"))
        #expect(report.contains("**Model:** llava  **Vision:** true  **Mode:** meeting"))
        #expect(report.contains("<transcript omitted>"))
        #expect(report.contains("## Response\n\nIt was a meeting."))
    }
}

/// Lock-guarded collection point for `@Sendable` progress callbacks in tests.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    func record(_ value: String) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }

    var latest: [String] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}