import Foundation

/// The mechanics of `whiz/ai.py`: the chat transport (`_post_chat` +
/// `chat_text`/`chat_vision` collapsed into one protocol method), chunking,
/// rolling-context map-reduce (`analyze`), auto-detection
/// (`resolve_prompt_auto`), and the server utilities. Prompt text lives in
/// `AnalysisPrompts`.
///
/// The transport is a protocol so the orchestration is testable without
/// network — the production `HTTPChatClient` mirrors `_post_chat`'s retry
/// contract exactly (429/5xx/connection errors retry with 2s/4s backoff,
/// other errors fail fast with the body prefix and, for vision requests
/// rejected by a text-only model, the is-it-vision-capable hint).
enum AnalysisEngine {

    // MARK: - Transport

    /// One completion call. `frames` empty = text-only; otherwise the client
    /// subsamples to `maxFrames`, skips missing files, and base64-encodes at
    /// send time so on-disk artifacts stay paths-only.
    protocol ChatClient: Sendable {
        func complete(prompt: String, frames: [URL], maxFrames: Int) async throws -> String
    }

    /// `_post_chat` + `chat_vision` over URLSession.
    struct HTTPChatClient: ChatClient {
        let baseURL: String
        let model: String
        let apiKey: String
        var timeout: TimeInterval = 600
        var retryMax = 3
        var retryBaseDelay: TimeInterval = 2.0
        /// Injectable so tests retry instantly instead of sleeping.
        var backoff: @Sendable (TimeInterval) async -> Void = { seconds in
            try? await Task.sleep(for: .milliseconds(Int(seconds * 1000)))
        }
        var session: URLSession = .shared

        private static let retryStatus: Set<Int> = [429, 500, 502, 503, 504]

        func complete(prompt: String, frames: [URL], maxFrames: Int) async throws -> String {
            // The OpenAI vision content shape, identical for Ollama's
            // /v1/chat/completions and any compatible server.
            var content: Any
            if frames.isEmpty {
                content = prompt
            } else {
                var parts: [[String: Any]] = [["type": "text", "text": prompt]]
                for url in AnalysisEngine.subsample(frames, maxFrames: maxFrames) {
                    guard let raw = try? Data(contentsOf: url), !raw.isEmpty else { continue }
                    parts.append([
                        "type": "image_url",
                        "image_url": ["url": "data:image/jpeg;base64,\(raw.base64EncodedString())"],
                    ])
                }
                content = parts
            }
            let body: [String: Any] = [
                "model": model,
                "messages": [["role": "user", "content": content]],
                "stream": false,
                "temperature": 0.3,
            ]
            let payload = try JSONSerialization.data(withJSONObject: body)

            let trimmed = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
            let urlString = trimmed + "/chat/completions"
            guard let endpoint = URL(string: urlString) else {
                throw AnalysisError.unreachable(urlString, "invalid URL")
            }
            var request = URLRequest(url: endpoint)
            request.httpMethod = "POST"
            request.timeoutInterval = timeout
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            request.httpBody = payload

            let data = try await send(request, urlString: urlString, frames: frames)

            // OpenAI shape: {"choices": [{"message": {"content": "..."}}]}
            guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let choices = object["choices"] as? [[String: Any]], !choices.isEmpty
            else {
                throw AnalysisError.noChoices(String(decoding: data.prefix(300), as: UTF8.self))
            }
            let message = choices[0]["message"] as? [String: Any]
            return ((message?["content"] as? String) ?? "").trimmingCharacters(
                in: .whitespacesAndNewlines)
        }

        /// _post_chat's retry loop: 429/5xx retry with 2s/4s backoff while
        /// attempts remain, everything else fails fast. Connection errors retry
        /// too; a classified AnalysisError never re-enters the loop.
        private func send(
            _ request: URLRequest,
            urlString: String,
            frames: [URL]
        ) async throws -> Data {
            let attempts = max(retryMax, 1)
            for attempt in 0..<attempts {
                do {
                    let (data, response) = try await session.data(for: request)
                    guard let http = response as? HTTPURLResponse else { return data }
                    if (200..<300).contains(http.statusCode) {
                        return data
                    }
                    if Self.retryStatus.contains(http.statusCode), attempt < attempts - 1 {
                        await backoff(retryBaseDelay * pow(2, Double(attempt)))
                        continue
                    }
                    var hint = ""
                    if [400, 404].contains(http.statusCode), !frames.isEmpty {
                        hint = "\nHint: this looks like a vision request. "
                            + "Is the configured model vision-capable? "
                            + "(e.g. llava, qwen2.5-vl, minicpm-v). "
                            + "A text-only model will reject image inputs."
                    }
                    throw AnalysisError.http(
                        status: http.statusCode, url: urlString,
                        body: String(decoding: data.prefix(500), as: UTF8.self), hint: hint)
                } catch let error as AnalysisError {
                    throw error
                } catch {
                    guard attempt < attempts - 1 else {
                        throw AnalysisError.unreachable(
                            urlString,
                            (error as? URLError)?.localizedDescription
                                ?? "connection failed")
                    }
                    await backoff(retryBaseDelay * pow(2, Double(attempt)))
                }
            }
            throw AnalysisError.unreachable(urlString, "retries exhausted")
        }
    }

    // MARK: - Transcript rendering

    /// ai.py:transcript_text for manifest rows.
    static func transcriptText(entries: [FrameExtractor.Entry]) -> String {
        entries
            .map { "[\(TranscriptFormatter.clock($0.start))] \($0.speaker): \($0.text)" }
            .joined(separator: "\n")
    }

    /// ai.py:transcript_text for (WhisperSeg, label) pairs.
    static func transcriptText(labeled: [LabeledSegment]) -> String {
        labeled
            .map {
                "[\(TranscriptFormatter.clock($0.segment.start))] \($0.speaker): "
                    + $0.segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            .joined(separator: "\n")
    }

    // MARK: - Chunking

    /// ai.py:chunk_entries — contiguous sublists of at most `chunkSize`.
    static func chunkEntries<T>(_ entries: [T], chunkSize: Int) -> [[T]] {
        let size = max(1, chunkSize)
        guard !entries.isEmpty else { return [] }
        return stride(from: 0, to: entries.count, by: size).map {
            Array(entries[$0..<min($0 + size, entries.count)])
        }
    }

    /// ai.py:_chunk_text — split near `targetChars` on line boundaries so
    /// each chunk is a coherent set of segments.
    static func chunkText(_ text: String, targetChars: Int = 6000) -> [String] {
        if targetChars <= 0 || text.count <= targetChars {
            return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? [] : [text]
        }
        var chunks: [String] = []
        var buffer: [String] = []
        var size = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let lineLength = line.count + 1   // +1 for the rejoined "\n"
            if !buffer.isEmpty && size + lineLength > targetChars {
                chunks.append(buffer.joined(separator: "\n"))
                buffer = []
                size = 0
            }
            buffer.append(String(line))
            size += lineLength
        }
        if !buffer.isEmpty {
            chunks.append(buffer.joined(separator: "\n"))
        }
        return chunks.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    // MARK: - Frames

    /// ai.py:_frames_for_entries.
    static func framesForEntries(_ entries: [FrameExtractor.Entry], framesDir: URL?) -> [URL] {
        guard let framesDir else { return [] }
        return entries.compactMap { entry in
            entry.frame.isEmpty ? nil : framesDir.appendingPathComponent(entry.frame)
        }
    }

    /// ai.py:_frame_manifest — a text timeline so the vision model can treat
    /// the unordered image array as a sequence, not a bag of screenshots.
    static func frameManifest(_ entries: [FrameExtractor.Entry]) -> String {
        var lines: [String] = []
        for entry in entries where !entry.frame.isEmpty {
            lines.append("  Frame \(lines.count + 1): [\(TranscriptFormatter.clock(entry.start))] \(entry.speaker)".rstrip())
        }
        guard !lines.isEmpty else { return "" }
        return "Frame timeline (\(lines.count) frame\(lines.count == 1 ? "" : "s"), in time order):\n"
            + lines.joined(separator: "\n") + "\n\n"
    }

    /// ai.py:_subsample — evenly spread to at most `maxFrames`.
    static func subsample<T>(_ items: [T], maxFrames: Int) -> [T] {
        if maxFrames <= 0 || items.count <= maxFrames {
            return items
        }
        if maxFrames == 1 {
            return [items[items.count / 2]]
        }
        let step = Double(items.count) / Double(maxFrames)
        var seen = Set<Int>()
        var out: [T] = []
        for index in 0..<maxFrames {
            let idx = Int(Double(index) * step)
            if !seen.contains(idx) {
                seen.insert(idx)
                out.append(items[idx])
            }
        }
        return out
    }

    // MARK: - Map-reduce

    /// ai.py:analyze. Short inputs use one call — identical to the unchunked
    /// behavior. Long inputs are split into contiguous chunks; the map phase
    /// is rolling-context (each chunk carries the prior partials, windowed
    /// to `contextTurns`), and the reduce step merges the coherent partials.
    /// Essentials is always on (ai.py:analyze docstring).
    static func analyze(
        prompt: AnalysisPrompts.AIPrompt,
        transcript: String,
        entries: [FrameExtractor.Entry]? = nil,
        framesDir: URL? = nil,
        useVision: Bool = false,
        maxFrames: Int = 50,
        chunkSize: Int = 8,
        chunkChars: Int = 6000,
        contextTurns: Int = 3,
        client: ChatClient,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        let builtIn = prompt.isBuiltIn
        let task = prompt.taskLabel + AnalysisPrompts.essentialsTaskSuffix
        let template = AnalysisPrompts.augmentPromptEssentials(prompt.template)

        if useVision, let entries, !entries.isEmpty {
            let chunks = chunkEntries(entries, chunkSize: chunkSize)
            if chunks.count <= 1 {
                let framePaths = framesForEntries(entries, framesDir: framesDir)
                let manifest = frameManifest(entries)
                let visionPrompt = manifest.isEmpty ? template : manifest + template
                return try await client.complete(
                    prompt: visionPrompt.replacingOccurrences(of: "{transcript}", with: transcript),
                    frames: framePaths, maxFrames: maxFrames)
            }
            return try await mapReduceVision(
                chunks: chunks, task: task, builtIn: builtIn, template: template,
                framesDir: framesDir, maxFrames: maxFrames, contextTurns: contextTurns,
                client: client, onProgress: onProgress)
        }

        let chunks = chunkText(transcript, targetChars: chunkChars)
        if chunks.count <= 1 {
            return try await client.complete(
                prompt: template.replacingOccurrences(of: "{transcript}", with: transcript),
                frames: [], maxFrames: maxFrames)
        }
        return try await mapReduceText(
            chunks: chunks, task: task, builtIn: builtIn, template: template,
            contextTurns: contextTurns, client: client, onProgress: onProgress)
    }

    private static func mapReduceText(
        chunks: [String],
        task: String,
        builtIn: Bool,
        template: String,
        contextTurns: Int,
        client: ChatClient,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> String {
        let n = chunks.count
        var partials: [String] = []
        for (offset, chunk) in chunks.enumerated() {
            let k = offset + 1
            onProgress?("analyzing chunk \(k)/\(n)")
            let context = runningContext(partials, contextTurns: contextTurns)
            let mapPrompt: String
            if builtIn {
                mapPrompt = AnalysisPrompts.mapPrompt
                    .replacingOccurrences(of: "{task}", with: task)
                    .replacingOccurrences(of: "{k}", with: "\(k)")
                    .replacingOccurrences(of: "{n}", with: "\(n)")
                    .replacingOccurrences(of: "{context_block}", with: context)
                    .replacingOccurrences(of: "{transcript}", with: chunk)
            } else {
                mapPrompt = context + template.replacingOccurrences(of: "{transcript}", with: chunk)
            }
            let partial = try await client.complete(prompt: mapPrompt, frames: [], maxFrames: 0)
            partials.append("### Part \(k) of \(n)\n\(partial)")
        }
        return try await synthesize(
            partials: partials, task: task, builtIn: builtIn, n: n, client: client,
            onProgress: onProgress)
    }

    private static func mapReduceVision(
        chunks: [[FrameExtractor.Entry]],
        task: String,
        builtIn: Bool,
        template: String,
        framesDir: URL?,
        maxFrames: Int,
        contextTurns: Int,
        client: ChatClient,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> String {
        let n = chunks.count
        var partials: [String] = []
        for (offset, chunk) in chunks.enumerated() {
            let k = offset + 1
            let chunkTranscript = transcriptText(entries: chunk)
            let frames = framesForEntries(chunk, framesDir: framesDir)
            let manifest = frameManifest(chunk)
            onProgress?("analyzing chunk \(k)/\(n) (\(chunk.count) segments, \(frames.count) frames)")
            let context = runningContext(partials, contextTurns: contextTurns)
            var mapPrompt: String
            if builtIn {
                mapPrompt = AnalysisPrompts.mapPrompt
                    .replacingOccurrences(of: "{task}", with: task)
                    .replacingOccurrences(of: "{k}", with: "\(k)")
                    .replacingOccurrences(of: "{n}", with: "\(n)")
                    .replacingOccurrences(of: "{context_block}", with: context)
                    .replacingOccurrences(of: "{transcript}", with: chunkTranscript)
            } else {
                mapPrompt = context + template.replacingOccurrences(of: "{transcript}", with: chunkTranscript)
            }
            if !manifest.isEmpty {
                mapPrompt = manifest + mapPrompt
            }
            let partial = try await client.complete(
                prompt: mapPrompt, frames: frames, maxFrames: maxFrames)
            partials.append("### Part \(k) of \(n)\n\(partial)")
        }
        return try await synthesize(
            partials: partials, task: task, builtIn: builtIn, n: n, client: client,
            onProgress: onProgress)
    }

    private static func synthesize(
        partials: [String],
        task: String,
        builtIn: Bool,
        n: Int,
        client: ChatClient,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> String {
        onProgress?("synthesizing \(n) partial analyses")
        let reducePrompt = builtIn
            ? AnalysisPrompts.synthPrompt
            : AnalysisPrompts.customReducePrompt + AnalysisPrompts.essentialsReduceInstruction
        let synth = reducePrompt
            .replacingOccurrences(of: "{task}", with: task)
            .replacingOccurrences(of: "{n}", with: "\(n)")
            .replacingOccurrences(of: "{partials}", with: partials.joined(separator: "\n\n"))
        return try await client.complete(prompt: synth, frames: [], maxFrames: 0)
    }

    /// ai.py:_running_context — the sliding window over prior partials.
    static func runningContext(_ partials: [String], contextTurns: Int) -> String {
        guard contextTurns > 0, !partials.isEmpty else { return "" }
        let window = partials.suffix(max(1, contextTurns))
        return AnalysisPrompts.contextBlock
            .replacingOccurrences(of: "{context}", with: window.joined(separator: "\n\n"))
    }

    // MARK: - Auto-detection

    /// ai.py:resolve_prompt_auto — one classify call, routed by the reply's
    /// first token. Any failure falls back to summary+actions with the
    /// "(fallback)" mode marker.
    static func resolvePromptAuto(
        transcript: String,
        client: ChatClient
    ) async -> (prompt: AnalysisPrompts.AIPrompt, mode: String) {
        do {
            let reply = try await client.complete(
                prompt: AnalysisPrompts.classify
                    .replacingOccurrences(of: "{transcript}", with: transcript),
                frames: [], maxFrames: 0)
            let token = reply
                .split(separator: "\n").first.map {
                    $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                } ?? ""
            // Prefer the most specific match so "WALKTHROUGH" is not misread
            // as containing "PLAN".
            if token.contains("WALKTHROUGH") {
                return (.custom(AnalysisPrompts.walkthrough), "walkthrough")
            }
            if token.contains("PLAN"), !token.contains("MEETING") {
                return (.plan, "plan")
            }
            if token.contains("MEETING") {
                return (.summaryAndActions, "meeting")
            }
        } catch {
            return (.summaryAndActions, "meeting (fallback)")
        }
        return (.summaryAndActions, "meeting")
    }

    // MARK: - Vision gate (cli.py:_resolve_vision + _looks_vision_capable)

    /// Substrings that signal a vision-capable model — cli.py:_VISION_TOKENS,
    /// the single source of truth for whether sending images is safe.
    static let visionTokens = [
        "vl", "vision", "llava", "minicpm-v", "qwen2.5-vl", "qwen-vl",
        "qwen3-vl", "qwen3.5", "multimodal", "gpt-4o", "gpt-4-vision",
        "llama-3.2-vision", "pixtral", "cogvlm", "internvl",
        "phi-3.5-vision", "phi-3-vision", "gemma3", "gemma4",
        "mistral-large-3", "minimax-m3", "kimi-k2.5", "kimi-k2.6",
        "kimi-k2.7",
    ]

    static func looksVisionCapable(_ model: String) -> Bool {
        let lower = model.lowercased()
        return visionTokens.contains { lower.contains($0) }
    }

    /// The app's slice of cli.py:_resolve_vision: no explicit flags exist in
    /// the menu flow, so this is the auto path — frames exist and the model
    /// looks vision-capable → enable; otherwise stay text-only rather than
    /// sending images a model might reject.
    static func resolveVision(hasFrames: Bool, model: String) -> (useVision: Bool, message: String) {
        guard hasFrames else { return (false, "") }
        if looksVisionCapable(model) {
            return (true, "frames found and '\(model)' is vision-capable — vision on")
        }
        return (false, "frames found but '\(model)' doesn't look vision-capable — staying text-only")
    }

    // MARK: - Server utilities

    /// ai.py:list_ollama_models — native /api/tags first, then the OpenAI
    /// /v1/models shape; empty on any failure.
    static func listModels(baseURL: String, session: URLSession = .shared) async -> [String] {
        var root = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        if root.hasSuffix("/v1") {
            root = String(root.dropLast(3))
        }
        if let names = await getModelNames(url: root + "/api/tags", session: session) {
            return names
        }
        let chatBase = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        return await getModelNames(url: chatBase + "/models", session: session) ?? []
    }

    private static func getModelNames(url: String, session: URLSession) async -> [String]? {
        guard let endpoint = URL(string: url) else { return nil }
        var request = URLRequest(url: endpoint)
        request.timeoutInterval = 10
        guard let (data, response) = try? await session.data(for: request),
              (response as? HTTPURLResponse)?.statusCode ?? 500 < 300,
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        if let models = object["models"] as? [[String: Any]] {
            let names = models.compactMap { ($0["name"] as? String).flatMap { $0.isEmpty ? nil : $0 } }
            if !names.isEmpty { return names }
        }
        if let data = object["data"] as? [[String: Any]] {
            let names = data.compactMap { ($0["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } }
            if !names.isEmpty { return names }
        }
        return nil
    }

    /// ai.py:probe_model — a trivial completion to check the configured model
    /// is actually usable (Ollama can list models that are retired
    /// server-side; the death only surfaces at call time).
    static func probeModel(client: ChatClient) async -> (ok: Bool, error: String) {
        do {
            _ = try await client.complete(prompt: "Reply with: ok", frames: [], maxFrames: 0)
            return (true, "")
        } catch {
            return (false, (error as? LocalizedError)?.errorDescription ?? String(describing: error))
        }
    }

    // MARK: - The .analysis.md artifact (cmd_analyze's format)

    /// cmd_analyze writes prompt + response to `<stem>.analysis.md`; the
    /// transcript itself is elided from the recorded prompt.
    static func reportMarkdown(
        inputName: String,
        model: String,
        vision: Bool,
        mode: String,
        promptTemplate: String,
        response: String
    ) -> String {
        let recordedPrompt = promptTemplate
            .replacingOccurrences(of: "{transcript}", with: "<transcript omitted>")
        return "# whiz analysis — \(inputName)\n\n"
            + "**Model:** \(model)  **Vision:** \(vision)  **Mode:** \(mode)\n\n"
            + "## Prompt\n\n```\n" + recordedPrompt + "\n```\n\n"
            + "## Response\n\n" + response + "\n"
    }
}

enum AnalysisError: LocalizedError, Sendable, Equatable {
    case http(status: Int, url: String, body: String, hint: String)
    case unreachable(String, String)
    case noChoices(String)

    var errorDescription: String? {
        switch self {
        case .http(let status, let url, let body, let hint):
            return "AI server returned HTTP \(status) for \(url).\nResponse: \(body)\(hint)"
        case .unreachable(let url, let reason):
            return "Could not reach AI server at \(url): \(reason)\n"
                + "Is Ollama running? Start it with:  ollama serve"
        case .noChoices(let detail):
            return "AI server returned no choices: \(detail)"
        }
    }
}

private extension String {
    /// Python's str.rstrip() — drop trailing whitespace.
    func rstrip() -> String {
        replacingOccurrences(
            of: "\\s+$", with: "", options: .regularExpression)
    }
}