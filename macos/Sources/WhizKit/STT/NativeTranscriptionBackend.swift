import Foundation

/// Read-only view of the pipeline keys in `~/.config/whiz/config.toml`.
///
/// `WhizConfig` owns only the `dictate_*` keys — deliberately, because its
/// `save()` is a read-modify-write that must never touch pipeline keys. The
/// batch pipeline needs the other half of the same file (the keys
/// `whiz/config.py` defines for `whiz transcribe`), so this type reads them
/// without ever writing: defaults mirror `whiz/config.py` exactly, and drift
/// in either direction is pinned by `TranscriptionFlowTests`.
struct BatchSettings {

    /// `config.py:model` — explicit model path or bare filename, "" => auto.
    var model: String = ""
    /// `config.py:language` — BCP-47 code or "auto".
    var language: String = "auto"
    /// `config.py:vad`.
    var vad: Bool = true
    /// `config.py:vad_model` — "" => auto-discover.
    var vadModel: String = ""
    /// `config.py:vad_threshold`.
    var vadThreshold: Double = 0.5

    static func load() -> BatchSettings {
        guard let text = try? String(contentsOf: WhizConfig.path, encoding: .utf8) else {
            return BatchSettings()
        }
        return from(FlatTOML.parse(text))
    }

    static func from(_ values: [String: FlatTOML.Value]) -> BatchSettings {
        var s = BatchSettings()
        if case .string(let v)? = values["model"] { s.model = v }
        if case .string(let v)? = values["language"] { s.language = v }
        if case .bool(let v)? = values["vad"] { s.vad = v }
        if case .string(let v)? = values["vad_model"] { s.vadModel = v }
        if case .double(let v)? = values["vad_threshold"] { s.vadThreshold = v }
        if case .int(let v)? = values["vad_threshold"] { s.vadThreshold = Double(v) }
        return s
    }
}

/// The real `TranscriptionBackend`: `AudioFileDecoder` →
/// `WhisperBatchTranscriber` (beam-search profile, built-in VAD) →
/// `TranscriptFormatter` outputs, mirroring what `whiz transcribe` runs
/// through whisper-cli — including the pipeline keys from the shared config.
///
/// Model lifetime is one run: load, transcribe, unload — unlike dictation,
/// which keeps the model warm, a batch file finishes and the ~1 GB of model
/// RAM should go back to the system. VAD mirrors the Python default
/// (`config.py: vad = True`): on with an auto-discovered Silero model, and
/// if none is installed the run proceeds without VAD after a logged warning,
/// exactly like `_build_transcribe_args` warns.
struct NativeTranscriptionBackend: TranscriptionBackend {

    var settings: BatchSettings

    init(settings: BatchSettings = .load()) {
        self.settings = settings
    }

    func transcribe(
        input: URL,
        outputDirectory: URL,
        onEvent: @escaping @Sendable (TranscriptionEvent) -> Void
    ) async throws -> URL {
        func log(_ text: String) {
            onEvent(.log(text))
        }

        // 1. Decode. Progress within the decode is reported against the
        // container's own duration; the phases below map it into the 0…0.12
        // band.
        onEvent(.phase("Decoding audio"))
        log("input: \(input.lastPathComponent)")
        let audio = try await AudioFileDecoder.extractSamples(at: input) { fraction in
            onEvent(.progress(0.12 * fraction))
        }
        log(String(
            format: "audio: 16 kHz mono · %d samples (%.1f s)",
            audio.samples.count, audio.duration))
        onEvent(.progress(0.12))

        // 2. Model — the batch pipeline's own preference order, mirroring
        // `models.py:PREFERENCE` (q5_0 turbo first), NOT dictation's.
        guard let modelURL = WhisperModel.resolveBatch(configured: settings.model) else {
            throw WhisperError.noModelFound
        }
        onEvent(.phase("Loading model"))
        log("model: \(modelURL.lastPathComponent)")
        let engine = WhisperBatchTranscriber(modelURL: modelURL)
        try await engine.load()
        onEvent(.progress(0.15))

        // 3. Transcribe, streaming segments into the log as they land.
        onEvent(.phase("Transcribing"))
        log("language: \(settings.language), VAD: \(settings.vad ? "on" : "off")")
        var vadModel: URL?
        if settings.vad {
            vadModel = resolveVAD()
            if vadModel == nil {
                log("warning: no Silero VAD model found — transcribing without VAD (whiz models download-vad)")
            }
        }
        let segments: [WhisperBatchTranscriber.Segment]
        do {
            segments = try await withTaskCancellationHandler {
                try await engine.transcribe(
                    samples: audio.samples,
                    language: settings.language,
                    vadModelPath: vadModel,
                    vadThreshold: Float(settings.vadThreshold),
                    onProgress: { fraction in
                        onEvent(.progress(0.15 + 0.80 * fraction))
                    },
                    onSegment: { segment in
                        onEvent(.log(TranscriptFormatter.segmentLogLine(segment)))
                    }
                )
            } onCancel: {
                Task { await engine.cancel() }
            }
        } catch WhisperBatchError.cancelled {
            await engine.unload()
            throw CancellationError()
        } catch {
            await engine.unload()
            throw error
        }
        await engine.unload()
        onEvent(.progress(0.95))

        // 4. Outputs — SRT + JSON, mirroring `config.py:outputs` default.
        // Output naming follows the `-of <stem>` convention: the input's stem
        // inside the run's directory.
        onEvent(.phase("Writing outputs"))
        let stem = input.deletingPathExtension().lastPathComponent
        try FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)
        let srtURL = outputDirectory.appendingPathComponent("\(stem).srt")
        try TranscriptFormatter.srt(segments).write(
            to: srtURL, atomically: true, encoding: .utf8)
        log("output: \(srtURL.path)")
        let jsonURL = outputDirectory.appendingPathComponent("\(stem).json")
        try TranscriptFormatter.json(segments).write(
            to: jsonURL, atomically: true, encoding: .utf8)
        log("output: \(jsonURL.path)")

        // 5. Frames — auto-on for video, mirroring cli.py:676 and cli.py:720:
        // one JPEG per segment plus the manifest, labeled with the same
        // generic "Speaker" the Python no-diarization path uses (cli.py:722)
        // until diarization lands. Manifest shape is shared with Python's
        // `load_manifest`, so `whiz analyze` and the future HTML pass read
        // either side's output.
        let framesDir = outputDirectory.appendingPathComponent("\(stem).frames")
        let labeled = segments.map { LabeledSegment(segment: $0, speaker: "Speaker") }
        if segments.isEmpty {
            log("frames: skipped — no segments to capture")
        } else if await !FrameExtractor.hasVideoTrack(input) {
            log("frames: skipped — no video track")
        } else {
            onEvent(.phase("Capturing frames"))
            let entries = try await FrameExtractor.extractFrames(
                video: input,
                segments: labeled,
                into: framesDir,
                onProgress: { fraction in
                    onEvent(.progress(0.95 + 0.04 * fraction))
                })
            // Written even when some or all captures failed — the manifest
            // aligns by index with empty `frame` fields, exactly like
            // write_manifest's contract.
            let manifestURL = outputDirectory.appendingPathComponent("\(stem).frames.json")
            try FrameExtractor.writeManifest(entries, framesDir: framesDir, to: manifestURL)
            let captured = entries.filter { !$0.frame.isEmpty }.count
            log(String(format: "frames: %d/%d extracted", captured, entries.count))
            log("output: \(manifestURL.path)")
        }

        // 6. The readable artifacts — the self-contained HTML transcript (frames
        // inlined as base64 when present) and the dialogue TXT, built from the
        // same labeled segments as the frames manifest. A documented divergence
        // from the Python pipeline, which writes these only on the diarized
        // path: there are no speaker labels yet, so every segment carries the
        // same generic "Speaker", and real labels arrive with diarization
        // through this exact shape. The labeled SRT is ported and pinned in
        // TranscriptMergeTests but deliberately not written here — with one
        // generic speaker it would only duplicate the plain SRT.
        if !segments.isEmpty {
            onEvent(.phase("Writing HTML transcript"))
            let htmlURL = outputDirectory.appendingPathComponent("\(stem).speakers.html")
            try SpeakersHTML.format(labeled, framesDir: framesDir, title: input.lastPathComponent)
                .write(to: htmlURL, atomically: true, encoding: .utf8)
            log("output: \(htmlURL.path)")

            let txtURL = outputDirectory.appendingPathComponent("\(stem).speakers.txt")
            try LabeledTranscript.formatDialogueTXT(labeled)
                .write(to: txtURL, atomically: true, encoding: .utf8)
            log("output: \(txtURL.path)")
        }

        onEvent(.phase("Finished"))
        onEvent(.progress(1))
        return outputDirectory
    }

    /// `config.py:vad_model` set => that path (tilde-expanded), else the
    /// search-order discovery shared with dictation.
    private func resolveVAD() -> URL? {
        if !settings.vadModel.isEmpty {
            let expanded = (settings.vadModel as NSString).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return WhisperModel.resolveVAD()
    }
}