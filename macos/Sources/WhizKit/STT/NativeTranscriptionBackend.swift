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

    // `config.py` diarization keys — note the default is OFF there too (video
    // inputs auto-enable, mirroring cli.py:_video_auto_flags).
    /// `config.py:diarize`.
    var diarize: Bool = false
    /// `config.py:num_speakers` — 0 => auto-detect via cluster threshold.
    var numSpeakers: Int = 0
    /// `config.py:cluster_threshold` — larger = fewer speakers (0.9 per
    /// sherpa-onnx guidance).
    var clusterThreshold: Double = 0.9
    /// `config.py:diarization_segmentation_model` — "" => auto-discover.
    var diarizationSegmentationModel: String = ""
    /// `config.py:diarization_embedding_model` — "" => auto-discover.
    var diarizationEmbeddingModel: String = ""

    /// `config.py:speaker_match_threshold` — 0.8 suits 3D-Speaker embeddings;
    /// higher = stricter (fewer auto-assignments).
    var speakerMatchThreshold: Double = 0.8
    /// `config.py:save_voice_profiles` — merge named speakers' embeddings
    /// into stored profiles after a run.
    var saveVoiceProfiles: Bool = true

    // `config.py` OCR keys — opt-in because OCR is the slowest pipeline stage
    // (one pass per segment frame); it never turns on automatically.
    /// `config.py:ocr`.
    var ocr: Bool = false
    /// `config.py:ocr_languages` — "en-US" style hints for Vision.
    var ocrLanguages: [String] = ["en-US"]
    /// `config.py:ocr_min_chars` — drop results shorter than this (noise from
    /// mostly-empty frames).
    var ocrMinChars: Int = 8
    /// `config.py:ocr_max_chars` — truncate one frame's OCR (guards against
    /// one pathological frame, not the whole prompt).
    var ocrMaxChars: Int = 4000
    /// `config.py:ocr_dedupe` — reuse OCR for byte-identical frames.
    var ocrDedupe: Bool = true
    /// `config.py:ocr_min_width` — small UI text doesn't survive the default
    /// 1280 downscale; raised automatically with a notice.
    var ocrMinWidth: Int = 1920

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
        if case .bool(let v)? = values["diarize"] { s.diarize = v }
        if case .int(let v)? = values["num_speakers"] { s.numSpeakers = v }
        if case .double(let v)? = values["cluster_threshold"] { s.clusterThreshold = v }
        if case .int(let v)? = values["cluster_threshold"] { s.clusterThreshold = Double(v) }
        if case .string(let v)? = values["diarization_segmentation_model"] { s.diarizationSegmentationModel = v }
        if case .string(let v)? = values["diarization_embedding_model"] { s.diarizationEmbeddingModel = v }
        if case .double(let v)? = values["speaker_match_threshold"] { s.speakerMatchThreshold = v }
        if case .int(let v)? = values["speaker_match_threshold"] { s.speakerMatchThreshold = Double(v) }
        if case .bool(let v)? = values["save_voice_profiles"] { s.saveVoiceProfiles = v }
        if case .bool(let v)? = values["ocr"] { s.ocr = v }
        if case .stringArray(let v)? = values["ocr_languages"] { s.ocrLanguages = v }
        if case .int(let v)? = values["ocr_min_chars"] { s.ocrMinChars = v }
        if case .int(let v)? = values["ocr_max_chars"] { s.ocrMaxChars = v }
        if case .bool(let v)? = values["ocr_dedupe"] { s.ocrDedupe = v }
        if case .int(let v)? = values["ocr_min_width"] { s.ocrMinWidth = v }
        // config.py:ocr_engine has no native counterpart — Vision is the only
        // engine on this platform, which is why the manifest records "apple".
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
        // container's own duration; the phases below map it into the 0…0.10
        // band.
        onEvent(.phase("Decoding audio"))
        log("input: \(input.lastPathComponent)")
        let audio = try await AudioFileDecoder.extractSamples(at: input) { fraction in
            onEvent(.progress(0.10 * fraction))
        }
        log(String(
            format: "audio: 16 kHz mono · %d samples (%.1f s)",
            audio.samples.count, audio.duration))
        onEvent(.progress(0.10))

        // 2. Diarization — Python's order (cli.py runs it before whisper) and
        // Python's auto-enable rule: on for video inputs even when `diarize`
        // is false in config (cli.py:_video_auto_flags), skipped with a hint
        // when the models are missing (cli.py:183). Without usable diarization
        // everything downstream falls back to the generic "Speaker" label.
        let hasVideo = await FrameExtractor.hasVideoTrack(input)
        var diarSegments: [DiarSegment] = []
        if settings.diarize || hasVideo {
            try Task.checkCancellation()
            if let segModel = DiarizationModel.findSegmentationModel(
                explicit: settings.diarizationSegmentationModel,
                searchDirectories: WhisperModel.searchDirectories),
               let embModel = DiarizationModel.findEmbeddingModel(
                explicit: settings.diarizationEmbeddingModel,
                searchDirectories: WhisperModel.searchDirectories)
            {
                onEvent(.phase("Diarizing speakers"))
                log("diarization: pyannote + eres2net · "
                    + "speakers: \(settings.numSpeakers > 0 ? String(settings.numSpeakers) : "auto"), "
                    + "threshold: \(settings.clusterThreshold)")
                diarSegments = try await Diarization.run(
                    samples: audio.samples,
                    segmentationModel: segModel,
                    embeddingModel: embModel,
                    numSpeakers: settings.numSpeakers,
                    threshold: Float(settings.clusterThreshold),
                    onProgress: { fraction in
                        onEvent(.progress(0.10 + 0.15 * fraction))
                    })
                let speakers = Set(diarSegments.map(\.speaker)).count
                log("diarization: \(diarSegments.count) segments, \(speakers) speaker(s)")
            } else {
                log("speakers: diarization models not found — skipped "
                    + "(whiz models download-diarization)")
            }
        }

        // 3. Model — the batch pipeline's own preference order, mirroring
        // `models.py:PREFERENCE` (q5_0 turbo first), NOT dictation's.
        guard let modelURL = WhisperModel.resolveBatch(configured: settings.model) else {
            throw WhisperError.noModelFound
        }
        onEvent(.phase("Loading model"))
        log("model: \(modelURL.lastPathComponent)")
        let engine = WhisperBatchTranscriber(modelURL: modelURL)
        try await engine.load()
        onEvent(.progress(0.25))

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
                        onEvent(.progress(0.28 + 0.60 * fraction))
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
        onEvent(.progress(0.88))

        // 4. Voice profiles — cli.py:636-650: one embedding per diarization
        // cluster, auto-matched against stored profiles; matched names relabel
        // everything downstream. When `save_voice_profiles` is on, each
        // auto-named cluster's embedding merges back into its stored profile
        // (save_profile's sample-weighted mean — that is how profiles grow more
        // accurate over time). New names still come from the Python CLI
        // (`--name-speakers`) until the app grows a naming UI; matching and
        // merging work with whatever the shared store already holds.
        var nameMap: [String: String] = [:]
        if !diarSegments.isEmpty,
           let embModel = DiarizationModel.findEmbeddingModel(
            explicit: settings.diarizationEmbeddingModel,
            searchDirectories: WhisperModel.searchDirectories)
        {
            onEvent(.phase("Recognizing speakers"))
            let clusterEmbeddings = try await SpeakerProfiles.computeSpeakerEmbeddings(
                samples: audio.samples,
                segments: diarSegments,
                embeddingModel: embModel)
            if !clusterEmbeddings.isEmpty {
                let (names, matches) = SpeakerProfiles.autoAssignNames(
                    clusterEmbeddings: clusterEmbeddings,
                    threshold: settings.speakerMatchThreshold)
                nameMap = names
                for cid in matches.keys.sorted() {
                    guard let match = matches[cid] ?? nil else { continue }
                    log(String(format: "speakers: cluster %d → %@ (score %.3f)",
                               cid, match.name, match.score))
                }
                if settings.saveVoiceProfiles {
                    for cid in matches.keys.sorted() {
                        guard let match = matches[cid] ?? nil,
                              let embedding = clusterEmbeddings[cid]
                        else { continue }
                        _ = try? SpeakerProfiles.saveProfile(name: match.name, embedding: embedding, samples: 1)
                        log("profile: updated \(match.name)")
                    }
                }
            }
        }

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
        // one JPEG per segment plus the manifest. Labels here are real when
        // diarization ran (max-overlap assignment) and the generic "Speaker"
        // otherwise — the same fallback Python's no-diarization frames path
        // uses (cli.py:722). Manifest shape is shared with Python's
        // `load_manifest`, so `whiz analyze` reads either side's output.
        let framesDir = outputDirectory.appendingPathComponent("\(stem).frames")
        var frameEntries: [FrameExtractor.Entry]? = nil
        let assigned = diarSegments.isEmpty
            ? segments.map { LabeledSegment(segment: $0, speaker: "Speaker") }
            : LabeledTranscript.assignSpeakers(segments: segments, diar: diarSegments)
        // cli.py:670-672 parity: profile names apply to the merged list itself
        // so the frames manifest, HTML, TXT and labeled SRT all carry them.
        let labeled = nameMap.isEmpty ? assigned : LabeledTranscript.relabel(assigned, nameMap)
        if segments.isEmpty {
            log("frames: skipped — no segments to capture")
        } else if !hasVideo {
            log("frames: skipped — no video track")
        } else {
            onEvent(.phase("Capturing frames"))
            // OCR wants larger frames — small UI text doesn't survive the
            // 1280 downscale — so the width is raised automatically with a
            // notice when OCR is on (config.py:ocr_min_width comment).
            var frameWidth = 1280
            if settings.ocr {
                frameWidth = max(frameWidth, settings.ocrMinWidth)
                log("ocr: frame width raised to \(frameWidth) — small UI text doesn't survive downscaling")
            }
            var entries = try await FrameExtractor.extractFrames(
                video: input,
                segments: labeled,
                into: framesDir,
                width: frameWidth,
                onProgress: { fraction in
                    onEvent(.progress(0.92 + 0.03 * fraction))
                })
            // Opt-in OCR (config.py:ocr — never auto, it is the slowest
            // stage). One pass over the captured frames, texts aligned by
            // index; failed captures keep an empty ocr because the path is
            // missing → counted failed by the batch. The manifest is written
            // once, after OCR, carrying v2's per-segment ocr fields.
            if settings.ocr {
                onEvent(.phase("Reading screens (OCR)"))
                let paths = entries.map { entry in
                    framesDir.appendingPathComponent(entry.frame)
                }
                let outcome = await FrameOCR.frames(
                    paths,
                    languages: settings.ocrLanguages,
                    minChars: settings.ocrMinChars,
                    maxChars: settings.ocrMaxChars,
                    dedupe: settings.ocrDedupe,
                    onProgress: { done, total, _ in
                        onEvent(.progress(0.95 + 0.04 * Double(done) / Double(max(total, 1))))
                    })
                for (index, text) in outcome.texts.enumerated() where index < entries.count {
                    entries[index].ocr = text
                }
                log(String(
                    format: "ocr: %d read, %d reused, %d empty, %d failed",
                    outcome.ok, outcome.reused, outcome.empty, outcome.failed))
            }
            // Written even when some or all captures failed — the manifest
            // aligns by index with empty `frame` fields, exactly like
            // write_manifest's contract.
            let manifestURL = outputDirectory.appendingPathComponent("\(stem).frames.json")
            try FrameExtractor.writeManifest(entries, framesDir: framesDir, to: manifestURL)
            let captured = entries.filter { !$0.frame.isEmpty }.count
            log(String(format: "frames: %d/%d extracted", captured, entries.count))
            log("output: \(manifestURL.path)")
            frameEntries = entries
        }

        // 6. The readable artifacts — the self-contained HTML transcript (frames
        // inlined as base64 when present), the dialogue TXT, and the labeled
        // SRT. Python writes these on the diarized path (_write_labeled_outputs:
        // .speakers.srt/.speakers.txt/.speakers.html); the HTML and TXT are a
        // documented divergence when diarization is absent — generic "Speaker"
        // labels keep the readable artifact available — while the labeled SRT
        // needs real labels to mean anything and is only written when it has
        // them (with one generic speaker it would only duplicate the plain
        // SRT).
        if !segments.isEmpty {
            onEvent(.phase("Writing HTML transcript"))
            let htmlURL = outputDirectory.appendingPathComponent("\(stem).speakers.html")
            try SpeakersHTML.format(
                labeled, framesDir: framesDir, entries: frameEntries,
                title: input.lastPathComponent)
                .write(to: htmlURL, atomically: true, encoding: .utf8)
            log("output: \(htmlURL.path)")

            let txtURL = outputDirectory.appendingPathComponent("\(stem).speakers.txt")
            try LabeledTranscript.formatDialogueTXT(labeled)
                .write(to: txtURL, atomically: true, encoding: .utf8)
            log("output: \(txtURL.path)")

            if !diarSegments.isEmpty {
                let labeledSRTURL = outputDirectory.appendingPathComponent("\(stem).speakers.srt")
                try LabeledTranscript.formatLabeledSRT(labeled)
                    .write(to: labeledSRTURL, atomically: true, encoding: .utf8)
                log("output: \(labeledSRTURL.path)")
            }
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