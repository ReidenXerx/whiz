import CSherpa
import Foundation

/// Speaker voice profiles — cross-recording speaker recognition, the Swift
/// half of `whiz/profiles.py`.
///
/// Profiles are fixed-size embedding vectors for named speakers, computed
/// with the same sherpa-onnx embedding extractor diarization uses. On later
/// recordings, each detected cluster's embedding is compared (cosine
/// similarity) to the stored profiles and a name is auto-assigned when the
/// best match exceeds `speaker_match_threshold` (config, default 0.8).
///
/// The store is shared with the Python side: one inspectable JSON file per
/// name under `~/.config/whiz/speakers/`, merged across recordings via a
/// sample-weighted running mean (capped by `maxHistoryWeight`) so profiles
/// grow more accurate instead of being overwritten. The Python CLI's
/// `--name-speakers` creates profiles; this port reads, matches and *merges*
/// them — new names from the app wait on a naming UI.
///
/// Everything pure is pinned in `SpeakerProfilesTests` fixture-for-fixture
/// against `tests/test_profiles.py`.
enum SpeakerProfiles {

    /// profiles.py:_MAX_HISTORY_WEIGHT — caps how much history can outweigh
    /// a new sample, so the profile keeps adapting to a changed mic/voice.
    static let maxHistoryWeight = 5

    /// profiles.py:Profile.
    struct Profile: Sendable, Equatable {
        var name: String
        var embedding: [Double]
        var dim: Int
        var created: String
        var samples: Int
    }

    // MARK: - Store

    /// profiles.py:profiles_dir — `<config dir>/speakers`, honouring
    /// `WHIZ_CONFIG_DIR` the same way Python does.
    static var profilesDirectory: URL {
        WhizConfig.directory.appendingPathComponent("speakers")
    }

    /// profiles.py:_profile_path — sanitize the name to a filename-safe one.
    static func profilePath(name: String, in directory: URL = profilesDirectory) -> URL {
        var safe = ""
        for scalar in name.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar)
                || scalar == "-" || scalar == "_" || scalar == " " {
                safe.unicodeScalars.append(scalar)
            } else {
                safe.append("_")
            }
        }
        safe = safe.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " ", with: "_")
        if safe.isEmpty { safe = "speaker" }
        return directory.appendingPathComponent("\(safe).json")
    }

    /// All stored profiles, sorted by filename (Python sorts the glob).
    /// Corrupt files are skipped, not fatal.
    static func loadProfiles(in directory: URL = profilesDirectory) -> [Profile] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        var out: [Profile] = []
        for file in files.sorted(by: { $0.path < $1.path })
        where file.pathExtension.lowercased() == "json" {
            guard let data = try? Data(contentsOf: file),
                  let object = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
            else { continue }
            guard let embedding = object["embedding"] as? [Any], !embedding.isEmpty
            else { continue }
            let vector = embedding.compactMap { ($0 as? NSNumber)?.doubleValue }
            guard vector.count == embedding.count, !vector.isEmpty else { continue }
            let name = (object["name"] as? String)
                ?? file.deletingPathExtension().lastPathComponent
            out.append(Profile(
                name: name,
                embedding: vector,
                dim: (object["dim"] as? NSNumber)?.intValue ?? vector.count,
                created: (object["created"] as? String) ?? "",
                samples: (object["samples"] as? NSNumber)?.intValue ?? 0))
        }
        return out
    }

    /// profiles.py:merge_embeddings — sample-weighted running mean with the
    /// history cap. Returns the merged vector and the new total sample count.
    /// Dimension matching is the caller's job (mismatched dims discard the
    /// old profile rather than average).
    static func mergeEmbeddings(
        old: [Double],
        oldSamples: Int,
        new: [Double],
        newSamples: Int = 1
    ) -> ([Double], Int) {
        if old.isEmpty { return (new, max(1, newSamples)) }
        if new.isEmpty { return (old, oldSamples) }
        let oldWeight = Double(min(max(0, oldSamples), maxHistoryWeight))
        let newWeight = Double(max(1, newSamples))
        let total = oldWeight + newWeight
        let merged = zip(old, new).map { ($0 * oldWeight + $1 * newWeight) / total }
        return (merged, oldSamples + newSamples)
    }

    /// profiles.py:save_profile — persist (and merge with) a profile. Same
    /// embedding dimension merges; a different dimension (embedding model
    /// swapped) replaces the old profile.
    static func saveProfile(
        name: String,
        embedding: [Double],
        samples: Int = 1,
        in directory: URL = profilesDirectory
    ) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var finalEmbedding = embedding
        var totalSamples = samples
        let path = profilePath(name: name, in: directory)
        if let data = try? Data(contentsOf: path),
           let prior = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let oldEmbedding = prior["embedding"] as? [Any],
           let oldVector = oldEmbedding.compactMap({ ($0 as? NSNumber)?.doubleValue })
                as? [Double],
           !oldVector.isEmpty,
           oldVector.count == finalEmbedding.count {
            let (merged, total) = mergeEmbeddings(
                old: oldVector,
                oldSamples: (prior["samples"] as? NSNumber)?.intValue ?? 0,
                new: finalEmbedding,
                newSamples: samples)
            finalEmbedding = merged
            totalSamples = total
        }
        // Dimension mismatch (or unreadable prior): the new profile replaces.

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let created = formatter.string(from: Date())

        var json = "{"
        json += "\"name\": \(jsonString(name)), "
        json += "\"dim\": \(finalEmbedding.count), "
        json += "\"embedding\": ["
        json += finalEmbedding.map { "\($0)" }.joined(separator: ", ")
        json += "], "
        json += "\"created\": \(jsonString(created)), "
        json += "\"samples\": \(totalSamples)"
        json += "}"
        if let data = json.data(using: .utf8) {
            try data.write(to: path, options: .atomic)
        } else {
            throw SpeakerProfilesError.writeFailed(path.path)
        }
        return path
    }

    /// profiles.py:forget_profile — delete by name.
    static func forgetProfile(name: String, in directory: URL = profilesDirectory) -> Bool {
        let path = profilePath(name: name, in: directory)
        guard FileManager.default.fileExists(atPath: path.path) else { return false }
        try? FileManager.default.removeItem(at: path)
        return true
    }

    // MARK: - Matching

    /// profiles.py:cosine_similarity of two vectors (range -1…1).
    static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        if a.isEmpty || b.isEmpty { return 0.0 }
        let n = min(a.count, b.count)
        var dot = 0.0, normA = 0.0, normB = 0.0
        for i in 0..<n {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        if normA <= 0.0 || normB <= 0.0 { return 0.0 }
        return dot / ((normA.squareRoot()) * (normB.squareRoot()))
    }

    /// profiles.py:match_speakers — greedy exclusive assignment: every
    /// (cluster, profile) pair is scored, best pairs claim their name and
    /// cluster first, and no name is ever assigned to two clusters.
    static func matchSpeakers(
        clusterEmbeddings: [Int: [Double]],
        profiles: [Profile]? = nil,
        threshold: Double = 0.8
    ) -> [Int: (name: String, score: Double)?] {
        let stored = profiles ?? loadProfiles()
        var matched = [Int: (name: String, score: Double)?]()
        for cid in clusterEmbeddings.keys { matched[cid] = nil }

        if stored.isEmpty || clusterEmbeddings.isEmpty { return matched }

        var scored: [(score: Double, cluster: Int, name: String)] = []
        for (cid, embedding) in clusterEmbeddings {
            for profile in stored {
                scored.append((cosineSimilarity(embedding, profile.embedding), cid, profile.name))
            }
        }
        scored.sort { $0.score > $1.score }

        var usedNames = Set<String>()
        var usedClusters = Set<Int>()
        for entry in scored {
            if entry.score < threshold { break }
            if usedClusters.contains(entry.cluster) || usedNames.contains(entry.name) { continue }
            matched[entry.cluster] = (entry.name, entry.score)
            usedClusters.insert(entry.cluster)
            usedNames.insert(entry.name)
        }
        return matched
    }

    /// profiles.py:auto_assign_names — the {speaker_label: name} map for
    /// relabeling, from matches that cleared the threshold.
    static func autoAssignNames(
        clusterEmbeddings: [Int: [Double]],
        threshold: Double = 0.8,
        profiles: [Profile]? = nil
    ) -> (nameMap: [String: String], matches: [Int: (name: String, score: Double)?]) {
        let matches = matchSpeakers(
            clusterEmbeddings: clusterEmbeddings, profiles: profiles, threshold: threshold)
        var nameMap = [String: String]()
        for (cid, match) in matches {
            if let match {
                nameMap[LabeledTranscript.speakerLabel(cid)] = match.name
            }
        }
        return (nameMap, matches)
    }

    // MARK: - Embedding extraction

    private static let queue = DispatchQueue(label: "whiz.speaker-embeddings", qos: .userInitiated)

    /// profiles.py:compute_speaker_embeddings — one averaged embedding per
    /// speaker cluster. Each cluster's diarization segments are concatenated
    /// and fed to the streaming extractor in ≤30 s chunks (utterances under
    /// 0.3 s are skipped — not enough audio for an embedding). Clusters whose
    /// total audio is too short are omitted.
    ///
    /// The C API has no progress callback for this pass; callers show it as a
    /// phase. Blocking inference runs on a dedicated serial queue.
    static func computeSpeakerEmbeddings(
        samples: [Float],
        segments: [DiarSegment],
        embeddingModel: URL,
        threads: Int? = nil
    ) async throws -> [Int: [Double]] {
        guard !segments.isEmpty else { return [:] }

        let request = ExtractionRequest(
            samples: samples,
            segments: segments,
            modelPath: embeddingModel.path,
            threads: threads ?? min(4, ProcessInfo.processInfo.activeProcessorCount))

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    continuation.resume(returning: try Self.runExtraction(request))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct ExtractionRequest: Sendable {
        let samples: [Float]
        let segments: [DiarSegment]
        let modelPath: String
        let threads: Int
    }

    private static func runExtraction(_ request: ExtractionRequest) throws -> [Int: [Double]] {
        let sampleRate = Int(WhisperEngine.sampleRate)
        let minimum = Int(Double(sampleRate) * 0.3)   // utterances shorter than this are skipped
        let chunk = sampleRate * 30                   // the streaming extractor's context window

        return try request.modelPath.withCString { modelPath in
            try "cpu".withCString { provider in
                var config = SherpaOnnxSpeakerEmbeddingExtractorConfig()
                config.model = modelPath
                config.num_threads = Int32(request.threads)
                config.provider = provider

                guard let extractor = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config) else {
                    throw SpeakerProfilesError.extractorFailed(request.modelPath)
                }
                defer { SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor) }
                let dim = Int(SherpaOnnxSpeakerEmbeddingExtractorDim(extractor))

                // Group sample ranges by speaker (profiles.py:218-225).
                var bySpeaker = [Int: [(Int, Int)]]()
                for segment in request.segments {
                    let start = max(0, Int(segment.start * Double(sampleRate)))
                    let end = min(request.samples.count, Int(segment.end * Double(sampleRate)))
                    if end > start {
                        bySpeaker[segment.speaker, default: []].append((start, end))
                    }
                }

                var out = [Int: [Double]]()
                for (speaker, ranges) in bySpeaker {
                    var vectors: [[Double]] = []
                    for (start, end) in ranges {
                        // Skip very short utterances — not enough for an embedding.
                        if end - start < minimum { continue }
                        var offset = start
                        while offset < end {
                            let blockEnd = min(offset + chunk, end)
                            if blockEnd - offset < minimum { break }
                            guard let stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor)
                            else { continue }
                            defer { SherpaOnnxDestroyOnlineStream(stream) }

                            let count = blockEnd - offset
                            request.samples[offset..<blockEnd].withUnsafeBufferPointer { buffer in
                                SherpaOnnxOnlineStreamAcceptWaveform(
                                    stream, Int32(sampleRate), buffer.baseAddress, Int32(count))
                            }
                            SherpaOnnxOnlineStreamInputFinished(stream)
                            if SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, stream) == 1,
                               let raw = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream) {
                                vectors.append((0..<dim).map { Double(raw[$0]) })
                                SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(raw)
                            }
                            offset = blockEnd
                        }
                    }
                    if !vectors.isEmpty {
                        out[speaker] = average(vectors, dim: dim)
                    }
                }
                return out
            }
        }
    }

    /// profiles.py:_average_vectors — element-wise mean with the same
    /// dimension guard.
    private static func average(_ vectors: [[Double]], dim: Int) -> [Double] {
        var accumulator = [Double](repeating: 0, count: dim)
        for vector in vectors {
            for i in 0..<min(dim, vector.count) {
                accumulator[i] += vector[i]
            }
        }
        let count = Double(vectors.count)
        return accumulator.map { $0 / count }
    }

    /// Minimal JSON string escaping for the two characters a name or
    /// timestamp can realistically contain.
    private static func jsonString(_ text: String) -> String {
        let escaped = text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }
}

enum SpeakerProfilesError: LocalizedError, Sendable, Equatable {
    case extractorFailed(String)
    case writeFailed(String)

    var errorDescription: String? {
        switch self {
        case .extractorFailed(let model):
            return "Could not load the speaker embedding extractor from \(model)."
        case .writeFailed(let path):
            return "Could not write a speaker profile at \(path)."
        }
    }
}