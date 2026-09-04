import Testing
import Foundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The merge math and profile-store behavior mirror tests/test_profiles.py
// fixture-for-fixture — the two implementations share one store
// (~/.config/whiz/speakers/), so they are held to one contract. Matching,
// filename sanitization and the extractor get their own coverage; the
// Python suite never pinned those, so the fixtures here are new.

@Suite("Speaker profiles")
struct SpeakerProfilesTests {

    // MARK: - merge_embeddings (test_profiles.py math fixtures)

    @Test("no old embedding returns the new one")
    func mergeNoOldReturnsNew() {
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [], oldSamples: 0, new: [0.1, 0.2, 0.3], newSamples: 1)
        #expect(out == [0.1, 0.2, 0.3])
        #expect(n == 1)
    }

    @Test("no new embedding returns the old one")
    func mergeNoNewReturnsOld() {
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [0.5, 0.5], oldSamples: 3, new: [], newSamples: 0)
        #expect(out == [0.5, 0.5])
        #expect(n == 3)
    }

    @Test("one sample on each side is a plain mean")
    func mergeEqualWeightedAverage() {
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [0.0, 2.0], oldSamples: 1, new: [2.0, 0.0], newSamples: 1)
        #expect(out == [1.0, 1.0])
        #expect(n == 2)
    }

    @Test("weighting is proportional to sample counts")
    func mergeWeightedBySamples() {
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [0.0], oldSamples: 3, new: [4.0], newSamples: 1)
        #expect(out == [1.0])
        #expect(n == 4)
    }

    @Test("the history cap keeps a new sample audible in the mean")
    func mergeHistoryCapped() {
        // 100 old samples cap at weight 5: (0*5 + 6*1)/6 = 1.0, not ~0.059.
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [0.0], oldSamples: 100, new: [6.0], newSamples: 1)
        #expect(out == [1.0])
        #expect(n == 101)
    }

    @Test("the cap affects weight, not the recorded count")
    func mergeCountPreservedWhenCapped() {
        let (_, n) = SpeakerProfiles.mergeEmbeddings(old: [1.0], oldSamples: 50, new: [1.0], newSamples: 2)
        #expect(n == 52)
    }

    @Test("zero new samples still count as one")
    func mergeZeroNewSamplesTreatedAsOne() {
        let (out, n) = SpeakerProfiles.mergeEmbeddings(old: [0.0, 0.0], oldSamples: 2, new: [2.0, 2.0], newSamples: 0)
        #expect(abs(out[0] - 2.0 / 3.0) < 1e-12)
        #expect(abs(out[1] - 2.0 / 3.0) < 1e-12)
        #expect(n == 2)
    }

    // MARK: - save_profile (test_profiles.py filesystem fixtures)

    @Test("saving creates a new profile with the exact payload")
    func saveCreatesNew() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let path = try SpeakerProfiles.saveProfile(name: "Alice", embedding: [0.1, 0.2, 0.3], samples: 1, in: dir)
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any]
        #expect((data?["name"] as? String) == "Alice")
        #expect((data?["samples"] as? NSNumber)?.intValue == 1)
        #expect((data?["dim"] as? NSNumber)?.intValue == 3)
        let embedding = (data?["embedding"] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        #expect(embedding == [0.1, 0.2, 0.3])
    }

    @Test("re-saving a same-dim profile merges via the weighted mean")
    func saveMergesWithExisting() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try SpeakerProfiles.saveProfile(name: "Bob", embedding: [0.0, 0.0], samples: 2, in: dir)
        try SpeakerProfiles.saveProfile(name: "Bob", embedding: [3.0, 3.0], samples: 1, in: dir)

        let path = SpeakerProfiles.profilePath(name: "Bob", in: dir)
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any]
        let embedding = (data?["embedding"] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        #expect(embedding == [1.0, 1.0])
        #expect((data?["samples"] as? NSNumber)?.intValue == 3)
        #expect((data?["dim"] as? NSNumber)?.intValue == 2)
    }

    @Test("a dimension mismatch replaces instead of averaging")
    func saveDimMismatchReplaces() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try SpeakerProfiles.saveProfile(name: "Carol", embedding: [1.0, 2.0, 3.0], samples: 4, in: dir)
        try SpeakerProfiles.saveProfile(name: "Carol", embedding: [9.0, 9.0], samples: 1, in: dir)

        let path = SpeakerProfiles.profilePath(name: "Carol", in: dir)
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any]
        #expect((data?["dim"] as? NSNumber)?.intValue == 2)
        let embedding = (data?["embedding"] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        #expect(embedding == [9.0, 9.0])
        #expect((data?["samples"] as? NSNumber)?.intValue == 1)
    }

    @Test("sample counts accumulate across repeated saves")
    func saveAccumulatesAcrossWrites() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        for _ in 0..<3 {
            try SpeakerProfiles.saveProfile(name: "Dave", embedding: [0.5, 0.5], samples: 1, in: dir)
        }
        let path = SpeakerProfiles.profilePath(name: "Dave", in: dir)
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any]
        #expect((data?["samples"] as? NSNumber)?.intValue == 3)
        let embedding = (data?["embedding"] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        #expect(embedding == [0.5, 0.5])
    }

    @Test("loading round-trips a merged profile")
    func loadRoundTripsMerged() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try SpeakerProfiles.saveProfile(name: "Eve", embedding: [1.0, 2.0], samples: 2, in: dir)
        try SpeakerProfiles.saveProfile(name: "Eve", embedding: [3.0, 4.0], samples: 1, in: dir)

        let profiles = SpeakerProfiles.loadProfiles(in: dir)
        #expect(profiles.count == 1)
        #expect(profiles[0].name == "Eve")
        #expect(profiles[0].samples == 3)
        #expect(abs(profiles[0].embedding[0] - 5.0 / 3.0) < 1e-12)
        #expect(abs(profiles[0].embedding[1] - 8.0 / 3.0) < 1e-12)
    }

    @Test("a profile without a name falls back to the file stem")
    func loadFallsBackToStemName() throws {
        let dir = tempDir()
        let path = dir.appendingPathComponent("Fay.json")
        try """
            {"embedding": [1.0, 0.0], "dim": 2, "samples": 1}
            """.data(using: .utf8)!.write(to: path)
        defer { try? FileManager.default.removeItem(at: dir) }

        let profiles = SpeakerProfiles.loadProfiles(in: dir)
        #expect(profiles.count == 1)
        #expect(profiles[0].name == "Fay")
    }

    @Test("profile filenames are sanitized like profiles.py:_profile_path")
    func profilePathSanitization() {
        let dir = URL(fileURLWithPath: "/tmp/store")
        #expect(SpeakerProfiles.profilePath(name: "Alice", in: dir)
                == dir.appendingPathComponent("Alice.json"))
        #expect(SpeakerProfiles.profilePath(name: "Bob Q", in: dir)
                == dir.appendingPathComponent("Bob_Q.json"))
        // Underscores are not whitespace: "!!!" stays "___" and only a
        // name that is empty after stripping falls back to "speaker"
        // (Python: .strip() then `if not safe`).
        #expect(SpeakerProfiles.profilePath(name: "!!!", in: dir)
                == dir.appendingPathComponent("___.json"))
        #expect(SpeakerProfiles.profilePath(name: "   ", in: dir)
                == dir.appendingPathComponent("speaker.json"))
    }

    // MARK: - Matching (no Python fixtures — pinned fresh)

    @Test("cosine similarity of aligned, orthogonal and empty vectors")
    func cosineSimilarityBasics() {
        #expect(abs(SpeakerProfiles.cosineSimilarity([1, 0], [1, 0]) - 1.0) < 1e-12)
        #expect(abs(SpeakerProfiles.cosineSimilarity([1, 0], [0, 1])) < 1e-12)
        #expect(SpeakerProfiles.cosineSimilarity([], [1.0]) == 0.0)
    }

    @Test("matching is greedy and exclusive — no name serves two clusters")
    func matchingIsGreedyAndExclusive() {
        let alice = SpeakerProfiles.Profile(
            name: "Alice", embedding: [1, 0, 0], dim: 3, created: "", samples: 1)
        let bob = SpeakerProfiles.Profile(
            name: "Bob", embedding: [0, 1, 0], dim: 3, created: "", samples: 1)
        // Cluster 0 is very close to Alice, cluster 1 somewhat close to Alice
        // too — the greedy order must let cluster 0 claim Alice, and cluster 1
        // can only take Bob if it actually clears the threshold.
        let clusters = [
            0: [0.95, 0.05, 0.0],
            1: [0.05, 0.95, 0.0],
        ]
        let matches = SpeakerProfiles.matchSpeakers(
            clusterEmbeddings: clusters, profiles: [alice, bob], threshold: 0.8)
        #expect((matches[0] ?? nil)?.name == "Alice")
        #expect((matches[1] ?? nil)?.name == "Bob")
    }

    @Test("below-threshold clusters stay unnamed")
    func belowThresholdStaysUnnamed() {
        let profile = SpeakerProfiles.Profile(
            name: "Alice", embedding: [1, 0], dim: 2, created: "", samples: 1)
        let matches = SpeakerProfiles.matchSpeakers(
            clusterEmbeddings: [0: [0.0, 1.0]], profiles: [profile], threshold: 0.8)
        #expect((matches[0] ?? nil) == nil)

        let (nameMap, raw) = SpeakerProfiles.autoAssignNames(
            clusterEmbeddings: [0: [0.0, 1.0]], threshold: 0.8, profiles: [profile])
        #expect(nameMap.isEmpty)
        #expect((raw[0] ?? nil) == nil)
    }

    @Test("autoAssignNames keys the map by speaker label")
    func autoAssignNamesUsesLabels() {
        let profile = SpeakerProfiles.Profile(
            name: "Alice", embedding: [1, 0], dim: 2, created: "", samples: 1)
        let (nameMap, matches) = SpeakerProfiles.autoAssignNames(
            clusterEmbeddings: [2: [1.0, 0.0]], threshold: 0.8, profiles: [profile])
        #expect(nameMap["Speaker C"] == "Alice")
        #expect((matches[2] ?? nil)?.name == "Alice")
    }

    // MARK: - Extraction (real dylibs + real model, gated)

    @Test(.disabled(if: DiarizationModel.findEmbeddingModel(
        explicit: "", searchDirectories: WhisperModel.searchDirectories) == nil))
    func extractionThroughTheRealExtractor() async throws {
        let model = DiarizationModel.findEmbeddingModel(
            explicit: "", searchDirectories: WhisperModel.searchDirectories)!

        // The corpus fixture: 5.2 s at 16 kHz, with two utterance-shaped
        // regions handed to two speaker clusters.
        let wav = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tuning/golden/quiet_two_utterances.wav")
        let audio = try await AudioFileDecoder.extractSamples(at: wav)

        let segments = [
            DiarSegment(start: 0.5, end: 2.0, speaker: 0),
            DiarSegment(start: 2.6, end: 4.8, speaker: 1),
        ]
        let embeddings = try await SpeakerProfiles.computeSpeakerEmbeddings(
            samples: audio.samples, segments: segments, embeddingModel: model)

        print("PROFILE E2E: \(embeddings.count)/2 clusters embedded, dim \(embeddings.values.first?.count ?? 0)")
        #expect(!embeddings.isEmpty)
        let dims = Set(embeddings.values.map(\.count))
        #expect(dims.count == 1)
        #expect(dims.first ?? 0 > 0)

        // A computed embedding must be usable: matching it against itself
        // scores 1.0.
        let first = embeddings[embeddings.keys.min()!]
        let profile = SpeakerProfiles.Profile(
            name: "Self", embedding: first!, dim: first!.count, created: "", samples: 1)
        let matches = SpeakerProfiles.matchSpeakers(
            clusterEmbeddings: embeddings, profiles: [profile], threshold: 0.9)
        #expect((matches[embeddings.keys.min()!] ?? nil)?.name == "Self")
    }

    // MARK: - Helpers

    private func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-profiles-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}