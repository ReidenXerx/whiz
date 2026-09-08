import Foundation

/// One-time download of the speaker diarization models.
///
/// The Swift counterpart to `cli.py:_ensure_diarization_ready`, minus its
/// hardest part: Python must `pip install sherpa-onnx` into the running venv
/// before it can diarize at all, because the library is an optional extra. The
/// app links sherpa statically, so only the two model assets are ever missing —
/// what remains is a download, not an installation.
///
/// Same assets, same destination, same filenames as `whiz/diarize.py`, so models
/// fetched by either side satisfy the other. Nothing here is whiz-specific: they
/// are the stock sherpa-onnx release assets.
///
/// Consent matters for the same reason it does in Python (review decision
/// 2026-09-07): this is a ~44 MB download the user did not explicitly ask for,
/// triggered by opening a video. It is offered, never assumed — see
/// `WhizConfig.autoDiarizationSetup`.
@MainActor
final class DiarizationSetup: ObservableObject {

    /// Stock sherpa-onnx release assets — the URLs in `whiz/diarize.py`.
    /// The upstream "recongition" typo is theirs and load-bearing.
    private static let segmentationURL = URL(string:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")!
    private static let embeddingURL = URL(string:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        + "speaker-recongition-models/"
        + "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")!

    /// Shown before asking for consent, so "download?" is an informed question.
    static let approximateBytes: Int64 = 44_400_000

    enum State: Equatable {
        case idle
        case downloading(phase: String, progress: Double)
        case failed(String)
        case finished
    }

    @Published private(set) var state: State = .idle

    var isRunning: Bool {
        if case .downloading = state { return true }
        return false
    }

    /// Whether both models are already resolvable — the same lookup the
    /// pipeline uses, so this cannot disagree with what a run will find.
    static var isInstalled: Bool {
        DiarizationModel.findSegmentationModel(
            explicit: "", searchDirectories: WhisperModel.searchDirectories) != nil
        && DiarizationModel.findEmbeddingModel(
            explicit: "", searchDirectories: WhisperModel.searchDirectories) != nil
    }

    private var task: Task<Void, Never>?

    func cancel() {
        task?.cancel()
        task = nil
        state = .idle
    }

    func install() {
        guard !isRunning else { return }
        state = .downloading(phase: "Starting…", progress: 0)
        task = Task { [weak self] in
            do {
                try await self?.run()
                self?.state = .finished
                Log.stt.notice("diarization models installed")
            } catch is CancellationError {
                self?.state = .idle
            } catch {
                Log.stt.error(
                    "diarization setup failed: \(error.localizedDescription, privacy: .public)")
                self?.state = .failed(error.localizedDescription)
            }
        }
    }

    private func run() async throws {
        let directory = DiarizationModel.defaultDirectory
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        // Embedding model first: it is a plain .onnx, so a failure here costs
        // nothing to retry and surfaces network problems before the archive.
        state = .downloading(phase: "Speaker embeddings (38 MB)", progress: 0.05)
        let embedding = directory.appendingPathComponent(DiarizationModel.embeddingModelName)
        if !FileManager.default.fileExists(atPath: embedding.path) {
            try await download(Self.embeddingURL, to: embedding) { [weak self] fraction in
                self?.state = .downloading(
                    phase: "Speaker embeddings (38 MB)", progress: 0.05 + fraction * 0.6)
            }
        }
        try Task.checkCancellation()

        state = .downloading(phase: "Segmentation model (7 MB)", progress: 0.7)
        let segmentationDirectory = directory.appendingPathComponent(
            DiarizationModel.segmentationDirectoryName)
        if !FileManager.default.fileExists(
            atPath: segmentationDirectory
                .appendingPathComponent(DiarizationModel.segmentationModelName).path)
        {
            let archive = directory.appendingPathComponent("segmentation.tar.bz2")
            try await download(Self.segmentationURL, to: archive) { [weak self] fraction in
                self?.state = .downloading(
                    phase: "Segmentation model (7 MB)", progress: 0.7 + fraction * 0.2)
            }
            state = .downloading(phase: "Extracting", progress: 0.92)
            try Self.extract(archive, into: directory)
            try? FileManager.default.removeItem(at: archive)
        }

        // Verify through the pipeline's own resolver rather than trusting that
        // the files landed: a truncated download or an archive whose layout
        // changed would otherwise be discovered at transcription time.
        guard Self.isInstalled else {
            throw DiarizationSetupError.incomplete
        }
        state = .downloading(phase: "Done", progress: 1.0)
    }

    /// `URLSession.download` with progress, honouring task cancellation.
    private func download(
        _ url: URL,
        to destination: URL,
        onProgress: @escaping @MainActor (Double) -> Void
    ) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw DiarizationSetupError.http(http.statusCode)
        }
        let expected = response.expectedContentLength

        var data = Data()
        if expected > 0 { data.reserveCapacity(Int(expected)) }
        var lastReported = 0.0
        for try await byte in bytes {
            data.append(byte)
            if expected > 0 {
                let fraction = Double(data.count) / Double(expected)
                // Republishing per byte would flood the main actor.
                if fraction - lastReported > 0.01 {
                    lastReported = fraction
                    onProgress(fraction)
                }
            }
        }
        try Task.checkCancellation()
        try data.write(to: destination, options: .atomic)
    }

    /// Extract with `/usr/bin/tar`. Foundation has no archive support, and the
    /// system tar handles bz2 without a third-party dependency.
    private static func extract(_ archive: URL, into directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xjf", archive.path, "-C", directory.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw DiarizationSetupError.extractionFailed(Int(process.terminationStatus))
        }
    }
}

enum DiarizationSetupError: LocalizedError {
    case http(Int)
    case extractionFailed(Int)
    case incomplete

    var errorDescription: String? {
        switch self {
        case .http(let code):
            return "Download failed (HTTP \(code))."
        case .extractionFailed(let status):
            return "Could not extract the segmentation model (tar exited \(status))."
        case .incomplete:
            return "Models downloaded but could not be found afterwards — "
                 + "the archive layout may have changed upstream."
        }
    }
}
