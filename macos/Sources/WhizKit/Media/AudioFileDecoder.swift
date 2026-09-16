import AVFoundation
import CoreMedia
import Foundation

/// Decodes any media file AVFoundation can read into Whisper-native PCM.
///
/// Replaces `whiz/audio.py`'s `extract_audio`: same contract — 16 kHz mono —
/// but entirely in memory. The Python pipeline shells out to ffmpeg precisely
/// because `whisper-cli` cannot demux video containers; here the samples go
/// straight into `whisper_full`, so no intermediate WAV ever touches disk.
///
/// AVFoundation coverage is narrower than ffmpeg's and that is a documented
/// trade, not an oversight: AVFoundation does not demux Matroska (.mkv),
/// WebM or AVI, and does not read Ogg (.ogg/.opus) or WMA. Those extensions
/// are rejected up front with a message pointing at the Python CLI, which
/// stays the escape hatch for exotic containers — better a clear "we can't"
/// than a generic AVFoundation error the user has to interpret.
///
/// Memory: the whole file's PCM is held in RAM — roughly 230 MB per hour of
/// audio — which is fine for transcription-length inputs and keeps the
/// decoder a simple pull loop with no partial state to manage.
enum AudioFileDecoder {

    /// Extensions AVFoundation cannot demux at all. Checked before opening so
    /// the error can name the real cause instead of AVFoundation's.
    private static let unsupportedContainers: Set<String> = [
        "mkv", "webm", "avi", "ogg", "opus", "wma",
    ]

    /// The decode result: samples in [-1, 1] at Whisper's native rate, and the
    /// duration implied by the decoded sample count (not the container's
    /// metadata — they can disagree, and the samples are what gets transcribed).
    struct ExtractedAudio: Sendable {
        let samples: [Float]
        let duration: Double
    }

    /// True when the file's container is known to be beyond AVFoundation, so UI
    /// can flag it before any decode is attempted (file picker validation).
    static func isUnsupportedContainer(_ url: URL) -> Bool {
        unsupportedContainers.contains(url.pathExtension.lowercased())
    }

    /// Extract 16 kHz mono Float32 samples from an audio or video file.
    ///
    /// Nonisolated async, so the decode runs on the global concurrent
    /// executor, never the main actor — and, unlike the detached-task version
    /// this once used, the surrounding `Task`'s cancellation propagates into
    /// the decode loop. A detached task kept decoding in the background after
    /// the caller cancelled: invisible, unstoppable work.
    ///
    /// `onProgress` reports decode completion as a 0…1 fraction, measured
    /// against the container's own duration estimate. Coarse if the metadata
    /// lies, but good enough for a progress bar — and it never reaches 1.0
    /// before the decode actually finishes.
    static func extractSamples(
        at url: URL,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> ExtractedAudio {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioDecodeError.fileNotFound(url)
        }
        let ext = url.pathExtension.lowercased()
        guard !unsupportedContainers.contains(ext) else {
            throw AudioDecodeError.unsupportedContainer(ext)
        }
        return try await Self.decode(at: url, onProgress: onProgress)
    }

    // MARK: - Decoding

    private static func decode(at url: URL, onProgress: (@Sendable (Double) -> Void)?) async throws -> ExtractedAudio {
        let asset = AVURLAsset(url: url)

        // Metadata loading throws its own NSError (damaged files, DRM) rather
        // than reporting through `isReadable` — observed as AVFoundation
        // -11849 "Operation Stopped" on a zero-byte file. Any such throw is the
        // same user-facing outcome as "cannot open", so it maps here instead
        // of leaking a raw AVFoundation error to callers.
        let track: AVAssetTrack
        let assetDuration: Double
        do {
            guard try await asset.load(.isReadable) else {
                throw AudioDecodeError.unreadableMedia(url.lastPathComponent)
            }
            guard let first = try await asset.loadTracks(withMediaType: .audio).first else {
                throw AudioDecodeError.noAudioTrack(url.lastPathComponent)
            }
            track = first
            assetDuration = CMTimeGetSeconds(try await asset.load(.duration))
        } catch let error as AudioDecodeError {
            throw error
        } catch {
            throw AudioDecodeError.unreadableMedia(url.lastPathComponent, error.localizedDescription)
        }

        // Reserve capacity from the container's own duration estimate; the
        // count reported back comes from the decoded samples, so a lying
        // metadata field costs at most a reallocation. The same estimate also
        // drives `onProgress` — 0 until real samples exist.
        let expectedTotal = assetDuration.isFinite && assetDuration > 0
            ? Int(assetDuration * WhisperEngine.sampleRate)
            : 0
        var samples: [Float] = []
        if expectedTotal > 0 {
            samples.reserveCapacity(expectedTotal)
        }

        let output = AVAssetReaderTrackOutput(
            track: track,
            // Requesting 16 kHz Float32 makes AVFoundation do the resample
            // itself — `audio.py` needed ffmpeg flags for exactly this
            // (`-ar 16000`). The channel count is deliberately NOT constrained:
            // the built-in stereo→mono mixdown is equal-power (×0.7071 per
            // channel), which boosts correlated content by 3 dB and — worse —
            // emits out-of-range PCM on loud material (observed +1.27 on a
            // 0.9-amplitude identical-pair stereo file). The mean downmix in
            // `append` is taken instead, which is ffmpeg `-ac 1`'s behaviour and
            // can never leave [-1, 1].
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: WhisperEngine.sampleRate,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
        )
        let reader = try AVAssetReader(asset: asset)
        reader.add(output)
        guard reader.startReading() else {
            throw AudioDecodeError.decodeFailed(reader.error?.localizedDescription)
        }

        while let buffer = output.copyNextSampleBuffer() {
            try append(buffer, to: &samples)
            // Cancellation must also stop AVFoundation's own decode pipeline,
            // or the reader keeps working in the background past the throw.
            if Task.isCancelled {
                reader.cancelReading()
                throw CancellationError()
            }
            if let onProgress, expectedTotal > 0 {
                onProgress(min(0.99, Double(samples.count) / Double(expectedTotal)))
            }
        }
        // A nil buffer means either end-of-file or a decode failure AVFoundation
        // already recorded — only the status tells them apart, so never treat a
        // clean loop exit as success without checking.
        guard reader.status == .completed else {
            throw AudioDecodeError.decodeFailed(reader.error?.localizedDescription)
        }

        let duration = Double(samples.count) / WhisperEngine.sampleRate
        return ExtractedAudio(samples: samples, duration: duration)
    }

    /// Append one Linear-PCM sample buffer as 16 kHz mono samples.
    ///
    /// The output settings promise interleaved Float32 at the source's channel
    /// count, so a multi-channel frame becomes one mono sample here by taking
    /// the plain mean across channels — ffmpeg `-ac 1`'s downmix, not
    /// AVFoundation's equal-power one (see the settings comment above). The
    /// mean of samples in [-1, 1] stays in [-1, 1], so loud files can never
    /// come back out of range the way AVFoundation's mixer produced.
    private static func append(_ buffer: CMSampleBuffer, to samples: inout [Float]) throws {
        guard let format = CMSampleBufferGetFormatDescription(buffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format),
              asbd.pointee.mChannelsPerFrame > 0 else {
            throw AudioDecodeError.decodeFailed("unexpected PCM format")
        }
        let channels = Int(asbd.pointee.mChannelsPerFrame)

        guard let block = CMSampleBufferGetDataBuffer(buffer) else {
            throw AudioDecodeError.decodeFailed(nil)
        }
        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        var unused = 0
        let status = CMBlockBufferGetDataPointer(
            block,
            atOffset: 0,
            lengthAtOffsetOut: &length,
            totalLengthOut: &unused,
            dataPointerOut: &pointer
        )
        guard status == kCMBlockBufferNoErr, let pointer else {
            throw AudioDecodeError.decodeFailed(nil)
        }
        let count = length / MemoryLayout<Float>.size
        let floats = UnsafeMutableRawPointer(pointer).assumingMemoryBound(to: Float.self)

        if channels == 1 {
            samples.append(contentsOf: UnsafeBufferPointer(start: floats, count: count))
            return
        }
        var i = 0
        while i + channels <= count {
            var sum: Float = 0
            for channel in 0..<channels {
                sum += floats[i + channel]
            }
            samples.append(sum / Float(channels))
            i += channels
        }
    }
}

enum AudioDecodeError: LocalizedError, Sendable, Equatable {
    case fileNotFound(URL)
    case unsupportedContainer(String)
    case unreadableMedia(String, String? = nil)
    case noAudioTrack(String)
    case decodeFailed(String?)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Input file not found: \(url.path)"
        case .unsupportedContainer(let ext):
            return """
                Whiz cannot read .\(ext) files — macOS has no decoder for that \
                container. Use the Python CLI instead: whiz transcribe <file>
                """
        case .unreadableMedia(let name, let detail):
            if let detail {
                return "'\(name)' could not be opened (\(detail))."
            }
            return """
                '\(name)' could not be opened — the file may be corrupt, \
                DRM-protected, or in a format macOS cannot decode.
                """
        case .noAudioTrack(let name):
            return "'\(name)' has no audio track."
        case .decodeFailed(let detail):
            let reason = detail.map { " (\($0))" } ?? ""
            return "Audio decoding failed\(reason)"
        }
    }
}