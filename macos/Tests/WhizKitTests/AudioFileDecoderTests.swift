import Testing
import Foundation
import AVFoundation
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// Fixtures are generated in-test as lossless PCM rather than committed to the
// repo: lossless means no codec variance, and regenerating follows the golden
// corpus's precedent of not committing what can be derived.

@Suite("Audio file decoding")
struct AudioFileDecoderTests {

    /// 2.0 s of a 440 Hz sine, identical in every channel: averaging identical
    /// channels is a no-op, so the decoded level directly measures the downmix
    /// the decoder applies. Every amplitude assertion below is calibrated to
    /// this signal.
    private static let seconds = 2.0
    private static let frequency = 440.0

    // MARK: - Fixture writing

    private func writeWav(at url: URL, sampleRate: Double, channels: AVAudioChannelCount, amplitude: Float = 0.5) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels) else {
            throw AudioDecodeError.unreadableMedia(url.lastPathComponent)
        }
        let frames = AVAudioFrameCount(Self.seconds * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw AudioDecodeError.unreadableMedia(url.lastPathComponent)
        }
        buffer.frameLength = frames
        for channel in 0..<Int(channels) {
            let data = buffer.floatChannelData![channel]
            for i in 0..<Int(frames) {
                let t = Double(i) / sampleRate
                let value = sin(2.0 * Double.pi * Self.frequency * t)
                data[i] = amplitude * Float(value)
            }
        }
        try file.write(from: buffer)
    }

    @Test("loud correlated stereo stays in range after the downmix")
    func loudStereoDoesNotExceedFullScale() async throws {
        let url = tempURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeWav(at: url, sampleRate: 44_100, channels: 2, amplitude: 0.9)

        let extracted = try await AudioFileDecoder.extractSamples(at: url)

        // A regression pin, not paranoia: AVFoundation's own equal-power mixdown
        // emitted samples at ±1.28 for exactly this input (×0.7071 per channel,
        // summed — no clipping, no range check). The mean downmix must come
        // back at the source's 0.9 and never leave [-1, 1].
        let peak = extracted.samples.map { abs($0) }.max() ?? 0
        #expect(peak > 0.8 && peak < 1.0)
        #expect(extracted.samples.filter { abs($0) > 1.0 }.isEmpty)
    }

    private func tempURL(_ ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-decoder-\(UUID().uuidString).\(ext)")
    }

    // MARK: - Decoding

    @Test("stereo 44.1 kHz WAV decodes to 16 kHz mono with its level intact")
    func decodesStereo44kHzToMono16kHz() async throws {
        let url = tempURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeWav(at: url, sampleRate: 44_100, channels: 2)

        let extracted = try await AudioFileDecoder.extractSamples(at: url)

        // 2 s at 16 kHz = 32 000 samples. Assert a band, not an exact count —
        // the rate converter may add or drop boundary samples — but far tighter
        // than the failure modes: no resampling yields 88 200 samples, and
        // keeping stereo interleaved yields 64 000.
        #expect(abs(extracted.duration - Self.seconds) < 0.1)
        #expect(extracted.samples.count > 31_000 && extracted.samples.count < 33_000)

        // The decoder owns the downmix (plain channel mean — ffmpeg `-ac 1`'s
        // behaviour), so the level is pinned exactly, not in a band: identical
        // stereo channels must average back to the source's 0.5. AVFoundation's
        // own equal-power mixdown would return 0.5×√2 ≈ 0.707 here, and a
        // one-channel-only mix would return 0.25 — both fail this band.
        let peak = extracted.samples.map { abs($0) }.max() ?? 0
        #expect(peak > 0.3 && peak < 0.7)
        let rms = TranscriptFilter.rms(extracted.samples)
        #expect(rms > 0.2 && rms < 0.5)
    }

    @Test("16 kHz mono WAV decodes with its length intact")
    func decodesNativeRateWithoutConversion() async throws {
        let url = tempURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeWav(at: url, sampleRate: 16_000, channels: 1)

        let extracted = try await AudioFileDecoder.extractSamples(at: url)

        // Already Whisper-native, so the converter is a no-op: 32 000 samples
        // with only boundary slop allowed.
        #expect(abs(extracted.duration - Self.seconds) < 0.05)
        #expect(extracted.samples.count > 31_800 && extracted.samples.count < 32_200)
        let rms = TranscriptFilter.rms(extracted.samples)
        #expect(rms > 0.25 && rms < 0.45)
    }

    // MARK: - Error paths

    @Test("containers AVFoundation cannot demux are rejected with the CLI fallback")
    func rejectsUnsupportedContainers() async throws {
        let url = tempURL("mkv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("not really a matroska".utf8).write(to: url)

        await #expect(throws: AudioDecodeError.unsupportedContainer("mkv")) {
            try await AudioFileDecoder.extractSamples(at: url)
        }
    }

    @Test("isUnsupportedContainer knows the matrix and ignores case")
    func flagsUnsupportedContainers() {
        #expect(AudioFileDecoder.isUnsupportedContainer(tempURL("mkv")))
        #expect(AudioFileDecoder.isUnsupportedContainer(tempURL("avi")))
        #expect(AudioFileDecoder.isUnsupportedContainer(
            FileManager.default.temporaryDirectory.appendingPathComponent("movie.MKV")))
        #expect(!AudioFileDecoder.isUnsupportedContainer(tempURL("mp4")))
        #expect(!AudioFileDecoder.isUnsupportedContainer(tempURL("wav")))
        #expect(!AudioFileDecoder.isUnsupportedContainer(tempURL("mov")))
    }

    @Test("a missing file throws fileNotFound")
    func missingFileThrows() async {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-decoder-absent-\(UUID().uuidString).wav")

        await #expect(throws: AudioDecodeError.fileNotFound(url)) {
            try await AudioFileDecoder.extractSamples(at: url)
        }
    }

    @Test("an unreadable file surfaces as a decode error, not a crash or hang")
    func corruptFileThrows() async throws {
        let url = tempURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data().write(to: url)   // zero bytes: opens, then has no track to read

        // Deliberately not pinned to a case: exactly where AVFoundation gives
        // up on a zero-byte file is its business, and the contract under test
        // is "a clear error, not garbage samples".
        await #expect(throws: AudioDecodeError.self) {
            try await AudioFileDecoder.extractSamples(at: url)
        }
    }
}