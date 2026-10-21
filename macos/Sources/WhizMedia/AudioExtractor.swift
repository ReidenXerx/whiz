import AVFoundation
import Foundation

/// A recording's audio as the 16 kHz mono 16-bit PCM WAV whisper and the
/// diarizer read — what whiz otherwise asks ffmpeg for
/// (`-vn -ac 1 -ar 16000 -c:a pcm_s16le`).
///
/// One difference from ffmpeg, on purpose: every audio track is mixed in.
/// ffmpeg takes a single stream, and a screen recording that keeps the
/// microphone and the system audio on separate tracks would lose one side
/// of the conversation.
enum AudioExtractor {
    static let sampleRate = 16_000

    static func extract(from input: URL, to output: URL, comment: String?) async throws {
        let asset = AVURLAsset(url: input)
        let tracks: [AVAssetTrack]
        var nativeRate = 0.0
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
            for track in tracks {
                for format in try await track.load(.formatDescriptions) {
                    if let description = CMAudioFormatDescriptionGetStreamBasicDescription(format) {
                        nativeRate = max(nativeRate, description.pointee.mSampleRate)
                    }
                }
            }
        } catch {
            throw MediaError(unreadable(input, error))
        }
        guard !tracks.isEmpty else { throw MediaError("\(input.lastPathComponent) has no audio track") }
        guard nativeRate > 0 else { throw MediaError("\(input.lastPathComponent): audio has no sample rate") }

        // Decoded at the recording's own rate and resampled by
        // AVAudioConverter, not by the reader: the reader's own 48 → 16 kHz
        // conversion runs fast (about 40 extra samples a minute), which
        // drifts the transcript away from the video over a long meeting.
        let decoded = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: nativeRate,
                                    channels: 1, interleaved: true)!
        let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(sampleRate),
                                   channels: 1, interleaved: true)!
        guard let converter = AVAudioConverter(from: decoded, to: target) else {
            throw MediaError("cannot convert \(Int(nativeRate)) Hz audio to \(sampleRate) Hz")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw MediaError(unreadable(input, error))
        }
        let mix = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: nativeRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ])
        mix.alwaysCopiesSampleData = false
        guard reader.canAdd(mix) else { throw MediaError("cannot decode the audio of \(input.lastPathComponent)") }
        reader.add(mix)

        try writeAtomically(to: output) { temporary in
            let wav = try WAVWriter(url: temporary, sampleRate: sampleRate, comment: comment)
            guard reader.startReading() else {
                throw MediaError(unreadable(input, reader.error))
            }
            let resampler = Resampler(converter: converter, output: target, wav: wav)
            while let sample = mix.copyNextSampleBuffer() {
                defer { CMSampleBufferInvalidate(sample) }
                let frames = CMSampleBufferGetNumSamples(sample)
                guard frames > 0,
                      let buffer = AVAudioPCMBuffer(pcmFormat: decoded, frameCapacity: AVAudioFrameCount(frames))
                else { continue }
                buffer.frameLength = AVAudioFrameCount(frames)
                let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
                    sample, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
                guard status == noErr else { throw MediaError("audio decode failed (\(status))") }
                try resampler.feed(buffer)
            }
            // A decoder that gives up halfway is a failure, never a WAV
            // holding the first part of the meeting.
            if reader.status == .failed {
                throw MediaError(unreadable(input, reader.error))
            }
            try resampler.feed(nil)
            try wav.finish()
        }
    }

    /// AVFoundation's errors read "The operation could not be completed";
    /// the useful part is which formats it does not open.
    static func unreadable(_ input: URL, _ error: Error?) -> String {
        let reason = error.map { ($0 as NSError).localizedFailureReason ?? $0.localizedDescription } ?? "unknown error"
        return "macOS cannot read \(input.lastPathComponent) (\(reason)). "
            + "Formats such as .mkv and .webm need ffmpeg: install it (brew install ffmpeg) and whiz uses it instead."
    }
}

/// Pushes decoded buffers through the converter into the WAV; `nil` is
/// the end of the stream, which flushes the converter's tail.
private final class Resampler {
    let converter: AVAudioConverter
    let output: AVAudioFormat
    let wav: WAVWriter

    init(converter: AVAudioConverter, output: AVAudioFormat, wav: WAVWriter) {
        self.converter = converter
        self.output = output
        self.wav = wav
    }

    func feed(_ input: AVAudioPCMBuffer?) throws {
        var consumed = false
        while true {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: 8192) else {
                throw MediaError("out of memory")
            }
            var error: NSError?
            let status = converter.convert(to: chunk, error: &error) { _, state in
                if let input, !consumed {
                    consumed = true
                    state.pointee = .haveData
                    return input
                }
                state.pointee = input == nil ? .endOfStream : .noDataNow
                return nil
            }
            if status == .error {
                throw MediaError("resampling failed: \(error?.localizedDescription ?? "unknown error")")
            }
            if chunk.frameLength > 0, let samples = chunk.int16ChannelData {
                try wav.append(Data(bytes: samples[0], count: Int(chunk.frameLength) * 2))
            }
            if status == .endOfStream || status == .inputRanDry || chunk.frameLength == 0 { return }
        }
    }
}

/// A streaming RIFF/WAVE writer: header first with placeholder sizes,
/// samples appended, sizes patched on `finish`.
///
/// The optional comment goes in a LIST/INFO/ICMT chunk ahead of the
/// samples, NUL-terminated — exactly where and how ffmpeg's
/// `-metadata comment=` puts it, because whiz finds its own extracted WAVs
/// by scanning the start of the file for that text.
final class WAVWriter {
    private let handle: FileHandle
    private let headerLength: UInt64
    private var dataLength: UInt64 = 0

    init(url: URL, sampleRate: Int, comment: String?) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw MediaError("cannot write \(url.deletingLastPathComponent().path)")
        }
        handle = try FileHandle(forWritingTo: url)
        var header = Data()
        header.append(chunkID: "RIFF"); header.append(uint32: UInt32(0)); header.append(chunkID: "WAVE")
        header.append(chunkID: "fmt "); header.append(uint32: UInt32(16))
        header.append(uint16: UInt16(1))                       // PCM
        header.append(uint16: UInt16(1))                       // mono
        header.append(uint32: UInt32(sampleRate))
        header.append(uint32: UInt32(sampleRate * 2))          // byte rate
        header.append(uint16: UInt16(2))                       // block align
        header.append(uint16: UInt16(16))                      // bits per sample
        if let comment {
            var text = Data(comment.utf8); text.append(0)
            if text.count % 2 == 1 { text.append(0) }  // chunks are word-aligned
            header.append(chunkID: "LIST"); header.append(uint32: UInt32(4 + 8 + text.count))
            header.append(chunkID: "INFO")
            header.append(chunkID: "ICMT"); header.append(uint32: UInt32(text.count)); header.append(text)
        }
        header.append(chunkID: "data"); header.append(uint32: UInt32(0))
        headerLength = UInt64(header.count)
        try handle.write(contentsOf: header)
    }

    func append(_ samples: Data) throws {
        try handle.write(contentsOf: samples)
        dataLength += UInt64(samples.count)
    }

    func finish() throws {
        // RIFF sizes are 32-bit: about 37 hours of 16 kHz mono.
        guard headerLength + dataLength <= UInt64(UInt32.max) else {
            throw MediaError("the recording is too long for one WAV file")
        }
        if dataLength % 2 == 1 { try handle.write(contentsOf: Data([0])) }
        try handle.seek(toOffset: 4)
        var riff = Data(); riff.append(uint32: UInt32(headerLength - 8 + dataLength + dataLength % 2))
        try handle.write(contentsOf: riff)
        try handle.seek(toOffset: headerLength - 4)
        var data = Data(); data.append(uint32: UInt32(dataLength))
        try handle.write(contentsOf: data)
        try handle.close()
    }
}

extension Data {
    mutating func append(chunkID text: String) { append(contentsOf: Array(text.utf8)) }
    mutating func append(uint32 value: UInt32) { Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) } }
    mutating func append(uint16 value: UInt16) { Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) } }
}
