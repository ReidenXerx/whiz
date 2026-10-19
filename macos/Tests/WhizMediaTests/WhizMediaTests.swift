import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import WhizMedia

/// whiz-media stands in for ffmpeg inside Whiz.app, so these pin what whiz
/// relies on: a 16 kHz mono 16-bit WAV of the right length carrying whiz's
/// marker, the frame at the time asked for (never a nearby one passed off
/// as it), and failure — not a partial file — when it cannot deliver.
///
/// The media is generated here with AVFoundation, so no binary fixtures.
@Suite("whiz-media", .timeLimit(.minutes(1)))
struct WhizMediaTests {
    let directory: URL

    init() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-media-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: - Audio

    @Test("48 kHz stereo AAC becomes a 16 kHz mono WAV of the same length, with the marker")
    func audioFromAAC() async throws {
        let source = directory.appendingPathComponent("talk.m4a")
        try Fixtures.writeTone(to: source, seconds: 3, sampleRate: 48_000, channels: 2)
        let output = directory.appendingPathComponent("talk.wav")

        try await AudioExtractor.extract(from: source, to: output, comment: "whiz-extracted:talk.m4a")

        let wav = try WAVFile(output)
        #expect(wav.format == 1 && wav.channels == 1 && wav.sampleRate == 16_000 && wav.bitsPerSample == 16)
        // Exact rate conversion: AAC padding aside, no drift (a few ms at most).
        #expect(abs(Double(wav.dataLength / 2) / 16_000 - 3) < 0.005)
        // whiz's is_whiz_extracted scans the head for the NUL-terminated marker.
        let head = try Data(contentsOf: output).prefix(4096)
        #expect(head.range(of: Data("whiz-extracted:talk.m4a\u{0}".utf8)) != nil)
        // The tone survived: not silence.
        #expect(wav.rms > 1000)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == ["talk.m4a", "talk.wav"])
    }

    @Test("a file macOS cannot read fails and leaves nothing behind")
    func audioUnreadable() async throws {
        let source = directory.appendingPathComponent("notes.mkv")
        try Data("not a recording".utf8).write(to: source)
        let output = directory.appendingPathComponent("notes.wav")

        await #expect(throws: MediaError.self) {
            try await AudioExtractor.extract(from: source, to: output, comment: nil)
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["notes.mkv"])
    }

    @Test("a video with no audio track says so")
    func audioMissingTrack() async throws {
        let video = directory.appendingPathComponent("silent.mov")
        try await Fixtures.writeColorVideo(to: video, colors: [(255, 0, 0)])
        do {
            try await AudioExtractor.extract(from: video, to: directory.appendingPathComponent("silent.wav"), comment: nil)
            Issue.record("expected a failure")
        } catch let error as MediaError {
            #expect(error.description.contains("no audio track"))
        }
    }

    @Test("WAV header: sizes, word-aligned comment chunk, readable by any RIFF parser")
    func wavHeader() throws {
        let url = directory.appendingPathComponent("h.wav")
        let writer = try WAVWriter(url: url, sampleRate: 16_000, comment: "abc")   // odd + NUL = even
        try writer.append(Data(repeating: 0x01, count: 10))
        try writer.finish()
        let bytes = [UInt8](try Data(contentsOf: url))
        // RIFF(12) + fmt(24) + LIST(8 + INFO 4 + ICMT 8 + "abc\0" 4) + data(8 + 10)
        #expect(bytes.count == 12 + 24 + 24 + 18)
        #expect(le32(bytes, 4) == UInt32(bytes.count - 8))
        #expect(Array(bytes[36..<40]) == Array("LIST".utf8))
        #expect(le32(bytes, 40) == 16)
        #expect(Array(bytes[44..<52]) == Array("INFOICMT".utf8))
        #expect(le32(bytes, 52) == 4)
        #expect(Array(bytes[56..<60]) == Array("abc\u{0}".utf8))
        #expect(Array(bytes[60..<64]) == Array("data".utf8))
        #expect(le32(bytes, 64) == 10)
    }

    // MARK: - Frames

    @Test("the frame at the time asked for, scaled to the width asked for")
    func frameAtTime() async throws {
        let video = directory.appendingPathComponent("slides.mov")
        // One second each: red, green, blue.
        try await Fixtures.writeColorVideo(to: video, colors: [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

        let green = directory.appendingPathComponent("g.jpg")
        try await FrameExtractor.extract(from: video, at: 1.5, to: green, width: 320, quality: 2)
        let image = try Fixtures.loadImage(green)
        #expect(image.width == 320 && image.height == 180)
        let (r, g, b) = try Fixtures.centerPixel(image)
        // Video color (BT.709) to sRGB tints pure green to about (66, 251, 0).
        #expect(g > 200 && r < 110 && b < 110, "center pixel \(r),\(g),\(b)")

        let blue = directory.appendingPathComponent("b.jpg")
        try await FrameExtractor.extract(from: video, at: 2.2, to: blue, width: 0, quality: 2)
        let native = try Fixtures.loadImage(blue)
        #expect(native.width == 640 && native.height == 360)
        let (_, _, blueness) = try Fixtures.centerPixel(native)
        #expect(blueness > 200)
    }

    @Test("past the end is a failure, never the last frame")
    func framePastEnd() async throws {
        let video = directory.appendingPathComponent("short.mov")
        try await Fixtures.writeColorVideo(to: video, colors: [(255, 0, 0)])
        let output = directory.appendingPathComponent("late.jpg")
        do {
            try await FrameExtractor.extract(from: video, at: 30, to: output, width: 0, quality: 2)
            Issue.record("expected a failure")
        } catch let error as MediaError {
            #expect(error.description.contains("past the end"))
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test("ffmpeg's -q:v scale maps onto ImageIO quality, best to worst")
    func jpegQuality() {
        #expect(FrameExtractor.jpegQuality(forQScale: 2) == 0.9)
        #expect(FrameExtractor.jpegQuality(forQScale: 1) == 0.9)
        #expect(abs(FrameExtractor.jpegQuality(forQScale: 31) - 0.1) < 1e-9)
        #expect(FrameExtractor.jpegQuality(forQScale: 10) < FrameExtractor.jpegQuality(forQScale: 5))
    }

    // MARK: - Command line

    @Test("argument parsing: options anywhere, wrong counts and unknown options are usage errors")
    func arguments() async throws {
        let parsed = try Arguments(["frame", "--width", "1280", "in.mov", "1.5", "out.jpg", "--quality", "3"])
        #expect(parsed.positionals == ["in.mov", "1.5", "out.jpg"])
        #expect(try parsed.int("--width") == 1280 && parsed.int("--quality") == 3)
        #expect(throws: UsageError.self) { try parsed.expectPositionals(3, options: ["--width"]) }
        #expect(throws: UsageError.self) { try Arguments(["audio", "in.mov", "--comment"]) }

        #expect(await MediaCommand.run([]) == 2)
        #expect(await MediaCommand.run(["audio", "only-one-argument"]) == 2)
        #expect(await MediaCommand.run(["frame", "a.mov", "soon", "b.jpg"]) == 2)
        #expect(await MediaCommand.run(["transcode"]) == 2)
        #expect(await MediaCommand.run(["audio", directory.appendingPathComponent("missing.mov").path,
                                        directory.appendingPathComponent("x.wav").path]) == 1)
    }

    private func le32(_ bytes: [UInt8], _ at: Int) -> UInt32 {
        UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16 | UInt32(bytes[at + 3]) << 24
    }
}

/// The fields of a PCM WAV the tests check, parsed independently of WAVWriter.
struct WAVFile {
    var format = 0, channels = 0, sampleRate = 0, bitsPerSample = 0, dataLength = 0
    var rms = 0.0

    init(_ url: URL) throws {
        let bytes = [UInt8](try Data(contentsOf: url))
        func u16(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
        var offset = 12
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
            let size = u32(offset + 4)
            if id == "fmt " {
                format = u16(offset + 8); channels = u16(offset + 10)
                sampleRate = u32(offset + 12); bitsPerSample = u16(offset + 22)
            } else if id == "data" {
                dataLength = size
                var sum = 0.0
                let count = size / 2
                for index in stride(from: 0, to: count, by: 1) {
                    let sample = Double(Int16(bitPattern: UInt16(u16(offset + 8 + index * 2))))
                    sum += sample * sample
                }
                rms = count > 0 ? (sum / Double(count)).squareRoot() : 0
            }
            offset += 8 + size + size % 2
        }
    }
}

enum Fixtures {
    /// A 440 Hz tone as AAC in an .m4a.
    static func writeTone(to url: URL, seconds: Double, sampleRate: Double, channels: AVAudioChannelCount) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels)!
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVEncoderBitRateKey: 128_000,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
        let frames = AVAudioFrameCount(seconds * sampleRate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<Int(channels) {
            let samples = buffer.floatChannelData![channel]
            for index in 0..<Int(frames) {
                samples[index] = 0.3 * Float(sin(2 * Double.pi * 440 * Double(index) / sampleRate))
            }
        }
        try file.write(from: buffer)
    }

    /// A 640×360 H.264 .mov showing each color for one second.
    static func writeColorVideo(to url: URL, colors: [(UInt8, UInt8, UInt8)]) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 640, AVVideoHeightKey: 360,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 640, kCVPixelBufferHeightKey as String: 360,
        ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        // 10 fps, so a second is ten frames of one color.
        for (second, color) in colors.enumerated() {
            for tenth in 0..<10 {
                while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
                var pixels: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pixels)
                let buffer = pixels!
                CVPixelBufferLockBaseAddress(buffer, [])
                let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
                let row = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0..<360 {
                    for x in 0..<640 {
                        let pixel = base + y * row + x * 4
                        pixel[0] = color.2; pixel[1] = color.1; pixel[2] = color.0; pixel[3] = 255
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(second * 10 + tenth), timescale: 10))
            }
        }
        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(value: CMTimeValue(colors.count * 10), timescale: 10))
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? MediaError("could not write \(url.path)") }
    }

    static func loadImage(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MediaError("cannot read \(url.path)")
        }
        return image
    }

    static func centerPixel(_ image: CGImage) throws -> (Int, Int, Int) {
        var pixel = [UInt8](repeating: 0, count: 4)
        guard let context = CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw MediaError("no context")
        }
        context.draw(image, in: CGRect(x: -image.width / 2, y: -image.height / 2, width: image.width, height: image.height))
        return (Int(pixel[0]), Int(pixel[1]), Int(pixel[2]))
    }
}
