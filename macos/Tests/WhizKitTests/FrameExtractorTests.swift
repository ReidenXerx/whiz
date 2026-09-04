import Testing
import Foundation
import AVFoundation
import AppKit
@testable import WhizKit

// swift-testing (see ConfigTests for the toolchain rationale).
//
// The extractor's contract with the Python pipeline is pinned two ways: the
// manifest payload shape (screenshots.py:write_manifest, which whiz's Python
// side reads back via load_manifest) and actual frame capture — verified with
// a generated video whose screen is a solid color that changes every second,
// so a frame's pixels prove which timestamp it came from.

@Suite("Frame extraction")
struct FrameExtractorTests {

    @Test("manifest matches screenshots.py's payload shape exactly")
    func manifestShape() throws {
        let path = tempURL("frames.json")
        defer { try? FileManager.default.removeItem(at: path) }

        let entries = [
            FrameExtractor.Entry(
                index: 1, start: 1.5, end: 3.0, speaker: "Speaker",
                text: "He said \"hi\" and \"bye\"", frame: "seg0001.jpg"),
            FrameExtractor.Entry(
                index: 2, start: 3.2, end: 5.5, speaker: "Speaker",
                // Text with characters that must not appear raw in JSON.
                text: "tab\tnew\nline", frame: ""),
        ]
        try FrameExtractor.writeManifest(
            entries,
            framesDir: URL(fileURLWithPath: "/tmp/out/clip.frames"),
            to: path)

        // Exact pin: key order, 2-space indent, no trailing newline — the
        // same layout `json.dumps(payload, indent=2)` produces.
        #expect(try String(contentsOf: path, encoding: .utf8) == """
            {
              "version": 1,
              "frames_dir": "clip.frames",
              "count": 2,
              "segments": [
                {
                  "index": 1,
                  "start": 1.5,
                  "end": 3.0,
                  "speaker": "Speaker",
                  "text": "He said \\"hi\\" and \\"bye\\"",
                  "frame": "seg0001.jpg"
                },
                {
                  "index": 2,
                  "start": 3.2,
                  "end": 5.5,
                  "speaker": "Speaker",
                  "text": "tab\\tnew\\nline",
                  "frame": ""
                }
              ]
            }
            """)

        // And it parses, so the Python reader's keys are all present.
        let object = try JSONSerialization.jsonObject(with: Data(try String(contentsOf: path, encoding: .utf8).utf8)) as? [String: Any]
        #expect((object?["version"] as? Int) == 1)
        #expect((object?["count"] as? Int) == 2)
        let segments = object?["segments"] as? [[String: Any]]
        #expect(segments?.count == 2)
        #expect((segments?[1]["frame"] as? String)?.isEmpty == true)
    }

    @Test("frames land at the right timestamps and the manifest aligns")
    func capturesFramesAtSegmentStarts() async throws {
        let video = tempURL("mov")
        defer { try? FileManager.default.removeItem(at: video) }
        try await writeColoredVideo(at: video)
        #expect(await FrameExtractor.hasVideoTrack(video))

        let framesDir = tempURL("frames-dir")
        defer { try? FileManager.default.removeItem(at: framesDir) }

        let base = [
            // Messy whitespace on purpose: the manifest stores it collapsed.
            WhisperBatchTranscriber.Segment(start: 0.5, end: 1.2, text: "  first   words\nsecond line "),
            WhisperBatchTranscriber.Segment(start: 1.5, end: 2.2, text: "green zone"),
            WhisperBatchTranscriber.Segment(start: 2.5, end: 3.0, text: "blue zone"),
            // Beyond the 3 s video: capture fails, the entry stays aligned.
            WhisperBatchTranscriber.Segment(start: 99.0, end: 100.0, text: "beyond the end"),
        ]
        let progressSeen = ProgressRecorder()
        let entries = try await FrameExtractor.extractFrames(
            video: video,
            segments: base.map { LabeledSegment(segment: $0, speaker: "Speaker") },
            into: framesDir,
            onProgress: { progressSeen.record($0) })

        // One entry per segment, always — failures keep empty frame names.
        #expect(entries.count == 4)
        #expect(entries.map(\.frame) == ["seg0001.jpg", "seg0002.jpg", "seg0003.jpg", ""])
        #expect(entries.map(\.index) == [1, 2, 3, 4])
        #expect(entries[0].text == "first words second line")
        #expect(progressSeen.latest.last == 1.0)

        // The screen is a solid color per second: red 0–1, green 1–2,
        // blue 2–3. A frame captured at segment.start must show the color of
        // that second — the pixel check is the timestamp-accuracy proof.
        let red = try centerPixels(ofJPEG: framesDir.appendingPathComponent("seg0001.jpg"))
        #expect(red.red > 0.8 && red.green < 0.2 && red.blue < 0.2)
        let green = try centerPixels(ofJPEG: framesDir.appendingPathComponent("seg0002.jpg"))
        #expect(green.green > 0.8 && green.red < 0.2 && green.blue < 0.2)
        let blue = try centerPixels(ofJPEG: framesDir.appendingPathComponent("seg0003.jpg"))
        #expect(blue.blue > 0.8 && blue.red < 0.2 && blue.green < 0.2)

        // The 1280 cap downsizes but never upsizes: 640×360 stays native.
        guard let nativeRep = try? NSBitmapImageRep(data: Data(contentsOf: framesDir.appendingPathComponent("seg0001.jpg")))
        else { throw FrameExtractorError.writeFailed(framesDir.path) }
        #expect(nativeRep.pixelsWide == 640 && nativeRep.pixelsHigh == 360)

        // And the manifest for these entries parses with the failed capture
        // aligned by index.
        let manifestPath = tempURL("frames.json")
        defer { try? FileManager.default.removeItem(at: manifestPath) }
        try FrameExtractor.writeManifest(entries, framesDir: framesDir, to: manifestPath)
        let object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestPath)) as? [String: Any]
        let segments = object?["segments"] as? [[String: Any]]
        #expect(segments?.count == 4)
        #expect((segments?[3]["frame"] as? String)?.isEmpty == true)
    }

    @Test("audio files have no video track and yield no frames")
    func audioInputYieldsNoFrames() async throws {
        let wav = tempURL("wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        try writeSilentWav(at: wav)

        #expect(await !FrameExtractor.hasVideoTrack(wav))
        let entries = try await FrameExtractor.extractFrames(
            video: wav,
            segments: [LabeledSegment(segment: WhisperBatchTranscriber.Segment(start: 0.1, end: 0.5, text: "hi"), speaker: "Speaker")],
            into: tempURL("frames-dir"))
        #expect(entries.isEmpty)
    }

    // MARK: - Helpers

    /// A 3-second 640×360 H.264 video whose screen is solid red, then green,
    /// then blue — one color per second, 15 fps.
    private func writeColoredVideo(at url: URL) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 640,
            AVVideoHeightKey: 360,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input)
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let fps: Int32 = 15
        let width = 640, height = 360
        let colors: [(UInt8, UInt8, UInt8)] = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for frame in 0..<(3 * fps) {
            // Bounded wait: a failed writer must not loop forever.
            for _ in 0..<5000 where !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(2))
            }
            guard input.isReadyForMoreMediaData else {
                throw FrameExtractorError.writeFailed(url.path)
            }

            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32ARGB, nil, &buffer)
            guard let buffer else { throw FrameExtractorError.writeFailed(url.path) }
            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let color = colors[Int(frame) / Int(fps)]
            for y in 0..<height {
                let row = base + y * bytesPerRow
                for x in 0..<width {
                    let pixel = row + x * 4
                    pixel[0] = 255   // A
                    pixel[1] = color.0
                    pixel[2] = color.1
                    pixel[3] = color.2
                }
            }
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: fps))
        }
        input.markAsFinished()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            writer.finishWriting {
                if writer.status == .completed {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: FrameExtractorError.writeFailed(url.path))
                }
            }
        }
    }

    private func writeSilentWav(at url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1600)
        else { throw FrameExtractorError.writeFailed(url.path) }
        buffer.frameLength = 1600
        try file.write(from: buffer)
    }

    /// Sample of the JPEG's center pixels — solid-color frames make this a
    /// reliable identity check for the timestamp the frame came from.
    private func centerPixels(ofJPEG url: URL) throws -> (red: CGFloat, green: CGFloat, blue: CGFloat) {
        guard let data = try? Data(contentsOf: url), let rep = NSBitmapImageRep(data: data) else {
            throw FrameExtractorError.writeFailed(url.path)
        }
        let points = [(rep.pixelsWide / 3, rep.pixelsHigh / 2),
                      (rep.pixelsWide / 2, rep.pixelsHigh / 2),
                      (2 * rep.pixelsWide / 3, rep.pixelsHigh / 2)]
        var reds: CGFloat = 0, greens: CGFloat = 0, blues: CGFloat = 0
        for (x, y) in points {
            guard let color = rep.colorAt(x: x, y: y) else {
                throw FrameExtractorError.writeFailed(url.path)
            }
            reds += color.redComponent
            greens += color.greenComponent
            blues += color.blueComponent
        }
        return (reds / 3, greens / 3, blues / 3)
    }

    private func tempURL(_ ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-frames-\(UUID().uuidString).\(ext)")
    }
}

/// Lock-guarded collection point for the `@Sendable` progress callback in tests.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []

    func record(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }

    var latest: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}