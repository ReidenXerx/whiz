import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// One video frame as a JPEG — what whiz otherwise asks ffmpeg for
/// (`-ss T -i video -frames:v 1 -q:v Q -vf scale=W:-1`).
enum FrameExtractor {
    static func extract(from input: URL, at seconds: Double, to output: URL,
                        width: Int, quality: Int) async throws {
        let asset = AVURLAsset(url: input)
        let duration: CMTime
        do {
            guard try await !asset.loadTracks(withMediaType: .video).isEmpty else {
                throw MediaError("\(input.lastPathComponent) has no video track")
            }
            duration = try await asset.load(.duration)
        } catch let error as MediaError {
            throw error
        } catch {
            throw MediaError(AudioExtractor.unreadable(input, error))
        }
        // Past the end is a failure, never the last frame passed off as the
        // one asked for (whiz's wave-3 rule for ffmpeg's silent rc 0).
        let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        guard duration.isNumeric, time < duration else {
            throw MediaError(String(format: "%.3fs is past the end of %@ (%.3fs)",
                                    seconds, input.lastPathComponent, duration.seconds))
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true   // rotated phone video stays upright
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let image: CGImage
        do {
            image = try await generator.image(at: time).image
        } catch {
            throw MediaError("no frame at \(String(format: "%.3f", seconds))s in \(input.lastPathComponent): "
                             + ((error as NSError).localizedFailureReason ?? error.localizedDescription))
        }

        let scaled = try width > 0 && width != image.width ? resize(image, toWidth: width) : image
        try writeAtomically(to: output) { temporary in
            guard let destination = CGImageDestinationCreateWithURL(
                temporary as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
                throw MediaError("cannot write \(output.path)")
            }
            CGImageDestinationAddImage(destination, scaled, [
                kCGImageDestinationLossyCompressionQuality: jpegQuality(forQScale: quality),
            ] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else {
                throw MediaError("cannot encode the frame as JPEG")
            }
        }
    }

    /// ffmpeg's `-q:v` runs 2 (best) to 31 (worst); ImageIO's quality runs
    /// 1 (best) to 0. q 2, whiz's default, maps to 0.9 — visually lossless
    /// for slides and screen text without ffmpeg-sized files.
    static func jpegQuality(forQScale q: Int) -> Double {
        let clamped = Double(min(max(q, 2), 31))
        return 0.9 - (clamped - 2) / 29 * 0.8
    }

    /// `scale=W:-1`: exactly `width` pixels wide, aspect kept.
    static func resize(_ image: CGImage, toWidth width: Int) throws -> CGImage {
        let height = max(1, Int((Double(image.height) * Double(width) / Double(image.width)).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw MediaError("cannot scale the frame to \(width) px")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let scaled = context.makeImage() else { throw MediaError("cannot scale the frame to \(width) px") }
        return scaled
    }
}
