import AVFoundation
import AVKit
import ImageIO
import QuickLookThumbnailing
import SwiftUI

/// Plays the run's input so the transcript can seek it.
///
/// One player per detail view; `currentTime` drives the "now playing"
/// highlight. The periodic observer is delivered on the main queue, hence
/// `assumeIsolated`.
@MainActor
final class PlaybackController: ObservableObject {
    let player: AVPlayer?
    let kind: MediaKind
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying = false
    private var timeObserver: Any?

    init(url: URL?) {
        if let url, FileManager.default.fileExists(atPath: url.path) {
            player = AVPlayer(url: url)
            kind = MediaKind(url: url)
        } else {
            player = nil
            kind = url.map(MediaKind.init(url:)) ?? .audio
        }
        guard let player else { return }
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.25, preferredTimescale: 600), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self else { return }
                let seconds = time.seconds
                if seconds.isFinite { self.currentTime = seconds }
                self.isPlaying = self.player?.timeControlStatus == .playing
            }
        }
    }

    var isAvailable: Bool { player != nil }

    /// Jump to `seconds` and play from there — clicking a timestamp means
    /// "let me hear this".
    func seek(to seconds: Double) {
        guard let player else { return }
        player.seek(to: CMTime(seconds: max(seconds, 0), preferredTimescale: 600),
                    toleranceBefore: .zero, toleranceAfter: .zero)
        currentTime = seconds
        player.play()
        isPlaying = true
    }

    func togglePlayback() {
        guard let player else { return }
        if player.timeControlStatus == .playing { player.pause() } else { player.play() }
        isPlaying = player.timeControlStatus != .paused
    }

    /// Called when the view goes away: a detail view that is gone must not
    /// keep talking.
    func shutDown() {
        player?.pause()
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        timeObserver = nil
    }
}

/// AVKit's own player view: native controls, scrubbing, full screen.
struct PlayerSurface: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.videoGravity = .resizeAspect
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}

/// CGImage is a CF type the compiler cannot prove Sendable; it is immutable,
/// which is what makes handing it across the task boundary safe.
private struct ImageBox: @unchecked Sendable {
    let image: CGImage
}

/// Decoded thumbnails, cached. Frames are 1280-px JPEGs — decoding the full
/// image for an 80-pt row would be ~5 MB a frame and an hour of video has
/// hundreds of them, so ImageIO decodes straight to the thumbnail size.
@MainActor
final class ThumbnailCache {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSURL, NSImage>()

    init() { cache.countLimit = 400 }

    func cached(_ url: URL, maxPixel: Int) -> NSImage? {
        cache.object(forKey: key(url, maxPixel))
    }

    func image(for url: URL, maxPixel: Int) async -> NSImage? {
        let key = key(url, maxPixel)
        if let hit = cache.object(forKey: key) { return hit }
        let box = await Task.detached(priority: .utility) { () -> ImageBox? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            ] as CFDictionary
            guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options) else { return nil }
            return ImageBox(image: image)
        }.value
        guard let box else { return nil }
        let image = NSImage(cgImage: box.image, size: NSSize(width: box.image.width, height: box.image.height))
        cache.setObject(image, forKey: key)
        return image
    }

    /// A Quick Look thumbnail of any file — a video's poster frame, or the
    /// system's icon for audio.
    func fileThumbnail(for url: URL, size: CGSize) async -> NSImage? {
        let key = key(url, Int(size.width) * 1000 + Int(size.height))
        if let hit = cache.object(forKey: key) { return hit }
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let request = QLThumbnailGenerator.Request(fileAt: url, size: size, scale: scale,
                                                   representationTypes: .thumbnail)
        let box: ImageBox? = await withCheckedContinuation { continuation in
            QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { representation, _ in
                continuation.resume(returning: representation.map { ImageBox(image: $0.cgImage) })
            }
        }
        guard let box else { return nil }
        let image = NSImage(cgImage: box.image, size: size)
        cache.setObject(image, forKey: key)
        return image
    }

    private func key(_ url: URL, _ variant: Int) -> NSURL {
        NSURL(string: url.absoluteString + "#\(variant)") ?? url as NSURL
    }
}

/// A frame or file thumbnail that loads itself.
struct ThumbnailView: View {
    enum Source: Equatable {
        case frame(URL)
        case file(URL)
    }

    let source: Source
    var size: CGSize

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.primary.opacity(0.06))
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: size.width, height: size.height)
                    .clipped()
            }
        }
        .frame(width: size.width, height: size.height)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous)
            .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5))
        .task(id: source) {
            switch source {
            case .frame(let url):
                if let hit = ThumbnailCache.shared.cached(url, maxPixel: maxPixel) {
                    image = hit
                } else {
                    image = await ThumbnailCache.shared.image(for: url, maxPixel: maxPixel)
                }
            case .file(let url):
                image = await ThumbnailCache.shared.fileThumbnail(for: url, size: size)
            }
        }
    }

    private var maxPixel: Int { Int(max(size.width, size.height) * 2) }
}

/// The media's length, read once from the file's header.
enum MediaInfo {
    static func duration(of url: URL) async -> Double? {
        await Task.detached(priority: .utility) { () -> Double? in
            let asset = AVURLAsset(url: url)
            guard let time = try? await asset.load(.duration) else { return nil }
            let seconds = time.seconds
            return seconds.isFinite && seconds > 0 ? seconds : nil
        }.value
    }

    static func fileSize(of url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }
}
