import SwiftUI
import AppKit

/// The few visual constants the app shares, so every card, chip and speaker
/// color comes from one place instead of a dozen inline guesses.
enum Theme {
    /// merge.py's `_SPEAKER_COLORS`, same order: a speaker keeps its color
    /// between this window and the HTML transcript the CLI wrote.
    static let speakerHex: [UInt32] = [
        0xe74c3c, 0x3498db, 0x2ecc71, 0xf39c12, 0x9b59b6,
        0x1abc9c, 0xe67e22, 0x34495e, 0x16a085, 0xd35400,
    ]

    /// A speaker color that stays legible in dark mode (#34495e on a dark
    /// background is nearly invisible): lifted toward white under Dark Aqua.
    static func speakerColor(_ index: Int) -> Color {
        let hex = speakerHex[((index % speakerHex.count) + speakerHex.count) % speakerHex.count]
        return Color(nsColor: NSColor(name: nil) { appearance in
            let base = NSColor(srgbRed: CGFloat((hex >> 16) & 0xff) / 255,
                               green: CGFloat((hex >> 8) & 0xff) / 255,
                               blue: CGFloat(hex & 0xff) / 255,
                               alpha: 1)
            let dark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return dark ? (base.blended(withFraction: 0.3, of: .white) ?? base) : base
        })
    }

    static let cardRadius: CGFloat = 12
    /// The reading column: long lines are hard to follow.
    static let readingWidth: CGFloat = 760
}

/// A raised surface: content on the control background with a hairline edge.
struct CardModifier: ViewModifier {
    var padding: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                    .strokeBorder(Color(nsColor: .separatorColor).opacity(0.7), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.04), radius: 2, y: 1)
    }
}

extension View {
    func card(padding: CGFloat = 16) -> some View {
        modifier(CardModifier(padding: padding))
    }
}

/// Small caps-ish section header used above cards.
struct SectionLabel: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
    }
}

/// The empty state, macOS 13 style (ContentUnavailableView is 14+).
struct EmptyState<Actions: View>: View {
    let symbol: String
    let title: String
    let message: String
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.tertiary)
                .padding(.bottom, 4)
            Text(title)
                .font(.title3.weight(.semibold))
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
            actions()
                .padding(.top, 6)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

extension EmptyState where Actions == EmptyView {
    init(symbol: String, title: String, message: String) {
        self.init(symbol: symbol, title: title, message: message) { EmptyView() }
    }
}

/// A colored dot + name: how a speaker is shown everywhere.
struct SpeakerBadge: View {
    let name: String
    let colorIndex: Int?

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(colorIndex.map(Theme.speakerColor) ?? Color.secondary)
                .frame(width: 8, height: 8)
            Text(name)
                .fontWeight(.semibold)
                .foregroundStyle(colorIndex.map(Theme.speakerColor) ?? Color.secondary)
        }
    }
}

enum Format {
    /// `1:05`, `12:40`, `1:02:03` — the transcript's timestamp column.
    static func clock(_ seconds: Double) -> String {
        TranscriptionViewModel.formatDuration(seconds)
    }

    /// "8 min", "1 h 12 min", "45 s" — for headers, not timestamps.
    static func spokenDuration(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        if total < 60 { return "\(total) s" }
        let hours = total / 3600
        let minutes = (total % 3600 + 30) / 60
        if hours == 0 { return "\(minutes) min" }
        return minutes == 0 ? "\(hours) h" : "\(hours) h \(minutes) min"
    }

    /// "about 3 min left" / "less than a minute left".
    static func remaining(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return "less than a minute left" }
        return "about \(spokenDuration(seconds)) left"
    }

    static func fileSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// "Today at 17:50", "Yesterday at 09:12", "Oct 3 at 14:00".
    static func when(_ date: Date) -> String {
        let calendar = Calendar.current
        let time = date.formatted(date: .omitted, time: .shortened)
        if calendar.isDateInToday(date) { return "Today at \(time)" }
        if calendar.isDateInYesterday(date) { return "Yesterday at \(time)" }
        return "\(date.formatted(.dateTime.month(.abbreviated).day())) at \(time)"
    }
}

/// What kind of media a path is, for icons and labels.
enum MediaKind {
    case video, audio

    init(url: URL) {
        let videoExtensions: Set<String> = ["mov", "mp4", "m4v", "mkv", "avi", "webm", "mpg", "mpeg", "wmv", "flv", "3gp", "ts", "mts"]
        self = videoExtensions.contains(url.pathExtension.lowercased()) ? .video : .audio
    }

    var symbol: String { self == .video ? "film" : "waveform" }
    var label: String { self == .video ? "Video" : "Audio" }
}
