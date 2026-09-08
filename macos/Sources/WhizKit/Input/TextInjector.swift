import AppKit

/// Types transcribed text into whatever app has keyboard focus.
///
/// Ported from `macos_inject.py`, keeping its central decision: split per
/// utterance, not per character. ASCII goes through synthesised keystrokes;
/// anything non-ASCII goes through the clipboard and a ⌘V, because CGEvent
/// keystrokes cannot emit arbitrary Unicode. Russian is entirely non-ASCII, so
/// in practice dictation takes the paste path.
///
/// Both paths need Accessibility. Without it CGEvent posting silently does
/// nothing — no error, no exception, just no text.
enum TextInjector {

    /// Above this many characters, one paste beats hundreds of key events.
    static let pasteThreshold = 120

    /// Exposes the chunking for tests — splitting a grapheme cluster produces
    /// mojibake, and that is not observable from outside otherwise.
    static func chunksForTesting(_ text: String) -> [String] { text.chunked(into: 16) }

    static func type(_ text: String) {
        guard !text.isEmpty else { return }

        // Without Accessibility, CGEvent posting does nothing at all — no
        // error, no exception, no text. That silence is the single most
        // confusing failure this app has, so name it explicitly.
        if !Permissions.isAccessibilityTrusted {
            Log.ui.error(
                "injection SKIPPED: Accessibility not granted — text is discarded")
            return
        }

        // Log where the text is going. "Nothing appeared" almost always means
        // it went somewhere unexpected — whichever app had focus when the
        // hotkey fired, which is not necessarily the one being looked at.
        // Non-ASCII no longer forces the clipboard: Unicode events carry
        // Cyrillic as happily as Latin. Paste is kept for long text, where one
        // ⌘V beats hundreds of synthesised events, and because a few apps
        // handle synthetic key events poorly.
        let usePaste = text.count > Self.pasteThreshold
        let target = NSWorkspace.shared.frontmostApplication?.localizedName ?? "unknown"
        let method = usePaste ? "paste" : "keystroke"
        Log.ui.notice(
            "injecting \(text.count) chars via \(method, privacy: .public) into \(target, privacy: .public)")
        usePaste ? paste(text) : keystroke(text)
    }

    // MARK: - Keystroke path

    /// Type `text` by attaching the literal characters to the events.
    ///
    /// `CGEventKeyboardSetUnicodeString` makes each event carry the exact
    /// string, so the active keyboard layout never gets to reinterpret it. The
    /// previous version posted *virtual keycodes* from a US-QWERTY table, which
    /// three separate bugs came out of:
    ///
    /// 1. A Russian layout rendered those keycodes as Cyrillic — "WHAT ARE YOU
    ///    DOING" arrived as "ЦРФЕ ФКУ НЩГ ВЩШТП".
    /// 2. Characters missing from the table (`?` among them) triggered a
    ///    fallback that pasted the whole string *after* part of it had already
    ///    been typed, so the text appeared twice.
    /// 3. Shift was applied by setting `.maskShift` but never cleared, so flags
    ///    leaked from ambient state: `'` became `"`, `.` became `>`.
    ///
    /// None of that can happen when the character travels with the event. There
    /// is no keycode table, no shift logic and no unmappable character.
    private static func keystroke(_ text: String) {
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }

        // Chunked because the payload is a fixed-size UniChar buffer, and a
        // whole utterance would not fit. The size is conservative rather than
        // probed — this runs once per utterance, so the extra events cost
        // nothing measurable.
        for chunk in text.chunked(into: 16) {
            let utf16 = Array(chunk.utf16)
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
                  let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
            else { continue }

            // Explicitly empty: any inherited modifier would be applied on top
            // of the literal string.
            down.flags = []
            up.flags = []
            utf16.withUnsafeBufferPointer { buffer in
                down.keyboardSetUnicodeString(stringLength: buffer.count, unicodeString: buffer.baseAddress)
                up.keyboardSetUnicodeString(stringLength: buffer.count, unicodeString: buffer.baseAddress)
            }
            down.post(tap: .cghidEventTap)
            up.post(tap: .cghidEventTap)
            // Some apps drop events that arrive too fast.
            usleep(2_000)
        }
    }

    // MARK: - Paste path

    private static func paste(_ text: String) {
        let pasteboard = NSPasteboard.general
        // TODO(phase 2): save and restore the user's clipboard. The Python
        // version left it clobbered too, but it is a real papercut.
        pasteboard.clearContents()
        let wrote = pasteboard.setString(text, forType: .string)
        // Separates "the clipboard write failed" from "the paste keystroke did
        // not land" — otherwise both look identical from the outside.
        Log.ui.notice("pasteboard write: \(wrote, privacy: .public)")

        // Let the clipboard write settle before the paste reads it.
        usleep(10_000)

        guard let source = CGEventSource(stateID: .hidSystemState) else { return }
        let vKey = Keycodes.v

        guard
            let commandDown = CGEvent(keyboardEventSource: source, virtualKey: Keycodes.command, keyDown: true),
            let vDown = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true),
            let vUp = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false),
            let commandUp = CGEvent(keyboardEventSource: source, virtualKey: Keycodes.command, keyDown: false)
        else { return }

        vDown.flags = .maskCommand
        vUp.flags = .maskCommand

        for event in [commandDown, vDown, vUp, commandUp] {
            event.post(tap: .cghidEventTap)
        }
        Log.ui.notice("posted paste keystroke")

        usleep(50_000)
    }
}

/// US QWERTY virtual keycodes.
///
/// No longer used for typing — text injection carries Unicode on the event, so
/// the active layout cannot reinterpret it. What remains is hotkey parsing:
/// `HotkeySpec` turns "<cmd>+<shift>+." into a keycode for
/// `RegisterEventHotKey`, which is genuinely keycode-based, plus the ⌘V the
/// paste path posts.
///
/// The shifted-character set that lived here went with the typing path: shift
/// state was what turned `'` into `"` and `.` into `>`, because flags were set
/// but never cleared.
enum Keycodes {

    static let command: CGKeyCode = 0x37
    static let v: CGKeyCode = 9

    private static let map: [Character: CGKeyCode] = [
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
        "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
        "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
        "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29,
        "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37,
        "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
        "n": 45, "m": 46, ".": 47, "`": 50, " ": 49,
        "\n": 36,  // return
        "\t": 48,  // tab
    ]

    static func forCharacter(_ character: Character) -> CGKeyCode? {
        map[character] ?? map[Character(character.lowercased())]
    }
}

private extension String {
    /// Split into chunks of at most `size` characters, without splitting a
    /// grapheme cluster — an emoji or a combining sequence must stay whole or
    /// it arrives as mojibake.
    func chunked(into size: Int) -> [String] {
        guard size > 0, count > size else { return isEmpty ? [] : [self] }
        var chunks: [String] = []
        var current = ""
        for character in self {
            current.append(character)
            if current.count >= size {
                chunks.append(current)
                current = ""
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}
