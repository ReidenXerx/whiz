import CWhisper
import Foundation

/// The languages this whisper build actually supports.
///
/// Read from the library rather than hardcoded, so the list cannot drift from
/// the engine when the pinned submodule is bumped. Replaces a free-text field
/// that accepted anything: `xx`, `eng`, `РУ` and empty were all taken without
/// complaint, and whisper silently fell back instead of erroring — so a typo
/// produced quietly degraded transcription with no visible cause.
enum WhisperLanguages {

    struct Language: Identifiable, Hashable {
        var code: String
        var name: String
        var id: String { code }
        var label: String { code == autoCode ? name : "\(name) (\(code))" }
    }

    /// whisper.cpp treats this as "detect the language", so it is not in the
    /// numeric table and has to be added by hand.
    static let autoCode = "auto"

    /// Languages the UI offers.
    ///
    /// whisper knows 100 (see `allSupported`), but only these three have been
    /// tested against real speech here, and an untested language is not a
    /// neutral choice: picking one whisper handles poorly produces confident
    /// nonsense rather than an error. Offering the full list implies a
    /// guarantee that does not exist.
    ///
    /// Widen it as languages get verified — the codes are validated against
    /// whisper's own table below, so a typo here fails a test rather than
    /// shipping.
    static let offeredCodes = ["en", "ru", "uk"]

    /// Auto-detect first, then the vetted languages, alphabetically.
    static let all: [Language] = {
        var out = [Language(code: autoCode, name: "Auto-detect")]
        out.append(contentsOf: allSupported
            .filter { offeredCodes.contains($0.code) }
            .sorted { $0.name < $1.name })
        return out
    }()

    /// Every language this whisper build knows, read from the library so it
    /// cannot drift when the pinned submodule moves.
    static let allSupported: [Language] = {
        let maxID = Int(whisper_lang_max_id())
        var known: [Language] = []
        for id in 0...maxID {
            guard let code = whisper_lang_str(Int32(id)),
                  let full = whisper_lang_str_full(Int32(id)) else { continue }
            known.append(Language(
                code: String(cString: code),
                name: String(cString: full).capitalized))
        }
        return known
    }()

    /// Whether whisper recognises `code`. Used to decide if a value carried
    /// over from a hand-edited config should be shown as-is rather than
    /// silently replaced.
    static func isKnown(_ code: String) -> Bool {
        code == autoCode || whisper_lang_id(code) >= 0
    }

    /// The entry for `code`, inventing one for an unrecognised value so a
    /// hand-edited config is displayed rather than reset behind the user's back.
    static func language(for code: String) -> Language {
        if let match = all.first(where: { $0.code == code }) { return match }
        // A code outside the offered set but known to whisper — e.g. set by the
        // Python CLI, which has no such restriction. Name it properly rather
        // than calling it "Unknown".
        if let match = allSupported.first(where: { $0.code == code }) { return match }
        return Language(code: code, name: code.isEmpty ? "(unset)" : "Unknown")
    }
}
