import CryptoKit
import Foundation

/// What a Whiz license says: who bought it, and which major version it
/// covers. Signed, so it cannot be edited; checked on the Mac, so the app
/// never needs a server or a connection to know it is licensed.
public struct License: Codable, Equatable, Sendable {
    /// The order it came from — the shop's order number, or anything
    /// unique for a key given away by hand.
    public var id: String
    /// Who it is for, shown in the app ("Licensed to …").
    public var name: String
    /// When it was issued, `yyyy-MM-dd`.
    public var issued: String
    /// The newest major version it unlocks: a license for 1 covers every
    /// 1.x, and the 0.x releases before them. A future 2.0 can ask for a
    /// new license; it never takes away the version that was bought.
    public var major: Int
    /// Always "whiz": a key signed for another product does not open this one.
    public var product: String

    public init(id: String, name: String, issued: String, major: Int, product: String = LicenseKey.product) {
        self.id = id
        self.name = name
        self.issued = issued
        self.major = major
        self.product = product
    }
}

/// Why a pasted key was not accepted, in words for the person who pasted it.
public enum LicenseError: Error, Equatable, LocalizedError, Sendable {
    case malformed
    case invalidSignature
    case wrongProduct
    case olderVersion(licensed: Int, app: Int)

    public var errorDescription: String? {
        switch self {
        case .malformed:
            return "That is not a Whiz license key. Paste the whole key from your purchase email, starting with WHIZ1-."
        case .invalidSignature:
            return "This license key is not valid. Check that it was pasted completely, or contact support."
        case .wrongProduct:
            return "This key is for a different product."
        case .olderVersion(let licensed, let app):
            return "This key is for Whiz \(licensed). This is Whiz \(app), which needs an upgrade license."
        }
    }
}

/// The text form of a license:
///
///     WHIZ1-<base64url(payload JSON)>.<base64url(Ed25519 signature)>
///
/// The signature covers the payload bytes exactly as encoded, so nothing
/// depends on JSON key order or spacing. Whitespace anywhere in a pasted key
/// is ignored — email clients wrap long lines.
///
/// Keys are signed with a private key that only the seller holds
/// (`whiz-license keygen` / `issue`); the app carries the public half.
public enum LicenseKey {
    public static let prefix = "WHIZ1-"
    public static let product = "whiz"

    /// Whiz's public key. The private half lives only on the seller's Mac
    /// (~/.config/whiz-license/signing.key) — lose it and no new key made
    /// for this build will ever verify, so it is backed up off the Mac.
    public static let publicKeyBase64 = "P4Gfn1Y1QlI2rvhWuKHnRYBpA0eCAsApxTW+uY8Lnk4="

    public static var publicKey: Curve25519.Signing.PublicKey {
        // A malformed constant is a build mistake, caught by the test suite.
        try! Curve25519.Signing.PublicKey(rawRepresentation: Data(base64Encoded: publicKeyBase64)!)
    }

    /// Sign `license` into its text form.
    public static func issue(_ license: License, signingKey: Curve25519.Signing.PrivateKey) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = try encoder.encode(license)
        let signature = try signingKey.signature(for: payload)
        return prefix + base64url(payload) + "." + base64url(signature)
    }

    /// Check a pasted key and return what it licenses.
    ///
    /// `appMajor` is this app's major version, at least 1: 0.x releases are
    /// the run-up to 1.0 and take the same license.
    public static func verify(_ text: String,
                              publicKey: Curve25519.Signing.PublicKey = LicenseKey.publicKey,
                              appMajor: Int) throws -> License {
        let compact = text.filter { !$0.isWhitespace }
        guard compact.uppercased().hasPrefix(prefix) else { throw LicenseError.malformed }
        let body = compact.dropFirst(prefix.count)
        let parts = body.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let payload = unbase64url(String(parts[0])),
              let signature = unbase64url(String(parts[1])) else {
            throw LicenseError.malformed
        }
        guard publicKey.isValidSignature(signature, for: payload) else {
            throw LicenseError.invalidSignature
        }
        guard let license = try? JSONDecoder().decode(License.self, from: payload) else {
            // Signed by us but unreadable: a format from a newer issuer.
            throw LicenseError.malformed
        }
        guard license.product == product else { throw LicenseError.wrongProduct }
        let needed = max(1, appMajor)
        guard license.major >= needed else {
            throw LicenseError.olderVersion(licensed: license.major, app: needed)
        }
        return license
    }

    static func base64url(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func unbase64url(_ text: String) -> Data? {
        var base64 = text
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        base64 += String(repeating: "=", count: (4 - base64.count % 4) % 4)
        return Data(base64Encoded: base64)
    }
}
