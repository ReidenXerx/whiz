import CryptoKit
import Foundation
import Testing
@testable import WhizLicense

/// A license must open the app it was sold for and nothing else: no edited
/// key, no key signed by anyone but the seller, no key for another product
/// or an older major version — and a real key must survive an email client
/// wrapping it.
@Suite("License keys")
struct LicenseKeyTests {
    let signer = Curve25519.Signing.PrivateKey()
    var publicKey: Curve25519.Signing.PublicKey { signer.publicKey }
    let license = License(id: "LS-1042", name: "Ada Lovelace", issued: "2026-10-21", major: 1)

    @Test("a key round-trips, with whitespace and line breaks anywhere")
    func roundTrip() throws {
        let key = try LicenseKey.issue(license, signingKey: signer)
        #expect(key.hasPrefix("WHIZ1-"))
        #expect(try LicenseKey.verify(key, publicKey: publicKey, appMajor: 1) == license)
        // Wrapped at 40 columns with indentation, as a mail client might.
        var wrapped = ""
        for (index, character) in key.enumerated() {
            if index > 0, index % 40 == 0 { wrapped += "\n   " }
            wrapped.append(character)
        }
        #expect(try LicenseKey.verify("  \(wrapped)\n", publicKey: publicKey, appMajor: 1) == license)
    }

    @Test("0.x releases take a version 1 license; a later major does not")
    func majorVersions() throws {
        let key = try LicenseKey.issue(license, signingKey: signer)
        #expect(try LicenseKey.verify(key, publicKey: publicKey, appMajor: 0).major == 1)
        #expect(throws: LicenseError.olderVersion(licensed: 1, app: 2)) {
            try LicenseKey.verify(key, publicKey: publicKey, appMajor: 2)
        }
        var upgrade = license
        upgrade.major = 2
        let newer = try LicenseKey.issue(upgrade, signingKey: signer)
        #expect(try LicenseKey.verify(newer, publicKey: publicKey, appMajor: 1).major == 2)
    }

    @Test("an edited payload fails the signature")
    func tampered() throws {
        let key = try LicenseKey.issue(license, signingKey: signer)
        let body = key.dropFirst(LicenseKey.prefix.count).split(separator: ".")
        var payload = LicenseKey.unbase64url(String(body[0]))!
        payload = Data(String(decoding: payload, as: UTF8.self)
            .replacingOccurrences(of: "\"major\":1", with: "\"major\":9").utf8)
        let forged = LicenseKey.prefix + LicenseKey.base64url(payload) + "." + body[1]
        #expect(throws: LicenseError.invalidSignature) {
            try LicenseKey.verify(forged, publicKey: publicKey, appMajor: 1)
        }
    }

    @Test("a key signed by someone else is rejected")
    func otherSigner() throws {
        let key = try LicenseKey.issue(license, signingKey: Curve25519.Signing.PrivateKey())
        #expect(throws: LicenseError.invalidSignature) {
            try LicenseKey.verify(key, publicKey: publicKey, appMajor: 1)
        }
    }

    @Test("another product's key is rejected even with a valid signature")
    func wrongProduct() throws {
        let key = try LicenseKey.issue(License(id: "1", name: "A", issued: "2026-10-21", major: 1, product: "mynah"),
                                       signingKey: signer)
        #expect(throws: LicenseError.wrongProduct) {
            try LicenseKey.verify(key, publicKey: publicKey, appMajor: 1)
        }
    }

    @Test("garbage, truncation and missing parts read as malformed")
    func malformed() throws {
        let key = try LicenseKey.issue(license, signingKey: signer)
        for text in ["", "hello", "WHIZ1-", "WHIZ1-abc", "WHIZ1-a.b.c", "WHIZ1-***.***"] {
            #expect(throws: LicenseError.malformed) { try LicenseKey.verify(text, publicKey: publicKey, appMajor: 1) }
        }
        // A key cut short loses the end of its signature.
        #expect(throws: (any Error).self) {
            try LicenseKey.verify(String(key.dropLast(6)), publicKey: publicKey, appMajor: 1)
        }
    }

    @Test("the store's keys verify here, and both issuers encode the same payload (store/test/vector.js)")
    func sharedVector() throws {
        // Test-only seed: bytes 1…32. The store's JS signer produced `key`.
        let seed = Data(base64Encoded: "AQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyA=")!
        let signer = try Curve25519.Signing.PrivateKey(rawRepresentation: seed)
        #expect(signer.publicKey.rawRepresentation.base64EncodedString() == "ebVWLo/mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ=")
        let license = License(id: "txn_01testvector00000000000000", name: "Олена O'Brien / Ølsen",
                              issued: "2026-10-21", major: 1)
        let key = "WHIZ1-eyJpZCI6InR4bl8wMXRlc3R2ZWN0b3IwMDAwMDAwMDAwMDAwMCIsImlzc3VlZCI6IjIwMjYtMTAtMjEiLCJtYWpvciI6MSwibmFtZSI6ItCe0LvQtdC90LAgTydCcmllbiAvIMOYbHNlbiIsInByb2R1Y3QiOiJ3aGl6In0.i0o0drm-XCCvUlxp191W7KwE0uuxs6DAtCipVpACyDZmgityrQZ6PdNcoJgA58OO6gZ00lInoYBoCM7YtqLiAg"
        #expect(try LicenseKey.verify(key, publicKey: signer.publicKey, appMajor: 1) == license)
        // CryptoKit signs with added randomness, so the signatures differ;
        // the payload bytes must not, and this side's key must verify too.
        let ours = try LicenseKey.issue(license, signingKey: signer)
        #expect(ours.split(separator: ".").first == key.split(separator: ".").first)
        #expect(try LicenseKey.verify(ours, publicKey: signer.publicKey, appMajor: 1) == license)
    }

    @Test("the public key built into the app is a valid Ed25519 key")
    func builtInKey() {
        #expect(LicenseKey.publicKey.rawRepresentation.count == 32)
    }
}
