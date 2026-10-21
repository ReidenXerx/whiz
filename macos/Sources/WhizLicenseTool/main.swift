import CryptoKit
import Foundation
import WhizLicense

// whiz-license — the seller's tool for Whiz license keys. Never shipped in
// the app: it signs keys, which only the seller can do.
//
//   whiz-license keygen [--key FILE]
//       Make the signing key (once, ever). Prints the public key to put in
//       Sources/WhizLicense/LicenseKey.swift. Refuses to overwrite.
//   whiz-license issue --id ORDER --name NAME [--major N] [--key FILE]
//       Print a license key for one buyer.
//   whiz-license verify KEY
//       Check a key against the public key built into this tool.
//
// FILE defaults to ~/.config/whiz-license/signing.key. Back it up somewhere
// safe off this Mac (a password manager): without it no new key can be made
// that existing copies of Whiz accept.

let defaultKeyPath = NSString(string: "~/.config/whiz-license/signing.key").expandingTildeInPath

func fail(_ message: String, status: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("whiz-license: \(message)\n".utf8))
    exit(status)
}

func option(_ name: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}

func loadSigningKey(_ path: String) -> Curve25519.Signing.PrivateKey {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8),
          let raw = Data(base64Encoded: text.trimmingCharacters(in: .whitespacesAndNewlines)),
          let key = try? Curve25519.Signing.PrivateKey(rawRepresentation: raw) else {
        fail("cannot read a signing key from \(path) (make one with: whiz-license keygen)")
    }
    return key
}

let arguments = Array(CommandLine.arguments.dropFirst())
switch arguments.first {
case "keygen":
    let path = option("--key", in: arguments) ?? defaultKeyPath
    guard !FileManager.default.fileExists(atPath: path) else {
        fail("\(path) already exists. Keys signed with it would stop working if it were replaced.")
    }
    let key = Curve25519.Signing.PrivateKey()
    let directory = (path as NSString).deletingLastPathComponent
    do {
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        guard FileManager.default.createFile(atPath: path, contents: Data((key.rawRepresentation.base64EncodedString() + "\n").utf8),
                                             attributes: [.posixPermissions: 0o600]) else {
            fail("cannot write \(path)")
        }
    } catch {
        fail("cannot write \(path): \(error.localizedDescription)")
    }
    print("Signing key written to \(path) (readable by you only). Back it up off this Mac.")
    print("Public key, for LicenseKey.publicKeyBase64:")
    print(key.publicKey.rawRepresentation.base64EncodedString())

case "issue":
    guard let id = option("--id", in: arguments), !id.isEmpty,
          let name = option("--name", in: arguments), !name.isEmpty else {
        fail("issue needs --id ORDER and --name NAME", status: 2)
    }
    let major = Int(option("--major", in: arguments) ?? "1") ?? 0
    guard major >= 1 else { fail("--major must be 1 or more", status: 2) }
    let key = loadSigningKey(option("--key", in: arguments) ?? defaultKeyPath)
    guard key.publicKey.rawRepresentation == LicenseKey.publicKey.rawRepresentation else {
        fail("this signing key does not match the public key built into Whiz; keys made with it would be rejected")
    }
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = "yyyy-MM-dd"
    let license = License(id: id, name: name, issued: formatter.string(from: Date()), major: major)
    do {
        print(try LicenseKey.issue(license, signingKey: key))
    } catch {
        fail("could not sign: \(error.localizedDescription)")
    }

case "verify":
    guard arguments.count >= 2 else { fail("verify needs a key", status: 2) }
    do {
        let license = try LicenseKey.verify(arguments.dropFirst().joined(), appMajor: 1)
        print("Valid: \(license.name) · order \(license.id) · Whiz \(license.major) · issued \(license.issued)")
    } catch {
        fail(error.localizedDescription)
    }

default:
    fail("usage: whiz-license keygen | issue --id ORDER --name NAME [--major N] | verify KEY", status: 2)
}
