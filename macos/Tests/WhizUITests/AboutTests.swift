import Foundation
import Testing
@testable import WhizUI

/// Acknowledgements and updates: what the app reads, and the files in the
/// repository that the release build and the published feed depend on.
@Suite("Acknowledgements and updates")
@MainActor
struct AboutTests {
    /// macos/, from this file's place in it.
    let macos = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    // MARK: - Acknowledgements

    @Test("reads the file build.py writes: a model has no text, a URL may be missing")
    func decode() throws {
        let json = """
        {"groups": [
          {"title": "Included in Whiz", "components": [
            {"name": "Python", "version": "3.13.13", "license": "PSF-2.0",
             "url": "https://www.python.org", "summary": "Runs the engine.", "text": "PSF LICENSE\\n"},
            {"name": "libffi", "license": "MIT", "url": null, "summary": "", "text": "MIT\\n"}]},
          {"title": "Models Whiz downloads", "components": [
            {"name": "Whisper", "license": "MIT", "url": "https://github.com/openai/whisper",
             "summary": "Speech recognition."}]}
        ]}
        """
        let acknowledgements = try Acknowledgements.decode(Data(json.utf8))
        #expect(acknowledgements.groups.map(\.title) == ["Included in Whiz", "Models Whiz downloads"])
        #expect(acknowledgements.first?.name == "Python")
        let libffi = acknowledgements.groups[0].components[1]
        #expect(libffi.version == nil && libffi.url == nil && libffi.text == "MIT\n")
        #expect(acknowledgements.groups[1].components[0].text == nil)
    }

    @Test("a build without the file (swift run) has no acknowledgements rather than a crash")
    func missingFile() {
        #expect(Acknowledgements.load(from: Bundle(for: BundleMarker.self)) == nil)
    }

    @Test("every licence file the manifest names is in the repository")
    func manifestFiles() throws {
        let directory = macos.appendingPathComponent("bundle/acknowledgements")
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as! [String: Any]
        var named = 0
        for group in manifest["groups"] as! [[String: Any]] {
            for component in group["components"] as! [[String: Any]] {
                for file in component["files"] as? [String] ?? [] where !file.hasPrefix("@") {
                    #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path),
                            "\(file) is missing")
                    named += 1
                }
            }
        }
        #expect(named >= 10)
    }

    // MARK: - Updates

    @Test("Info.plist points at the v1 feed over HTTPS, with an EdDSA public key")
    func infoPlist() throws {
        let plist = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: macos.appendingPathComponent("bundle/Info.plist")),
            format: nil) as! [String: Any]
        #expect(plist["SUFeedURL"] as? String == "https://duduphudu.app/whiz/updates/v1.xml")
        let key = try #require(plist["SUPublicEDKey"] as? String)
        #expect(Data(base64Encoded: key)?.count == 32, "an Ed25519 public key is 32 bytes")
    }

    @Test("the feed the app reads exists, with the marker publish-update.sh inserts under")
    func feed() throws {
        let feed = try String(contentsOf: macos.deletingLastPathComponent()
            .appendingPathComponent("docs/updates/v1.xml"), encoding: .utf8)
        #expect(feed.contains("xmlns:sparkle=\"http://www.andymatuschak.org/xml-namespaces/sparkle\""))
        #expect(feed.contains("<!-- releases, newest first -->"))
    }

    @Test("no feed in the bundle: the updater stays off instead of failing at launch")
    func updaterWithoutFeed() {
        let bundle = Bundle(for: BundleMarker.self)
        #expect(UpdaterModel.feedURL(in: bundle) == nil)
        let updater = UpdaterModel(bundle: bundle)
        #expect(!updater.isAvailable)
        #expect(!updater.canCheckForUpdates)
        updater.checkForUpdates()   // a no-op, not a crash
    }
}

/// The test bundle: no Acknowledgements.json, no SUFeedURL.
private final class BundleMarker {}
