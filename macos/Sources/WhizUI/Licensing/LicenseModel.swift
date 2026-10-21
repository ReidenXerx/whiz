import CryptoKit
import Foundation
import WhizLicense

/// Whether this Mac may start a transcription: licensed, or still within
/// the free trial.
///
/// The trial is a number of finished transcriptions, not days: a meeting
/// app gets used once a week, and a clock would run out before anyone
/// tried it on a real recording. Failed and stopped runs do not count.
///
/// Honor system, on purpose: the key is checked on this Mac with no
/// account or server, and the trial count lives in the app's defaults. The
/// point is to make paying the easy path, not to fight the few who won't.
/// What is never locked: past transcripts, their files, summaries, and the
/// command-line tool.
@MainActor
final class LicenseModel: ObservableObject {
    enum Status: Equatable {
        case licensed(License)
        case trial(remaining: Int)
        case trialOver
    }

    static let freeTranscriptions = 5
    /// Where "Buy Whiz" goes.
    static let purchaseURL = URL(string: "https://duduphudu.app/whiz/#buy")!

    static let licenseDefault = "dev.reidenxerx.whiz.license"
    static let trialUsedDefault = "dev.reidenxerx.whiz.trialUsed"

    @Published private(set) var license: License?
    @Published private(set) var trialUsed: Int
    /// The activation sheet.
    @Published var isActivationPresented = false
    /// It opened because Transcribe was pressed with the trial used up.
    @Published private(set) var openedAtTrialEnd = false

    private let defaults: UserDefaults
    private let publicKey: Curve25519.Signing.PublicKey
    let appMajor: Int

    /// Injectable for tests: a suite instead of the real defaults, a test
    /// signing key, a chosen app version.
    init(defaults: UserDefaults = .standard,
         publicKey: Curve25519.Signing.PublicKey = LicenseKey.publicKey,
         appMajor: Int = LicenseModel.bundleMajor) {
        self.defaults = defaults
        self.publicKey = publicKey
        self.appMajor = appMajor
        trialUsed = max(0, defaults.integer(forKey: Self.trialUsedDefault))
        // Re-checked on every launch: a stored key that no longer verifies
        // (an edited defaults file, a later major version) licenses nothing,
        // but stays stored — going back to the version it was bought for
        // brings it back.
        if let stored = defaults.string(forKey: Self.licenseDefault) {
            license = try? LicenseKey.verify(stored, publicKey: publicKey, appMajor: appMajor)
        }
    }

    /// The app's major version from its Info.plist; 0 outside a bundle.
    nonisolated static var bundleMajor: Int {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
        return Int(version.split(separator: ".").first ?? "0") ?? 0
    }

    var status: Status {
        if let license { return .licensed(license) }
        let remaining = Self.freeTranscriptions - trialUsed
        return remaining > 0 ? .trial(remaining: remaining) : .trialOver
    }

    var isLicensed: Bool { license != nil }
    var canTranscribe: Bool { status != .trialOver }

    /// Called before a run starts. Past the trial it opens the sheet and
    /// says no, so the Transcribe button explains itself instead of
    /// sitting greyed out.
    func requestTranscription() -> Bool {
        if canTranscribe { return true }
        openedAtTrialEnd = true
        isActivationPresented = true
        return false
    }

    func presentActivation() {
        openedAtTrialEnd = false
        isActivationPresented = true
    }

    /// A run finished with results. Licensed runs are not counted.
    func recordTranscription() {
        guard license == nil else { return }
        trialUsed += 1
        defaults.set(trialUsed, forKey: Self.trialUsedDefault)
    }

    /// Check and store a pasted key. Throws `LicenseError` with a reason
    /// to show; nothing is stored unless it verifies.
    @discardableResult
    func activate(_ text: String) throws -> License {
        let verified = try LicenseKey.verify(text, publicKey: publicKey, appMajor: appMajor)
        defaults.set(text.filter { !$0.isWhitespace }, forKey: Self.licenseDefault)
        license = verified
        return verified
    }

    /// Remove the key from this Mac, e.g. before moving it to another one.
    func deactivate() {
        defaults.removeObject(forKey: Self.licenseDefault)
        license = nil
    }
}
