import AppKit
import AVFoundation
import Combine
import CoreGraphics

/// What the Record view remembers between recordings.
struct RecordingPreferences: Codable, Equatable, Sendable {
    /// "default" follows the system's input device, "off" records none,
    /// anything else is a microphone's unique ID.
    var microphone = "default"
    var systemAudio = true
    var screen = false
    /// 0 = the main display.
    var display: UInt32 = 0
    var transcribeWhenDone = true
    /// nil = `RecordingPreferences.defaultFolder`.
    var folderPath: String?

    static let key = "dev.reidenxerx.whiz.recording"

    /// ~/Movies/Whiz: where macOS puts screen recordings, and — unlike
    /// Documents and Desktop — never synced to iCloud by "Desktop &
    /// Documents Folders", which would upload calls the buyer chose Whiz to
    /// keep on the Mac.
    static var defaultFolder: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Movies/Whiz", isDirectory: true)
    }

    var folder: URL {
        folderPath.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? Self.defaultFolder
    }

    static func load(from defaults: UserDefaults = .standard) -> RecordingPreferences {
        guard let data = defaults.data(forKey: key),
              let preferences = try? JSONDecoder().decode(RecordingPreferences.self, from: data)
        else { return RecordingPreferences() }
        return preferences
    }

    func save(to defaults: UserDefaults = .standard) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.key)
    }

    /// A recording Whiz made: an audio file in the recordings folder. Audio
    /// is not diarized unless asked (video is), and a recorded call always
    /// has more than one voice.
    func isRecording(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "m4a"
            && url.deletingLastPathComponent().standardizedFileURL.path == folder.standardizedFileURL.path
    }
}

/// The level meters, apart from the model: they change ten times a second,
/// and only the meters should redraw that often.
@MainActor
final class RecordingMeter: ObservableObject {
    @Published var microphone: Float = 0
    @Published var system: Float = 0
}

/// The Record button: sources, permissions, and one recording at a time.
@MainActor
final class RecorderModel: ObservableObject {
    static let shared = RecorderModel()

    enum State: Equatable {
        case idle
        case starting
        case recording(since: Date)
        case stopping
    }

    struct Device: Identifiable, Hashable {
        var id: String
        var name: String
    }

    struct Display: Identifiable, Hashable {
        var id: UInt32
        var name: String
    }

    @Published private(set) var state: State = .idle
    /// Whole seconds since the recording started.
    @Published private(set) var elapsed = 0
    @Published var preferences: RecordingPreferences {
        didSet { if preferences != oldValue { preferences.save(to: defaults) } }
    }
    @Published private(set) var microphones: [Device] = []
    @Published private(set) var displays: [Display] = []
    @Published private(set) var screenAllowed = false
    @Published private(set) var microphoneAccess: AVAuthorizationStatus = .notDetermined
    /// Why the last recording did not start, or stopped early.
    @Published var errorMessage: String?
    /// Said while recording, without stopping it.
    @Published private(set) var warning: String?
    @Published private(set) var lastRecording: URL?
    let meter = RecordingMeter()

    /// The finished recording, to transcribe when the preference says so.
    var onFinished: ((URL) -> Void)?

    private let defaults: UserDefaults
    private var recorder: CallRecorder?
    private var ticker: Task<Void, Never>?
    private var activity: NSObjectProtocol?
    private var deviceObservers: [NSObjectProtocol] = []

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        preferences = RecordingPreferences.load(from: defaults)
        refreshDevices()
        refreshPermissions()
        let center = NotificationCenter.default
        for name in [AVCaptureDevice.wasConnectedNotification, AVCaptureDevice.wasDisconnectedNotification,
                     NSApplication.didChangeScreenParametersNotification] {
            deviceObservers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.refreshDevices() }
            })
        }
    }

    var isRecording: Bool {
        if case .recording = state { return true }
        return false
    }

    var isBusy: Bool { state != .idle }

    // MARK: - Devices and permissions

    func refreshDevices() {
        microphones = Self.discoverMicrophones()
        displays = NSScreen.screens.compactMap { screen in
            guard let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber
            else { return nil }
            return Display(id: number.uint32Value, name: screen.localizedName)
        }
    }

    func refreshPermissions() {
        screenAllowed = CGPreflightScreenCaptureAccess()
        microphoneAccess = AVCaptureDevice.authorizationStatus(for: .audio)
    }

    private static func discoverMicrophones() -> [Device] {
        let types: [AVCaptureDevice.DeviceType]
        if #available(macOS 14.0, *) {
            types = [.microphone, .external]
        } else {
            types = [.builtInMicrophone, .externalUnknown]
        }
        return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .audio, position: .unspecified)
            .devices.map { Device(id: $0.uniqueID, name: $0.localizedName) }
    }

    /// The system's current input device, for "Default (…)".
    var defaultMicrophoneName: String? { AVCaptureDevice.default(for: .audio)?.localizedName }

    /// The sources the preferences name, as they stand on this Mac now.
    var sources: RecordingSources {
        let microphone: String?
        switch preferences.microphone {
        case "off": microphone = nil
        case "default": microphone = AVCaptureDevice.default(for: .audio)?.uniqueID
        case let id: microphone = microphones.contains { $0.id == id } ? id : AVCaptureDevice.default(for: .audio)?.uniqueID
        }
        var display: CGDirectDisplayID?
        if preferences.screen {
            display = displays.contains { $0.id == preferences.display } ? preferences.display : CGMainDisplayID()
        }
        return RecordingSources(microphone: microphone, systemAudio: preferences.systemAudio, display: display)
    }

    /// The first time, macOS asks; after that only System Settings can.
    func requestScreenAccess() {
        if !CGRequestScreenCaptureAccess() {
            NSWorkspace.shared.open(Self.screenSettingsURL)
        }
        refreshPermissions()
    }

    static let screenSettingsURL = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!
    static let microphoneSettingsURL = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!

    // MARK: - Recording

    func toggle() {
        Task {
            switch state {
            case .idle: await start()
            case .recording: await stop()
            default: break
            }
        }
    }

    func start() async {
        guard state == .idle else { return }
        errorMessage = nil
        warning = nil
        refreshDevices()
        refreshPermissions()
        let sources = sources
        guard sources.hasAudio else {
            errorMessage = preferences.microphone == "off"
                ? "Choose a microphone or turn on the call's audio: a recording needs sound."
                : "No microphone is connected. Connect one, or record the call's audio only."
            return
        }
        if sources.microphone != nil {
            if microphoneAccess == .notDetermined {
                _ = await AVCaptureDevice.requestAccess(for: .audio)
                refreshPermissions()
            }
            guard microphoneAccess == .authorized else {
                errorMessage = "Whiz is not allowed to use the microphone. Allow it in System Settings → Privacy & Security → Microphone, or turn the microphone off here."
                return
            }
        }
        if sources.needsScreenCapture && !screenAllowed {
            requestScreenAccess()
            if !screenAllowed {
                errorMessage = "Whiz needs permission to record the call's audio\(sources.display != nil ? " and the screen" : ""). Allow Whiz in System Settings → Privacy & Security → Screen & System Audio Recording, then quit and reopen it."
                return
            }
        }

        let url = Self.uniqueURL(in: preferences.folder, name: Self.fileName(at: Date()),
                                 extension: sources.display != nil ? "mov" : "m4a")
        let recorder = CallRecorder(url: url, sources: sources)
        recorder.onInterruption = { [weak self] message in
            Task { @MainActor in self?.interrupted(message) }
        }
        state = .starting
        do {
            try await recorder.start()
        } catch {
            state = .idle
            errorMessage = error.localizedDescription
            return
        }
        self.recorder = recorder
        let since = Date()
        state = .recording(since: since)
        elapsed = 0
        // A long call the person only listens to must not end in idle sleep.
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "Recording a call")
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                self?.tick(since: since)
            }
        }
    }

    /// Stop and keep the file; transcribe it when the preference says so
    /// and `transcribe` allows (not on quit).
    func stop(transcribe: Bool = true) async {
        guard isRecording, let recorder else { return }
        state = .stopping
        ticker?.cancel()
        ticker = nil
        do {
            try await recorder.stop()
            lastRecording = recorder.url
        } catch {
            errorMessage = error.localizedDescription
        }
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
        self.recorder = nil
        meter.microphone = 0
        meter.system = 0
        state = .idle
        if transcribe, preferences.transcribeWhenDone, let url = lastRecording,
           url == recorder.url, FileManager.default.fileExists(atPath: url.path) {
            onFinished?(url)
        }
    }

    private func tick(since: Date) {
        guard let recorder, isRecording else { return }
        let reading = recorder.readMeter()
        meter.microphone = reading.microphone
        meter.system = reading.system
        if reading.warning != warning { warning = reading.warning }
        let seconds = Int(Date().timeIntervalSince(since))
        if seconds != elapsed { elapsed = seconds }
    }

    /// The recording stopped on its own; keep what it has.
    private func interrupted(_ message: String) {
        guard isRecording else { return }
        errorMessage = message
        Task { await stop(transcribe: false) }
    }

    // MARK: - Files

    /// "Recording 2026-10-26 at 14.03", in the style of macOS screenshots.
    nonisolated static func fileName(at date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd 'at' HH.mm"
        return "Recording \(formatter.string(from: date))"
    }

    /// `name.ext`, or `name 2.ext` and on when that exists.
    nonisolated static func uniqueURL(in folder: URL, name: String, extension ext: String) -> URL {
        var candidate = folder.appendingPathComponent(name).appendingPathExtension(ext)
        var number = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent("\(name) \(number)").appendingPathExtension(ext)
            number += 1
        }
        return candidate
    }

    /// "12:34" or "1:02:03".
    nonisolated static func clock(_ seconds: Int) -> String {
        TranscriptionViewModel.formatDuration(Double(seconds))
    }
}
