import AppKit
import SwiftUI
import WhizLicense

/// Enter a license key — or, at the end of the trial, the choice between
/// buying and entering one. Opened from Transcribe, the app menu, the
/// sidebar and Settings.
struct ActivationView: View {
    @EnvironmentObject private var license: LicenseModel
    @Environment(\.dismiss) private var dismiss
    @State private var key = ""
    @State private var error: String?
    @State private var activated: License?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            if let activated {
                Label("Licensed to \(activated.name). Thank you for buying Whiz.",
                      systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .font(.headline)
                HStack {
                    Spacer()
                    Button("Done") { dismiss() }
                        .keyboardShortcut(.defaultAction)
                }
            } else {
                form
            }
        }
        .padding(24)
        .frame(width: 520)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 56, height: 56)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.title2.bold())
                Text(subtitle)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var title: String {
        if activated != nil { return "Whiz is activated" }
        return license.openedAtTrialEnd ? "Your free transcriptions are used up" : "Activate Whiz"
    }

    private var subtitle: String {
        if let activated {
            return "No subscription: every Whiz \(activated.major).x update is included."
        }
        if license.openedAtTrialEnd {
            return "Buy Whiz once to keep transcribing: no subscription, and every 1.x update included. "
                + "Your transcripts stay yours either way."
        }
        switch license.status {
        case .licensed(let current): return "Licensed to \(current.name). A new key replaces it."
        case .trial(let remaining):
            return "\(remaining) of \(LicenseModel.freeTranscriptions) free transcriptions left. "
                + "Paste the key from your purchase email."
        case .trialOver: return "Paste the key from your purchase email."
        }
    }

    private var form: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("License key").font(.headline)
            TextEditor(text: $key)
                .font(.system(.callout, design: .monospaced))
                .frame(height: 76)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
                .onChange(of: key) { _ in error = nil }
            if let error {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Buy Whiz…") { NSWorkspace.shared.open(LicenseModel.purchaseURL) }
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Activate") { activate() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.top, 4)
        }
        .onAppear {
            // A key copied from the purchase email is likely on the clipboard.
            if key.isEmpty, let copied = NSPasteboard.general.string(forType: .string),
               copied.trimmingCharacters(in: .whitespacesAndNewlines).uppercased().hasPrefix(LicenseKey.prefix) {
                key = copied.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }

    private func activate() {
        do {
            activated = try license.activate(key)
        } catch {
            self.error = error.localizedDescription
        }
    }
}

/// The sidebar's foot during the trial: how much is left, and the way out.
/// Licensed copies show nothing.
struct TrialFooter: View {
    @EnvironmentObject private var license: LicenseModel

    var body: some View {
        switch license.status {
        case .licensed:
            EmptyView()
        case .trial(let remaining):
            card(title: "Free trial",
                 detail: "\(remaining) of \(LicenseModel.freeTranscriptions) transcriptions left",
                 symbol: "gift")
        case .trialOver:
            card(title: "Trial over",
                 detail: "Buy Whiz once to keep transcribing.",
                 symbol: "lock")
        }
    }

    private func card(title: String, detail: String, symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbol).font(.headline)
            Text(detail).font(.callout).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button("Buy Whiz") { NSWorkspace.shared.open(LicenseModel.purchaseURL) }
                    .buttonStyle(.borderedProminent)
                Button("Enter Key…") { license.presentActivation() }
            }
            .controlSize(.small)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
        .padding(10)
    }
}

/// Settings → License.
struct LicenseSettings: View {
    @EnvironmentObject private var license: LicenseModel
    @State private var confirmingRemoval = false

    var body: some View {
        Form {
            switch license.status {
            case .licensed(let current):
                Section("License") {
                    LabeledContent("Licensed to", value: current.name)
                    LabeledContent("Order", value: current.id)
                    LabeledContent("Covers", value: "Whiz \(current.major).x")
                    LabeledContent("Issued", value: current.issued)
                }
                Section {
                    Button("Remove License from This Mac…") { confirmingRemoval = true }
                    Text("Use this before moving Whiz to another Mac. Your key still works there.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            case .trial(let remaining):
                trial("\(remaining) of \(LicenseModel.freeTranscriptions) free transcriptions left.")
            case .trialOver:
                trial("The free transcriptions are used up. Past transcripts stay readable.")
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("Remove the license from this Mac?", isPresented: $confirmingRemoval) {
            Button("Remove License", role: .destructive) { license.deactivate() }
        } message: {
            Text("Whiz goes back to the free trial here until a key is entered again.")
        }
    }

    private func trial(_ detail: String) -> some View {
        Section("Free trial") {
            Text(detail)
            Text("Whiz is a one-time purchase: no subscription, every 1.x update included.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Buy Whiz…") { NSWorkspace.shared.open(LicenseModel.purchaseURL) }
                Button("Enter License Key…") { license.presentActivation() }
            }
        }
    }
}
