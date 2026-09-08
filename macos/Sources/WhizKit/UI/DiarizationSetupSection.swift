import SwiftUI

/// Speaker-diarization model status and one-time setup.
///
/// The app's answer to `cli.py:_auto_setup_consent`. The CLI can ask on a TTY
/// and fall back to a printed hint; a GUI user has neither, so without this the
/// only symptom of missing models is transcripts that quietly say "Speaker"
/// instead of naming anyone — with no indication that anything is installable.
///
/// Consent is explicit and remembered. Declining writes
/// `auto_diarization_setup = false`, which the Python CLI honours too, so
/// "don't ask me again" means it across both tools.
struct DiarizationSetupSection: View {
    @ObservedObject var controller: SessionController
    @StateObject private var setup = DiarizationSetup()

    /// Bumped after install so the "is it there?" check re-runs — the
    /// filesystem does not notify us.
    @State private var refresh = 0

    var body: some View {
        Section("Speaker diarization") {
            if installed {
                LabeledContent("Models") {
                    Text("Installed").foregroundStyle(.secondary)
                }
                Text("Transcripts of videos and multi-speaker audio are labelled "
                     + "by speaker.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                switch setup.state {
                case .downloading(let phase, let progress):
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: progress)
                        HStack {
                            Text(phase).font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button("Cancel") { setup.cancel() }
                        }
                    }
                case .failed(let message):
                    VStack(alignment: .leading, spacing: 6) {
                        Text(message).font(.caption).foregroundStyle(.red)
                        Button("Try Again") { setup.install() }
                    }
                default:
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Not installed. Without these models, transcripts are "
                             + "not labelled by speaker — everyone appears as "
                             + "\"Speaker\".")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack {
                            Text("One-time download, about 44 MB.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            if controller.config.autoDiarizationSetup == false {
                                Button("Ask Again") {
                                    controller.updateConfig { $0.autoDiarizationSetup = nil }
                                }
                            } else {
                                Button("Don't Ask") {
                                    controller.updateConfig { $0.autoDiarizationSetup = false }
                                }
                            }
                            Button("Download") { setup.install() }
                        }
                    }
                }
            }
        }
        .onChange(of: setup.state) { state in
            if state == .finished { refresh += 1 }
        }
    }

    private var installed: Bool {
        _ = refresh
        return DiarizationSetup.isInstalled
    }
}
