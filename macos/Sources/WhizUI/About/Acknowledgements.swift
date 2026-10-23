import AppKit
import SwiftUI

/// The open-source software Whiz ships, and the models it downloads, with
/// their licences: Whiz > Acknowledgements.
///
/// The list is not written here. build-engine.sh generates
/// Acknowledgements.json from what is actually in the bundle (see
/// bundle/acknowledgements/), so a dependency added or bumped can never be
/// missing from it, and the versions shown are the versions shipped.
struct Acknowledgements: Decodable, Equatable {
    struct Group: Decodable, Equatable, Identifiable {
        let title: String
        let components: [Component]
        var id: String { title }
    }

    struct Component: Decodable, Equatable, Hashable, Identifiable {
        let name: String
        let version: String?
        let license: String
        let url: URL?
        let summary: String?
        /// The licence itself; nil for a model, which Whiz downloads from
        /// its publisher rather than redistributes.
        let text: String?
        var id: String { name }
    }

    let groups: [Group]

    static let resourceName = "Acknowledgements"

    /// nil when the app was not assembled by build-app.sh (`swift run`).
    static func load(from bundle: Bundle = .main) -> Acknowledgements? {
        guard let url = bundle.url(forResource: resourceName, withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? decode(data)
    }

    static func decode(_ data: Data) throws -> Acknowledgements {
        try JSONDecoder().decode(Acknowledgements.self, from: data)
    }

    static let bundled = load()

    var first: Component? { groups.first?.components.first }
}

struct AcknowledgementsView: View {
    let acknowledgements: Acknowledgements?
    @State private var selection: Acknowledgements.Component?

    var body: some View {
        if let acknowledgements {
            NavigationSplitView {
                List(selection: $selection) {
                    ForEach(acknowledgements.groups) { group in
                        Section(group.title) {
                            ForEach(group.components) { component in
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(component.name)
                                    Text([component.version, component.license].compactMap { $0 }.joined(separator: " · "))
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                .tag(component)
                            }
                        }
                    }
                }
                .navigationSplitViewColumnWidth(min: 200, ideal: 230)
            } detail: {
                if let selection {
                    ComponentDetail(component: selection)
                } else {
                    Text("Select a component").foregroundStyle(.secondary)
                }
            }
            .onAppear { if selection == nil { selection = acknowledgements.first } }
        } else {
            EmptyState(symbol: "doc.text", title: "No acknowledgements in this build",
                       message: "build-app.sh adds them when it assembles Whiz.app.") { EmptyView() }
        }
    }
}

private struct ComponentDetail: View {
    let component: Acknowledgements.Component

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline) {
                    Text(component.name).font(.title2.weight(.semibold))
                    if let version = component.version {
                        Text(version).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let url = component.url {
                        Link("Website", destination: url)
                    }
                }
                Text("License: \(component.license)").font(.callout)
                if let summary = component.summary, !summary.isEmpty {
                    Text(summary).font(.callout).foregroundStyle(.secondary)
                }
            }
            .padding(16)
            Divider()
            if let text = component.text {
                LicenseText(text: text)
            } else {
                Text("Whiz downloads this model from its publisher when you choose it; it is not part of the app.")
                    .foregroundStyle(.secondary)
                    .padding(16)
                Spacer()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// NSTextView, not Text: ONNX Runtime's notices alone are 300 KB, which a
/// SwiftUI Text lays out all at once.
private struct LicenseText: NSViewRepresentable {
    let text: String

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let view = scroll.documentView as! NSTextView
        view.isEditable = false
        view.isSelectable = true
        view.drawsBackground = false
        view.textContainerInset = NSSize(width: 12, height: 12)
        view.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        view.textColor = .labelColor
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        let view = scroll.documentView as! NSTextView
        guard view.string != text else { return }
        view.string = text
        view.scrollToBeginningOfDocument(nil)
    }
}
