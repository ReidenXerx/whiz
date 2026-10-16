import SwiftUI

/// One block of a markdown document. The analysis is an LLM's markdown:
/// headings, bullets, bold, the odd table and code fence. SwiftUI's
/// `AttributedString(markdown:)` handles the inline half; this is the block
/// half, kept deliberately small — what the reports actually contain.
enum MarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    /// `ordinal` is nil for a bullet; `depth` counts two-space indents.
    case listItem(ordinal: Int?, depth: Int, text: String)
    case quote(String)
    case code(String)
    case table([[String]])
    case rule

    static func parse(_ markdown: String) -> [MarkdownBlock] {
        let lines = markdown.replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n")
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var index = 0

        func flushParagraph() {
            let text = paragraph.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { blocks.append(.paragraph(text)) }
            paragraph.removeAll()
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                flushParagraph()
                var body: [String] = []
                index += 1
                while index < lines.count, !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    body.append(lines[index])
                    index += 1
                }
                blocks.append(.code(body.joined(separator: "\n")))
                index += 1
                continue
            }
            if trimmed.isEmpty {
                flushParagraph()
                index += 1
                continue
            }
            if let heading = headingLevel(trimmed) {
                flushParagraph()
                let text = trimmed.drop { $0 == "#" }.trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(level: heading, text: text))
                index += 1
                continue
            }
            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                flushParagraph()
                blocks.append(.rule)
                index += 1
                continue
            }
            if trimmed.hasPrefix("|") {
                flushParagraph()
                var rows: [[String]] = []
                while index < lines.count, lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    let row = lines[index].trimmingCharacters(in: .whitespaces)
                    let cells = row.trimmingCharacters(in: CharacterSet(charactersIn: "|"))
                        .components(separatedBy: "|")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                    // The |---|:---:| separator row carries alignment only.
                    let isSeparator = cells.allSatisfy { cell in
                        !cell.isEmpty && cell.allSatisfy { "-:".contains($0) }
                    }
                    if !isSeparator { rows.append(cells) }
                    index += 1
                }
                if !rows.isEmpty { blocks.append(.table(rows)) }
                continue
            }
            if trimmed.hasPrefix(">") {
                flushParagraph()
                blocks.append(.quote(trimmed.drop { $0 == ">" }.trimmingCharacters(in: .whitespaces)))
                index += 1
                continue
            }
            if let item = listItem(line) {
                flushParagraph()
                blocks.append(item)
                index += 1
                continue
            }
            paragraph.append(trimmed)
            index += 1
        }
        flushParagraph()
        return blocks
    }

    private static func headingLevel(_ line: String) -> Int? {
        let hashes = line.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes), line.dropFirst(hashes).first == " " else { return nil }
        return hashes
    }

    private static func listItem(_ line: String) -> MarkdownBlock? {
        let indent = line.prefix { $0 == " " }.count
        let body = line.dropFirst(indent)
        let depth = min(indent / 2, 4)
        for bullet in ["- ", "* ", "+ ", "• "] where body.hasPrefix(bullet) {
            return .listItem(ordinal: nil, depth: depth, text: String(body.dropFirst(bullet.count)))
        }
        let digits = body.prefix { $0.isNumber }
        if !digits.isEmpty, digits.count <= 3,
           body.dropFirst(digits.count).hasPrefix(". ") || body.dropFirst(digits.count).hasPrefix(") "),
           let ordinal = Int(digits) {
            return .listItem(ordinal: ordinal, depth: depth,
                             text: String(body.dropFirst(digits.count + 2)))
        }
        return nil
    }
}

/// Renders `MarkdownBlock`s in a reading column.
struct MarkdownView: View {
    let blocks: [MarkdownBlock]

    init(_ markdown: String) {
        blocks = MarkdownBlock.parse(markdown)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(Self.inline(text))
                .font(level <= 1 ? .title2.weight(.semibold)
                      : level == 2 ? .title3.weight(.semibold) : .headline)
                .padding(.top, level <= 2 ? 8 : 4)
        case .paragraph(let text):
            Text(Self.inline(text))
                .font(.body)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
        case .listItem(let ordinal, let depth, let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(ordinal.map { "\($0)." } ?? "•")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(minWidth: 14, alignment: .trailing)
                Text(Self.inline(text))
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(depth) * 18)
        case .quote(let text):
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(.tertiary)
                    .frame(width: 3)
                Text(Self.inline(text))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .padding(12)
            }
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.primary.opacity(0.05)))
        case .table(let rows):
            tableView(rows)
        case .rule:
            Divider().padding(.vertical, 4)
        }
    }

    private func tableView(_ rows: [[String]]) -> some View {
        let columns = rows.map(\.count).max() ?? 0
        return ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                    GridRow {
                        ForEach(0..<columns, id: \.self) { column in
                            Text(Self.inline(column < row.count ? row[column] : ""))
                                .font(rowIndex == 0 ? .callout.weight(.semibold) : .callout)
                                .fixedSize(horizontal: false, vertical: true)
                                .frame(maxWidth: 320, alignment: .leading)
                        }
                    }
                    if rowIndex == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                }
            }
            .padding(12)
        }
        .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
            .strokeBorder(Color.primary.opacity(0.1)))
    }

    /// Inline markdown (bold, italic, code, links); plain text if it does not
    /// parse. Whitespace is preserved so an LLM's two-space line breaks hold.
    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}
