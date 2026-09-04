import Foundation

/// The self-contained HTML transcript — `merge.py:format_speakers_html`.
///
/// One portable file: color-coded per-speaker cues, a sticky header with a
/// speaker legend and a live search box, and — when the frames manifest's
/// JPEGs exist — every screenshot inlined as `data:image/jpeg;base64` so the
/// artifact carries its frames with no external files. Clicking a thumbnail
/// opens a lightbox; the CSS/JS are transcribed verbatim from the Python
/// implementation so both sides render identically.
enum SpeakersHTML {

    /// merge.py:_SPEAKER_COLORS — deterministic per-speaker colors.
    static let palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#d35400",
    ]

    /// A stable color per label: the sum of the label's code points modulo the
    /// palette — same hash, same color, across runs and implementations.
    static func speakerColor(_ label: String) -> String {
        let hash = label.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        return palette[hash % palette.count]
    }

    /// merge.py:_html_escape.
    static func htmlEscape(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    /// Emit the page. `framesDir` is optional: a missing directory or missing
    /// `segNNNN.jpg` per cue simply emits no thumbnail, and the lightbox +
    /// script only appear when at least one frame was inlined.
    static func format(
        _ merged: [LabeledSegment],
        framesDir: URL? = nil,
        title: String = "whiz transcript"
    ) -> String {
        let legend = LabeledTranscript.speakersInOrder(merged).map {
            (label: $0, color: speakerColor($0))
        }

        var parts: [String] = []
        parts.append("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">")
        parts.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        parts.append("<title>\(htmlEscape(title))</title>")
        parts.append("<style>\(css)</style>")
        parts.append("</head>\n<body>")

        parts.append("<header class=\"bar\">")
        parts.append("<h1>\(htmlEscape(title))</h1>")
        if !legend.isEmpty {
            parts.append("<div class=\"legend\">")
            for entry in legend {
                parts.append(
                    "<span class=\"chip\" style=\"background:\(entry.color)\">"
                        + "\(htmlEscape(entry.label))</span>")
            }
            parts.append("</div>")
        }
        parts.append("<div class=\"spacer\"></div>")
        parts.append(
            "<input id=\"search\" class=\"search\" type=\"search\" "
                + "placeholder=\"Search transcript&hellip;\" aria-label=\"Search transcript\">")
        parts.append("</header>")

        parts.append("<main>")
        var cueCount = 0
        var hasFrame = false
        for (index, entry) in merged.enumerated() {
            let text = entry.segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            cueCount += 1
            let color = speakerColor(entry.speaker)
            let timestamp = TranscriptFormatter.clock(entry.segment.start)
            parts.append("<div class=\"cue\" style=\"--c:\(color)\">")

            // Inline the frame thumbnail if it exists (clickable → lightbox).
            // `index` matches the frames manifest's 1-based segment numbering,
            // including segments whose text turned out empty.
            if let framesDir {
                let framePath = framesDir.appendingPathComponent(
                    String(format: "seg%04d.jpg", index + 1))
                if let data = try? Data(contentsOf: framePath) {
                    hasFrame = true
                    parts.append(
                        "<div class=\"frame\"><img src=\"data:image/jpeg;base64,"
                            + "\(data.base64EncodedString())\" alt=\"frame \(index + 1)\"></div>")
                }
            }

            parts.append("<div class=\"body\">")
            parts.append("<div class=\"meta\">")
            parts.append("<a class=\"ts\" href=\"#cue-\(index + 1)\" id=\"cue-\(index + 1)\">\(timestamp)</a>")
            parts.append("<span class=\"speaker\" style=\"color:\(color)\">\(htmlEscape(entry.speaker))</span>")
            parts.append("</div>")
            parts.append("<div class=\"text\">\(htmlEscape(text))</div>")
            parts.append("</div></div>")
        }
        parts.append("</main>")

        if cueCount > 0 {
            parts.append("<div class=\"foot\">\(cueCount) cue(s)</div>")
        }

        // Lightbox overlay only when at least one frame was inlined — a
        // frames-less transcript emits no <img> tags at all (mirrors Python).
        if hasFrame {
            parts.append("<div class=\"lightbox\" id=\"lightbox\" aria-hidden=\"true\">")
            parts.append("<button class=\"close\" aria-label=\"Close\">&times;</button>")
            parts.append("<img alt=\"fullscreen frame\">")
            parts.append("</div>")
            parts.append("<script>\(js)</script>")
        }

        parts.append("</body>\n</html>")
        return parts.joined(separator: "\n")
    }

    // MARK: - Page assets (transcribed verbatim from merge.py)

    private static let css = """
    :root {
      --bg: #fafafa;
      --card: #ffffff;
      --text: #1f2328;
      --muted: #6e7781;
      --border: #e4e7eb;
      --accent: #2f81f7;
      --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    header.bar {
      position: sticky; top: 0; z-index: 20;
      background: rgba(255,255,255,.92);
      backdrop-filter: saturate(180%) blur(12px);
      -webkit-backdrop-filter: saturate(180%) blur(12px);
      border-bottom: 1px solid var(--border);
      padding: .65em 1em;
      display: flex; align-items: center; gap: 1em; flex-wrap: wrap;
    }
    header.bar h1 { font-size: 1.05em; margin: 0; font-weight: 650; letter-spacing: -.01em; }
    header.bar .spacer { flex: 1; }
    header.bar input.search {
      font: inherit; font-size: .9em; padding: .35em .6em .35em 1.8em;
      border: 1px solid var(--border); border-radius: 8px;
      width: 16em; max-width: 40vw; background: var(--card) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236e7781' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>") .55em .5em no-repeat; outline: none;
    }
    header.bar input.search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(47,129,247,.18); }
    .legend { display: flex; gap: .4em; flex-wrap: wrap; align-items: center; }
    .legend .chip { font-size: .72em; padding: .15em .55em; border-radius: 999px; color: #fff; font-weight: 600; white-space: nowrap; }
    main { max-width: 920px; margin: 0 auto; padding: 1em; }
    .cue {
      display: flex; gap: .9em; align-items: flex-start;
      margin: .35em 0; padding: .7em .8em;
      background: var(--card);
      border: 1px solid var(--border); border-left: 3px solid var(--c, var(--border));
      border-radius: 10px; box-shadow: var(--shadow);
      transition: transform .06s ease, box-shadow .12s ease;
    }
    .cue:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,.08); }
    .cue .frame {
      flex-shrink: 0; cursor: zoom-in; position: relative;
      border-radius: 8px; overflow: hidden; line-height: 0;
      border: 1px solid var(--border);
    }
    .cue .frame img { width: 180px; height: 116px; object-fit: cover; display: block; }
    .cue .frame::after {
      content: ""; position: absolute; inset: 0; background: rgba(0,0,0,0); transition: background .12s;
    }
    .cue .frame:hover::after { background: rgba(0,0,0,.12); }
    .cue .body { flex: 1; min-width: 0; }
    .cue .meta { display: flex; align-items: baseline; gap: .55em; margin-bottom: .15em; }
    .cue .ts { color: var(--muted); font-size: .8em; font-variant-numeric: tabular-nums; text-decoration: none; border-radius: 4px; padding: 0 .2em; }
    .cue .ts:hover { background: var(--border); color: var(--text); }
    .cue .speaker { font-weight: 650; font-size: .92em; }
    .cue .text { font-size: .95em; white-space: pre-wrap; word-wrap: break-word; }
    .cue.hidden { display: none; }
    footer.foot { text-align: center; color: var(--muted); font-size: .8em; padding: 1.5em; }
    /* Lightbox */
    .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.86); z-index: 100;
      display: none; align-items: center; justify-content: center; padding: 2em; }
    .lightbox.open { display: flex; }
    .lightbox img { max-width: 96vw; max-height: 92vh; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,.5); }
    .lightbox .close { position: absolute; top: 1em; right: 1.4em; background: rgba(255,255,255,.12); color: #fff;
      border: none; font-size: 1.6em; line-height: 1; width: 2.2em; height: 2.2em; border-radius: 50%; cursor: pointer; }
    .lightbox .close:hover { background: rgba(255,255,255,.22); }
    @media (max-width: 640px) {
      .cue .frame img { width: 120px; height: 78px; }
      header.bar input.search { width: 10em; }
    }
    """

    private static let js = """
    (function () {
      var box = document.getElementById('lightbox');
      var boxImg = box.querySelector('img');
      function open(src) { boxImg.src = src; box.classList.add('open'); }
      function close() { box.classList.remove('open'); boxImg.src = ''; }
      document.querySelectorAll('.cue .frame').forEach(function (el) {
        el.addEventListener('click', function () {
          var img = el.querySelector('img'); if (img) open(img.src);
        });
      });
      box.addEventListener('click', function (e) { if (e.target === box || e.target.classList.contains('close')) close(); });
      document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
      var search = document.getElementById('search');
      if (search) {
        search.addEventListener('input', function () {
          var q = search.value.trim().toLowerCase();
          document.querySelectorAll('.cue').forEach(function (cue) {
            var hay = (cue.textContent || '').toLowerCase();
            cue.classList.toggle('hidden', q && hay.indexOf(q) === -1);
          });
        });
      }
    })();
    """
}