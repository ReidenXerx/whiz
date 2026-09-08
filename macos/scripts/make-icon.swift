// Renders Whiz.icns from the same waveform-W the menu bar and pill use.
//
//   swift macos/scripts/make-icon.swift
//
// Run it after changing the mark; the result is committed so a normal build
// needs no icon tooling.
//
// The geometry is duplicated from `WhizKit/UI/WhizLogo.swift` rather than
// imported: this runs as a standalone script, before and outside the package
// build, so it cannot depend on WhizKit. Keep the two in step — the shape is
// stable, but if `WhizLogo.points` ever changes, change it here too.
import AppKit

// Unit-square W: top-left, valley, middle peak, valley, top-right. AppKit's
// origin is bottom-left, so these are WhizLogo's y values flipped.
let points: [(CGFloat, CGFloat)] = [
    (0.10, 0.75), (0.28, 0.25), (0.50, 0.55), (0.72, 0.25), (0.90, 0.75),
]
let strokeRatio: CGFloat = 0.14

// Brand cyan — the "listening" tint from DictationState.
let mark = NSColor(srgbRed: 0.20, green: 0.80, blue: 0.95, alpha: 1.0)
let background = NSColor(srgbRed: 0.11, green: 0.12, blue: 0.14, alpha: 1.0)

/// One icon at `size` points, drawn into a square canvas.
///
/// The art is inset to ~82% of the canvas: macOS reserves the outer margin so
/// icons of different shapes optically match on the shelf. Drawing edge to edge
/// makes an icon look oversized next to every other app.
func render(size: CGFloat) -> Data? {
    let image = NSImage(size: NSSize(width: size, height: size), flipped: false) { _ in
        let inset = size * 0.09
        let box = NSRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)

        // Rounded-rect plate, matching the macOS squircle proportion closely
        // enough at every size we emit.
        let plate = NSBezierPath(roundedRect: box,
                                 xRadius: box.width * 0.225,
                                 yRadius: box.width * 0.225)
        background.setFill()
        plate.fill()

        // The W, scaled to the plate rather than the canvas.
        let side = box.width * 0.62
        let originX = box.midX - side / 2
        let originY = box.midY - side / 2
        let path = NSBezierPath()
        for (index, point) in points.enumerated() {
            let p = NSPoint(x: originX + point.0 * side, y: originY + point.1 * side)
            index == 0 ? path.move(to: p) : path.line(to: p)
        }
        path.lineWidth = side * strokeRatio
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        mark.setStroke()
        path.stroke()
        return true
    }
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff) else { return nil }
    rep.size = NSSize(width: size, height: size)
    return rep.representation(using: .png, properties: [:])
}

let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
let iconset = root.appendingPathComponent("build/Whiz.iconset")
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

// The exact set `iconutil` expects; a missing size makes it fail outright.
let variants: [(name: String, size: CGFloat)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]
for variant in variants {
    guard let png = render(size: variant.size) else {
        FileHandle.standardError.write(Data("failed to render \(variant.name)\n".utf8))
        exit(1)
    }
    try png.write(to: iconset.appendingPathComponent("\(variant.name).png"))
}

let output = root.appendingPathComponent("Resources/Whiz.icns")
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", output.path]
try iconutil.run()
iconutil.waitUntilExit()
guard iconutil.terminationStatus == 0 else {
    FileHandle.standardError.write(Data("iconutil failed\n".utf8))
    exit(1)
}
try? FileManager.default.removeItem(at: iconset)
print("wrote \(output.path)")
