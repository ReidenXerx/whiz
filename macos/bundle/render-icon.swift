// render-icon.swift — draw the Whiz app icon with CoreGraphics.
//
// Usage: swift render-icon.swift <out.png> <pixels>
//
// The icon is geometry, not an asset: a squircle on Apple's 1024-pt icon
// grid (824-pt body, 100-pt margin for the shadow), an indigo gradient, and
// a speech bubble holding a waveform whose bars use the app's speaker
// colors (merge.py's palette) — speech becoming a transcript, and telling
// who said what. Everything is in 1024-unit coordinates and scaled, so every
// size is drawn, not resampled.

import AppKit
import CoreGraphics
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3, let pixels = Int(arguments[2]), pixels > 0 else {
    FileHandle.standardError.write(Data("usage: render-icon.swift <out.png> <pixels>\n".utf8))
    exit(2)
}
let outURL = URL(fileURLWithPath: arguments[1])

func rgb(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat((hex >> 16) & 0xff) / 255, green: CGFloat((hex >> 8) & 0xff) / 255,
            blue: CGFloat(hex & 0xff) / 255, alpha: alpha)
}

let space = CGColorSpace(name: CGColorSpace.sRGB)!
guard let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
                          space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    exit(1)
}
let scale = CGFloat(pixels) / 1024
ctx.scaleBy(x: scale, y: scale)
// Draw top-down like the design grid.
ctx.translateBy(x: 0, y: 1024)
ctx.scaleBy(x: 1, y: -1)
ctx.setShouldAntialias(true)
ctx.interpolationQuality = .high

/// Apple's continuous-corner squircle, approximated with a superellipse-ish
/// rounded rect: NSBezierPath has no continuous corners, so the corner
/// curves use control points pulled in past a circular arc (n ≈ 5).
func squircle(_ rect: CGRect, radius: CGFloat) -> CGPath {
    let path = CGMutablePath()
    let k: CGFloat = 1.28 // how far the corner curve reaches along each edge
    let r = radius
    let (minX, minY, maxX, maxY) = (rect.minX, rect.minY, rect.maxX, rect.maxY)
    let c: CGFloat = 0.64 // control-point tension
    path.move(to: CGPoint(x: minX + r * k, y: minY))
    path.addLine(to: CGPoint(x: maxX - r * k, y: minY))
    path.addCurve(to: CGPoint(x: maxX, y: minY + r * k),
                  control1: CGPoint(x: maxX - r * k * (1 - c), y: minY),
                  control2: CGPoint(x: maxX, y: minY + r * k * (1 - c)))
    path.addLine(to: CGPoint(x: maxX, y: maxY - r * k))
    path.addCurve(to: CGPoint(x: maxX - r * k, y: maxY),
                  control1: CGPoint(x: maxX, y: maxY - r * k * (1 - c)),
                  control2: CGPoint(x: maxX - r * k * (1 - c), y: maxY))
    path.addLine(to: CGPoint(x: minX + r * k, y: maxY))
    path.addCurve(to: CGPoint(x: minX, y: maxY - r * k),
                  control1: CGPoint(x: minX + r * k * (1 - c), y: maxY),
                  control2: CGPoint(x: minX, y: maxY - r * k * (1 - c)))
    path.addLine(to: CGPoint(x: minX, y: minY + r * k))
    path.addCurve(to: CGPoint(x: minX + r * k, y: minY),
                  control1: CGPoint(x: minX, y: minY + r * k * (1 - c)),
                  control2: CGPoint(x: minX + r * k * (1 - c), y: minY))
    path.closeSubpath()
    return path
}

let body = CGRect(x: 100, y: 100, width: 824, height: 824)
let bodyPath = squircle(body, radius: 160)

// 1. Drop shadow under the tile.
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: 12), blur: 28, color: rgb(0x000000, 0.30))
ctx.addPath(bodyPath)
ctx.setFillColor(rgb(0x3020B0))
ctx.fillPath()
ctx.restoreGState()

// 2. The tile: indigo gradient, lit from the top-left.
ctx.saveGState()
ctx.addPath(bodyPath)
ctx.clip()
let tile = CGGradient(colorsSpace: space,
                      colors: [rgb(0x7466FF), rgb(0x4A36E8), rgb(0x2A1A93)] as CFArray,
                      locations: [0, 0.5, 1])!
ctx.drawLinearGradient(tile, start: CGPoint(x: 260, y: 100), end: CGPoint(x: 620, y: 924),
                       options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
let glow = CGGradient(colorsSpace: space,
                      colors: [rgb(0xFFFFFF, 0.30), rgb(0xFFFFFF, 0)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 330, y: 150), startRadius: 0,
                       endCenter: CGPoint(x: 330, y: 150), endRadius: 620, options: [])
// A hairline of light along the top edge, as on Apple's own tiles.
ctx.addPath(bodyPath)
ctx.setStrokeColor(rgb(0xFFFFFF, 0.18))
ctx.setLineWidth(4)
ctx.strokePath()
ctx.restoreGState()

// 3. The speech bubble.
let bubbleRect = CGRect(x: 214, y: 236, width: 596, height: 452)
let bubble = CGMutablePath()
bubble.addPath(CGPath(roundedRect: bubbleRect, cornerWidth: 132, cornerHeight: 132, transform: nil))
// The tail, low on the left, curling out of the bubble's bottom edge.
let tail = CGMutablePath()
tail.move(to: CGPoint(x: 300, y: 640))
tail.addLine(to: CGPoint(x: 430, y: 670))
tail.addCurve(to: CGPoint(x: 278, y: 806), control1: CGPoint(x: 380, y: 740), control2: CGPoint(x: 320, y: 790))
tail.addCurve(to: CGPoint(x: 262, y: 788), control1: CGPoint(x: 262, y: 812), control2: CGPoint(x: 256, y: 800))
tail.addCurve(to: CGPoint(x: 300, y: 640), control1: CGPoint(x: 290, y: 740), control2: CGPoint(x: 300, y: 690))
tail.closeSubpath()

ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: 16), blur: 30, color: rgb(0x0B0640, 0.40))
ctx.addPath(bubble)
ctx.addPath(tail)
ctx.setFillColor(rgb(0xFFFFFF))
ctx.fillPath()
ctx.restoreGState()

ctx.saveGState()
ctx.addPath(bubble)
ctx.addPath(tail)
ctx.clip()
let paper = CGGradient(colorsSpace: space,
                       colors: [rgb(0xFFFFFF), rgb(0xEAE7FF)] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(paper, start: CGPoint(x: 512, y: 236), end: CGPoint(x: 512, y: 810),
                       options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
ctx.restoreGState()

// 4. The waveform: two speakers' worth of bars, in the transcript's colors.
// Five bars, not more: at 32 and 16 px a sixth only blurs the others.
let bars: [(x: CGFloat, half: CGFloat, color: UInt32)] = [
    (352, 54, 0xE74C3C),
    (432, 118, 0xE74C3C),
    (512, 162, 0xE74C3C),
    (592, 112, 0x3498DB),
    (672, 62, 0x3498DB),
]
let centerY: CGFloat = 462
ctx.setLineCap(.round)
ctx.setLineWidth(58)
for bar in bars {
    ctx.setStrokeColor(rgb(bar.color))
    ctx.move(to: CGPoint(x: bar.x, y: centerY - bar.half))
    ctx.addLine(to: CGPoint(x: bar.x, y: centerY + bar.half))
    ctx.strokePath()
}

guard let image = ctx.makeImage() else { exit(1) }
let rep = NSBitmapImageRep(cgImage: image)
guard let png = rep.representation(using: .png, properties: [:]) else { exit(1) }
try png.write(to: outURL)
