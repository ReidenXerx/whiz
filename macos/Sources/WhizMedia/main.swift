import Foundation

// whiz-media — the two things whiz asks of ffmpeg, done with macOS's own
// decoders, so Whiz.app needs no ffmpeg (and ships no GPL code).
//
//   whiz-media audio <input> <output.wav> [--comment TEXT]
//   whiz-media frame <input> <seconds> <output.jpg> [--width N] [--quality Q]
//
// whiz still decides everything — what to extract, where, and when a
// result counts. This is a codec, not a second pipeline: whiz/audio.py and
// whiz/screenshots.py build these argv lists the way they build ffmpeg's,
// and prefer ffmpeg whenever one is installed.

exit(await MediaCommand.run(Array(CommandLine.arguments.dropFirst())))
