import Foundation
import Testing
@testable import WhizUI

/// The running clock and the "Finished · 3:12" chip both go through
/// `formatDuration`. A transcription runs for minutes, so the formats that
/// matter are seconds-padded minutes and the hour case — a "1:47:30" run
/// printed as "107:30" would be read as noise.
@Suite("Duration formatting")
@MainActor
struct FormatDurationTests {

    @Test("minutes and seconds")
    func minutesAndSeconds() {
        #expect(TranscriptionViewModel.formatDuration(192) == "3:12")
        #expect(TranscriptionViewModel.formatDuration(65) == "1:05")
        #expect(TranscriptionViewModel.formatDuration(0) == "0:00")
    }

    @Test("hours appear only when they exist, with minutes and seconds padded")
    func hours() {
        #expect(TranscriptionViewModel.formatDuration(3600) == "1:00:00")
        #expect(TranscriptionViewModel.formatDuration(6450) == "1:47:30")
    }

    @Test("rounding and the clock-skew clamp")
    func roundingAndClamp() {
        // Sub-second jitter rounds rather than truncating...
        #expect(TranscriptionViewModel.formatDuration(192.6) == "3:13")
        // ...and a clock-skewed negative (sleep/wake, NTP step) prints zero
        // instead of a sign the chip has no way to explain.
        #expect(TranscriptionViewModel.formatDuration(-5) == "0:00")
    }
}