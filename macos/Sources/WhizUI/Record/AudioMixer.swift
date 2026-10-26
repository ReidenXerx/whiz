import Foundation

/// Mixes the microphone and the call's audio into one mono track, placed
/// by timestamp.
///
/// One track, not two: ffmpeg (which whiz uses whenever one is installed)
/// decodes a single audio stream, so a file keeping the two sides apart
/// would lose half the conversation there — and many players do the same.
///
/// The sources run on different clocks and arrive in buffers of different
/// sizes, sometimes late, sometimes not at all (ScreenCaptureKit sends
/// nothing while the Mac is silent). Each buffer is written at the frame its
/// timestamp names; a buffer that lands within `jitterTolerance` of where
/// its source's last one ended continues it seamlessly instead, so timestamp
/// jitter never opens a click-sized gap. Frames are released once every
/// source has reached them, or once they are older than the caller's
/// deadline — a source that has gone quiet counts as silence then.
struct AudioMixer {
    enum Source: Hashable, Sendable { case microphone, system }

    static let sampleRate = 48_000
    /// 50 ms: closer than this to the expected position is jitter, further
    /// is a real gap (or overlap) and the buffer goes where it says.
    static let jitterTolerance = 2_400

    /// Frames released so far; the next `drain` starts here.
    private(set) var released = 0
    /// Mixed frames from `released` onward, not yet released.
    private var pending: [Float] = []
    /// Where each source's last buffer ended.
    private var cursors: [Source: Int] = [:]

    /// Add a buffer of mono samples that starts at `frame` on the timeline.
    /// Whatever falls before `released` is too late and is dropped.
    mutating func add(_ samples: [Float], from source: Source, at frame: Int) {
        var start = frame
        var overlap = 0
        if let cursor = cursors[source] {
            if abs(frame - cursor) <= Self.jitterTolerance {
                start = cursor
            } else if frame < cursor {
                // The source's clock fell behind the Mac's: what it repeats
                // is already written, and mixed twice it would echo.
                overlap = cursor - frame
                start = cursor
            }
        }
        guard overlap < samples.count else { return }
        cursors[source] = start + samples.count - overlap

        // samples[i] belongs at frame start + i - overlap; anything before
        // `released` is already written.
        let first = max(released - start, 0) + overlap
        guard first < samples.count else { return }
        let base = start - overlap - released
        let end = base + samples.count
        if pending.count < end {
            pending.append(contentsOf: repeatElement(0, count: end - pending.count))
        }
        for index in first..<samples.count {
            pending[base + index] += samples[index]
        }
    }

    /// Release the frames ready to write: up to where every source that has
    /// spoken has reached, and at least up to `deadline`, silence filling
    /// what no source covered. Clipped to ±1, as the encoder would.
    mutating func drain(deadline: Int) -> (start: Int, samples: [Float])? {
        let reached = cursors.values.min() ?? released
        let end = max(reached, deadline)
        guard end > released else { return nil }
        let count = end - released
        if pending.count < count {
            pending.append(contentsOf: repeatElement(0, count: count - pending.count))
        }
        let samples = pending[..<count].map { min(max($0, -1), 1) }
        pending.removeFirst(count)
        let start = released
        released = end
        return (start, samples)
    }

    /// Everything any source delivered, for the end of the recording.
    mutating func drainAll() -> (start: Int, samples: [Float])? {
        drain(deadline: max(cursors.values.max() ?? released, released + pending.count))
    }
}
