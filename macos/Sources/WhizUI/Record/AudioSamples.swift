import AVFoundation
import CoreMedia

/// A captured audio buffer as the mono 48 kHz floats `AudioMixer` takes.
///
/// Both sources are asked for exactly that (the microphone through its
/// output's audio settings, ScreenCaptureKit through its configuration);
/// this still averages any channels it is handed, and refuses another
/// sample rate rather than playing it at the wrong speed.
enum AudioSamples {
    enum Failure: Error, Equatable {
        case unsupported(String)
        case sampleRate(Double)
    }

    static func mono(from sample: CMSampleBuffer) throws -> [Float] {
        guard let description = CMSampleBufferGetFormatDescription(sample),
              let basic = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee
        else { throw Failure.unsupported("no audio format") }
        guard basic.mSampleRate == Double(AudioMixer.sampleRate) else {
            throw Failure.sampleRate(basic.mSampleRate)
        }
        let frames = CMSampleBufferGetNumSamples(sample)
        guard frames > 0 else { return [] }
        let format = AVAudioFormat(cmAudioFormatDescription: description)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))
        else { throw Failure.unsupported("format \(basic.mFormatID)") }
        buffer.frameLength = AVAudioFrameCount(frames)
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sample, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        guard status == noErr else { throw Failure.unsupported("copy failed (\(status))") }
        return mono(buffer)
    }

    /// The average of the buffer's channels.
    static func mono(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        let interleaved = buffer.format.isInterleaved
        var out = [Float](repeating: 0, count: frames)
        let scale = 1 / Float(max(channels, 1))
        if let data = buffer.floatChannelData {
            for channel in 0..<channels {
                for frame in 0..<frames {
                    out[frame] += (interleaved ? data[0][frame * channels + channel] : data[channel][frame]) * scale
                }
            }
        } else if let data = buffer.int16ChannelData {
            let unit = scale / Float(Int16.max)
            for channel in 0..<channels {
                for frame in 0..<frames {
                    out[frame] += Float(interleaved ? data[0][frame * channels + channel] : data[channel][frame]) * unit
                }
            }
        }
        return out
    }

    /// The loudest sample, for the level meters.
    static func peak(_ samples: [Float]) -> Float {
        samples.reduce(0) { max($0, abs($1)) }
    }
}
