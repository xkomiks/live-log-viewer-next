import { float32ToPcm16 } from "@/lib/audio";

/* whisper.cpp's whisper-cli reads WAV, not the webm/opus a MediaRecorder
   produces, and the Viewer adds no ffmpeg for it. The browser already carries
   a decoder, so a recording bound for that backend is decoded and resampled
   by an OfflineAudioContext and re-wrapped here as 16 kHz mono PCM16. */
export const WAV_SAMPLE_RATE = 16_000;

/** A canonical 44-byte-header RIFF/WAVE file around mono little-endian PCM16. */
export function encodeWavPcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const pcm = float32ToPcm16(samples);
  const buffer = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + pcm.byteLength, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, pcm.byteLength, true);
  new Int16Array(buffer, 44).set(pcm);
  return buffer;
}

/** Averages every channel into one; a mono buffer is returned as-is. */
export function mixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 1) return channels[0]!;
  const length = Math.min(...channels.map((channel) => channel.length));
  const mono = new Float32Array(length);
  for (const channel of channels) {
    for (let i = 0; i < length; i += 1) mono[i]! += channel[i]! / channels.length;
  }
  return mono;
}

/** Decodes a recorded blob (webm/opus) and re-encodes it as 16 kHz mono WAV. */
export async function recordingToWav(blob: Blob): Promise<Blob> {
  /* decodeAudioData resamples to its context's rate; the length is irrelevant
     because the context is never rendered. */
  const ctx = new OfflineAudioContext(1, 1, WAV_SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  const channels = Array.from({ length: decoded.numberOfChannels }, (_, i) => decoded.getChannelData(i));
  return new Blob([encodeWavPcm16(mixToMono(channels), decoded.sampleRate)], { type: "audio/wav" });
}
