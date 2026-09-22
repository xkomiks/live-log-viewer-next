import { describe, expect, test } from "bun:test";

import { isWav } from "@/lib/transcribe/whispercpp";

import { encodeWavPcm16, mixToMono, WAV_SAMPLE_RATE } from "./wav";

describe("browser-side WAV for the whispercpp backend", () => {
  test("writes a 16 kHz mono PCM16 RIFF/WAVE header the route sniffs as WAV", () => {
    const wav = encodeWavPcm16(new Float32Array([0, 0.5, -1, 1]), WAV_SAMPLE_RATE);
    const view = new DataView(wav);
    expect(wav.byteLength).toBe(44 + 8);
    expect(isWav(new Uint8Array(wav))).toBe(true);
    expect(view.getUint32(4, true)).toBe(36 + 8);
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16_000);
    expect(view.getUint32(28, true)).toBe(32_000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(8);
    expect(Array.from(new Int16Array(wav, 44))).toEqual([0, 16383, -32768, 32767]);
  });

  test("averages channels into mono", () => {
    expect(Array.from(mixToMono([new Float32Array([1, 0]), new Float32Array([0, 1])]))).toEqual([0.5, 0.5]);
    const mono = new Float32Array([0.25]);
    expect(mixToMono([mono])).toBe(mono);
  });
});
