import { afterEach, describe, expect, spyOn, test } from "bun:test";
import { NextRequest } from "next/server";

import { POST } from "@/app/api/transcribe/route";
import { encodeWavPcm16, WAV_SAMPLE_RATE } from "@/lib/audio/wav";
import * as whispercpp from "@/lib/transcribe/whispercpp";
import * as backend from "@/lib/transcribeBackend";

import { uploadRecording } from "./useDictation";

const restores: Array<() => void> = [];
afterEach(() => {
  for (const restore of restores.splice(0).reverse()) restore();
});

/* Any bytes do: the webm is never decoded, toWav stands in for the browser. */
const webm = new Blob(["\x1aE\xdf\xa3 recorded webm/opus"], { type: "audio/webm" });
const toWav = async () => new Blob([encodeWavPcm16(new Float32Array(1600), WAV_SAMPLE_RATE)], { type: "audio/wav" });

/** The real /api/transcribe handler behind the client's send. */
async function sendToRoute(form: FormData): Promise<Response> {
  const serialized = new Request("http://127.0.0.1/api/transcribe", { method: "POST", body: form });
  return POST(
    new NextRequest(serialized.url, {
      method: "POST",
      headers: { "content-type": serialized.headers.get("content-type")!, host: "127.0.0.1" },
      body: await serialized.arrayBuffer(),
    }),
  );
}

function whispercppSelected() {
  const select = spyOn(backend, "resolveTranscribeBackend").mockReturnValue("whispercpp");
  restores.push(() => select.mockRestore());
  const status = spyOn(whispercpp, "whisperCppStatus").mockReturnValue({
    available: true,
    binary: "/fixture/whisper-cli",
    model: "/fixture/ggml-medium-q8_0.bin",
    keyPath: "/fixture/ggml-medium-q8_0.bin",
    hint: "",
  });
  restores.push(() => status.mockRestore());
  const transcribe = spyOn(whispercpp, "whisperCppTranscribe").mockResolvedValue({ text: "Привіт" });
  restores.push(() => transcribe.mockRestore());
  return transcribe;
}

describe("batch upload with a stale format", () => {
  test("a webm sent to whispercpp is re-encoded and resent once: one press transcribes, no reload", async () => {
    const transcribe = whispercppSelected();
    const sent: string[] = [];
    const result = await uploadRecording(webm, "webm", {
      send: (form) => {
        sent.push((form.get("file") as File).name);
        return sendToRoute(form);
      },
      toWav,
    });
    expect(result).toEqual({ ok: true, text: "Привіт", format: "wav" });
    expect(sent).toEqual(["dictation.webm", "dictation.wav"]);
    expect(transcribe).toHaveBeenCalledTimes(1);
  });

  test("a current wav format goes straight through", async () => {
    whispercppSelected();
    const sent: string[] = [];
    const result = await uploadRecording(webm, "wav", {
      send: (form) => {
        sent.push((form.get("file") as File).name);
        return sendToRoute(form);
      },
      toWav,
    });
    expect(result).toMatchObject({ ok: true, format: "wav" });
    expect(sent).toEqual(["dictation.wav"]);
  });

  test("the resend happens at most once, and a failed WAV encode is reported as such", async () => {
    let calls = 0;
    const refuse = async () => {
      calls += 1;
      return Response.json({ error: "whisper.cpp reads WAV only", batchFormat: "wav" }, { status: 415 });
    };
    expect(await uploadRecording(webm, "webm", { send: refuse, toWav })).toEqual({
      ok: false,
      error: "whisper.cpp reads WAV only",
      format: "wav",
    });
    expect(calls).toBe(2);

    const failed = await uploadRecording(webm, "webm", {
      send: refuse,
      toWav: async () => {
        throw new Error("decode failed");
      },
    });
    expect(failed).toMatchObject({ ok: false, wavFailed: true });
  });

  test("a 415 without batchFormat is not retried", async () => {
    let calls = 0;
    const result = await uploadRecording(webm, "webm", {
      send: async () => {
        calls += 1;
        return Response.json({ error: "expected audio" }, { status: 415 });
      },
      toWav,
    });
    expect(result).toEqual({ ok: false, error: "expected audio", format: "webm" });
    expect(calls).toBe(1);
  });
});
