import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, spyOn, test } from "bun:test";
import { NextRequest } from "next/server";

import * as codexAuth from "@/lib/codexAuth";
import * as chatgpt from "@/lib/transcribe/chatgpt";
import * as local from "@/lib/transcribe/local";
import * as whispercpp from "@/lib/transcribe/whispercpp";
import * as backend from "@/lib/transcribeBackend";

import { POST } from "./route";

const payload = "recorded audio fixture";
const restores: Array<() => void> = [];

afterEach(() => {
  for (const restore of restores.splice(0).reverse()) restore();
});

async function upload(filename: string, type: string, body: BlobPart = payload, language = "uk") {
  const form = new FormData();
  form.append("file", new Blob([body], { type }), filename);
  if (language) form.append("language", language);
  // Serialize and parse real multipart bytes: returning a fabricated FormData
  // would hide Bun's filename-based MIME inference.
  const serialized = new Request("http://127.0.0.1/api/transcribe", { method: "POST", body: form });
  const req = new NextRequest(serialized.url, {
    method: "POST",
    headers: { "content-type": serialized.headers.get("content-type")!, host: "127.0.0.1" },
    body: await serialized.arrayBuffer(),
  });
  const parsed = (await req.clone().formData()).get("file") as File;
  return { parsedType: parsed.type, response: await POST(req) };
}

describe("transcription multipart media guard", () => {
  test.each([
    ["dictation.webm", "audio/webm", "video/webm", "audio/webm"],
    ["dictation.weba", "audio/webm", "audio/webm", "audio/webm"],
    ["dictation.ogv", "video/ogg", "video/ogg", "audio/ogg"],
    ["dictation.ogg", "audio/ogg", "audio/ogg", "audio/ogg"],
    ["dictation.mp4", "audio/mp4", "video/mp4", "audio/mp4"],
    ["dictation.m4a", "audio/mp4", "audio/x-m4a", "audio/x-m4a"],
    ["dictation.mp3", "audio/mpeg", "audio/mpeg", "audio/mpeg"],
    ["dictation", "audio/webm", "", "audio/webm"],
  ])("transcribes %s through the ChatGPT branch", async (filename, type, parsedType, upstreamType) => {
    const select = spyOn(backend, "resolveTranscribeBackend").mockReturnValue("chatgpt");
    restores.push(() => select.mockRestore());
    const auth = spyOn(codexAuth, "readCodexAuth").mockReturnValue({ accessToken: "fixture", accountId: "fixture" });
    restores.push(() => auth.mockRestore());
    let audioPath = "";
    const transcribe = spyOn(chatgpt, "callTranscribe").mockImplementation(async (_auth, filePath, mime, language) => {
      audioPath = filePath;
      expect(fs.readFileSync(filePath, "utf8")).toBe(payload);
      expect(mime).toBe(upstreamType);
      expect(language).toBe("uk");
      return { status: 200, body: JSON.stringify({ text: "Transcribed speech" }) };
    });
    restores.push(() => transcribe.mockRestore());

    const result = await upload(filename, type);
    expect(result.parsedType).toBe(parsedType);
    expect(result.response.status).toBe(200);
    expect(await result.response.json()).toEqual({ text: "Transcribed speech" });
    expect(transcribe).toHaveBeenCalledTimes(1);
    expect(fs.existsSync(audioPath)).toBe(false);
  });

  test.each([
    ["image.png", "image/png"],
    ["text.txt", "text/plain"],
    ["archive.zip", "application/zip"],
    ["movie.avi", "video/x-msvideo"],
  ])("rejects %s before selecting a backend", async (filename, type) => {
    const select = spyOn(backend, "resolveTranscribeBackend");
    restores.push(() => select.mockRestore());
    const { response } = await upload(filename, type);
    expect(response.status).toBe(415);
    expect(await response.json()).toEqual({ error: "expected audio" });
    expect(select).not.toHaveBeenCalled();
  });
});

/* A RIFF/WAVE header followed by `size - 44` bytes of silence. */
function wavBytes(size = 64): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(size);
  bytes.set(new TextEncoder().encode("RIFF"), 0);
  bytes.set(new TextEncoder().encode("WAVEfmt "), 8);
  return bytes;
}

function selectBackend(id: backend.TranscribeBackend) {
  const select = spyOn(backend, "resolveTranscribeBackend").mockReturnValue(id);
  restores.push(() => select.mockRestore());
}

function whisperCppReady(available = true) {
  const status = spyOn(whispercpp, "whisperCppStatus").mockReturnValue({
    available,
    binary: available ? "/fixture/whisper-cli" : null,
    model: available ? "/fixture/ggml-medium-q8_0.bin" : null,
    keyPath: "/fixture/whisper-cli",
    hint: available ? "" : "whisper-cli is not installed (brew install whisper-cpp) — run scripts/setup-whispercpp.sh",
  });
  restores.push(() => status.mockRestore());
}

describe("whispercpp branch", () => {
  test("hands a sniffed WAV to whisper-cli with the language, and removes the temp file", async () => {
    selectBackend("whispercpp");
    whisperCppReady();
    let audioPath = "";
    const transcribe = spyOn(whispercpp, "whisperCppTranscribe").mockImplementation(async (bin, model, file, language) => {
      audioPath = file;
      expect(bin).toBe("/fixture/whisper-cli");
      expect(model).toBe("/fixture/ggml-medium-q8_0.bin");
      expect(file.endsWith(".wav")).toBe(true);
      expect(Uint8Array.from(fs.readFileSync(file))).toEqual(wavBytes());
      expect(language).toBe("uk");
      return { text: "Привіт" };
    });
    restores.push(() => transcribe.mockRestore());

    const { response } = await upload("dictation.wav", "audio/wav", wavBytes());
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ text: "Привіт" });
    expect(fs.existsSync(audioPath)).toBe(false);
  });

  test("an empty language reaches whisper-cli empty (it runs -l auto)", async () => {
    selectBackend("whispercpp");
    whisperCppReady();
    const transcribe = spyOn(whispercpp, "whisperCppTranscribe").mockResolvedValue({ text: "hi" });
    restores.push(() => transcribe.mockRestore());
    await upload("dictation.wav", "audio/wav", wavBytes(), "");
    expect(transcribe.mock.calls[0]![3]).toBe("");
  });

  test("a webm upload is refused with a readable reason instead of a whisper-cli failure", async () => {
    selectBackend("whispercpp");
    whisperCppReady();
    const transcribe = spyOn(whispercpp, "whisperCppTranscribe");
    restores.push(() => transcribe.mockRestore());
    const { response } = await upload("dictation.webm", "audio/webm");
    expect(response.status).toBe(415);
    /* Machine-readable, so the client re-encodes and resends instead of
       telling the operator to reload. */
    expect(await response.json()).toMatchObject({ batchFormat: "wav", error: expect.stringContaining("WAV") });
    expect(transcribe).not.toHaveBeenCalled();
  });

  test("a missing binary or model answers 503 naming what is missing", async () => {
    selectBackend("whispercpp");
    whisperCppReady(false);
    const { response } = await upload("dictation.wav", "audio/wav", wavBytes());
    expect(response.status).toBe(503);
    expect((await response.json()).error).toContain("whisper-cli is not installed");
  });

  test("a whisper-cli failure surfaces as a 502 with its reason", async () => {
    selectBackend("whispercpp");
    whisperCppReady();
    const transcribe = spyOn(whispercpp, "whisperCppTranscribe").mockRejectedValue(new Error("timed out after 120 s"));
    restores.push(() => transcribe.mockRestore());
    const { response } = await upload("dictation.wav", "audio/wav", wavBytes());
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: "whisper.cpp: timed out after 120 s" });
  });

  test("a whisper-cli run past its timeout answers 502 and its temp WAV is removed", async () => {
    selectBackend("whispercpp");
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llv-whispercpp-route-"));
    restores.push(() => fs.rmSync(dir, { recursive: true, force: true }));
    const argFile = path.join(dir, "audio-path");
    const bin = path.join(dir, "whisper-cli");
    /* $4 is the -f argument: the temp WAV the route wrote. */
    fs.writeFileSync(bin, `#!/bin/sh\necho "$4" > "${argFile}"\nexec /bin/sleep 30\n`, { mode: 0o755 });
    const status = spyOn(whispercpp, "whisperCppStatus").mockReturnValue({
      available: true,
      binary: bin,
      model: "/fixture/ggml-medium-q8_0.bin",
      keyPath: "/fixture/ggml-medium-q8_0.bin",
      hint: "",
    });
    restores.push(() => status.mockRestore());
    const previous = process.env.LLV_WHISPERCPP_TIMEOUT_MS;
    process.env.LLV_WHISPERCPP_TIMEOUT_MS = "300";
    restores.push(() => {
      if (previous === undefined) delete process.env.LLV_WHISPERCPP_TIMEOUT_MS;
      else process.env.LLV_WHISPERCPP_TIMEOUT_MS = previous;
    });

    const { response } = await upload("dictation.wav", "audio/wav", wavBytes());
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: "whisper.cpp: timed out after 0.3 s" });
    const audioPath = fs.readFileSync(argFile, "utf8").trim();
    expect(audioPath.endsWith(".wav")).toBe(true);
    expect(fs.existsSync(audioPath)).toBe(false);
  });

  test("WAV gets the 20 MB allowance; non-WAV keeps the 16 MB cap", async () => {
    selectBackend("whispercpp");
    whisperCppReady();
    const transcribe = spyOn(whispercpp, "whisperCppTranscribe").mockResolvedValue({ text: "long" });
    restores.push(() => transcribe.mockRestore());
    const eighteenMb = 18 * 1024 * 1024;
    expect((await upload("dictation.wav", "audio/wav", wavBytes(eighteenMb))).response.status).toBe(200);
    expect((await upload("dictation.webm", "audio/webm", new Uint8Array(eighteenMb))).response.status).toBe(413);
    expect((await upload("dictation.wav", "audio/wav", wavBytes(21 * 1024 * 1024))).response.status).toBe(413);
  });
});

describe("unavailable backend reasons", () => {
  test("an unset faster-whisper answers 503 with the setup step, before spawning anything", async () => {
    selectBackend("local");
    const ready = spyOn(local, "localWhisperReady").mockReturnValue(false);
    restores.push(() => ready.mockRestore());
    const transcribe = spyOn(local, "localTranscribe");
    restores.push(() => transcribe.mockRestore());
    const { response } = await upload("dictation.webm", "audio/webm");
    expect(response.status).toBe(503);
    expect((await response.json()).error).toContain("scripts/setup-whisper.sh");
    expect(transcribe).not.toHaveBeenCalled();
  });
});
