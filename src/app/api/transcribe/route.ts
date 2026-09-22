import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { NextRequest, NextResponse } from "next/server";

import { readCodexAuth } from "@/lib/codexAuth";
import { rejectCrossOrigin } from "@/lib/sameOrigin";
import { callTranscribe } from "@/lib/transcribe/chatgpt";
import { elevenLabsTranscribe } from "@/lib/transcribe/elevenlabs";
import { localTranscribe, localWhisperReady } from "@/lib/transcribe/local";
import { sonioxTranscribe } from "@/lib/transcribe/soniox";
import type { TranscribeResponse } from "@/lib/transcribe/types";
import { isWav, whisperCppStatus, whisperCppTranscribe } from "@/lib/transcribe/whispercpp";
import { readElevenLabsApiKey, readSonioxApiKey, resolveTranscribeBackend } from "@/lib/transcribeBackend";
import type { ApiError } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_AUDIO_BYTES = 16 * 1024 * 1024;
/* Uncompressed 16 kHz mono PCM16 runs 1.92 MB a minute, so a recording at the
   10-minute cap is ~19.2 MB; only a sniffed WAV gets this larger allowance. */
const MAX_WAV_BYTES = 20 * 1024 * 1024;
const LANGUAGE_RE = /^[a-z]{2}(?:-[A-Z]{2})?$/;

export async function POST(
  req: NextRequest,
): Promise<NextResponse<TranscribeResponse | ApiError | (ApiError & { batchFormat: "wav" })>> {
  const rejection = rejectCrossOrigin(req);
  if (rejection) return rejection;

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ error: "expected multipart/form-data with a file field" }, { status: 400 });
  }
  const file = form.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "missing audio file in the file field" }, { status: 400 });
  }
  if (file.size > MAX_WAV_BYTES) {
    return NextResponse.json({ error: "audio is too large (20 MB limit for WAV, 16 MB otherwise)" }, { status: 413 });
  }
  // Audio-only recordings can carry a video container MIME. In particular,
  // Bun infers video/webm from dictation.webm when parsing multipart uploads.
  const type = file.type.split(";")[0].trim().toLowerCase();
  const containerAudioType = new Map([
    ["video/webm", "audio/webm"],
    ["video/ogg", "audio/ogg"],
    ["video/mp4", "audio/mp4"],
  ]).get(type);
  if (type && !type.startsWith("audio/") && !containerAudioType) {
    return NextResponse.json({ error: "expected audio" }, { status: 415 });
  }
  const mime = containerAudioType || type || "audio/webm";
  const rawLanguage = form.get("language");
  const language = typeof rawLanguage === "string" && LANGUAGE_RE.test(rawLanguage) ? rawLanguage : "";
  const bytes = new Uint8Array(await file.arrayBuffer());
  const wav = isWav(bytes);
  if (!wav && file.size > MAX_AUDIO_BYTES) {
    return NextResponse.json({ error: "audio is too large (16 MB limit)" }, { status: 413 });
  }
  const backend = resolveTranscribeBackend();

  if (backend === "local" && !localWhisperReady()) {
    return NextResponse.json(
      { error: "Faster Whisper is not set up — run scripts/setup-whisper.sh, or pick another method from the mic's right-click menu" },
      { status: 503 },
    );
  }

  if (backend === "whispercpp") {
    const status = whisperCppStatus();
    if (!status.available || !status.binary || !status.model) {
      return NextResponse.json({ error: `whisper.cpp is not set up: ${status.hint}` }, { status: 503 });
    }
    /* The client re-encodes the same recording as WAV and resends it once when
       it sees batchFormat here, so a stale format costs a round trip, not the
       dictation; the message only surfaces if that resend also fails. */
    if (!wav) {
      return NextResponse.json(
        { error: "whisper.cpp reads WAV only, and this recording was not WAV", batchFormat: "wav" as const },
        { status: 415 },
      );
    }
    const wavPath = path.join(os.tmpdir(), `viewer-dictation-${Date.now()}-${Math.floor(Math.random() * 1e6)}.wav`);
    try {
      fs.writeFileSync(wavPath, bytes);
      return NextResponse.json(await whisperCppTranscribe(status.binary, status.model, wavPath, language));
    } catch (error) {
      return NextResponse.json(
        { error: `whisper.cpp: ${error instanceof Error ? error.message : String(error)}` },
        { status: 502 },
      );
    } finally {
      fs.rmSync(wavPath, { force: true });
    }
  }

  if (backend === "elevenlabs") {
    const key = readElevenLabsApiKey();
    if (!key) {
      return NextResponse.json(
        { error: "missing ElevenLabs key (~/.config/agent-log-viewer/elevenlabs-api-key or ELEVENLABS_API_KEY)" },
        { status: 503 },
      );
    }
    try {
      return NextResponse.json(await elevenLabsTranscribe(key, file, language));
    } catch (error) {
      return NextResponse.json(
        { error: `ElevenLabs STT: ${error instanceof Error ? error.message : String(error)}` },
        { status: 502 },
      );
    }
  }

  if (backend === "soniox") {
    const key = readSonioxApiKey();
    if (!key) {
      return NextResponse.json(
        { error: "missing Soniox key (~/.config/agent-log-viewer/soniox-api-key or SONIOX_API_KEY)" },
        { status: 503 },
      );
    }
    try {
      return NextResponse.json(await sonioxTranscribe(key, file, language));
    } catch (error) {
      return NextResponse.json(
        { error: `Soniox STT: ${error instanceof Error ? error.message : String(error)}` },
        { status: 502 },
      );
    }
  }

  const tmpPath = path.join(
    os.tmpdir(),
    `viewer-dictation-${Date.now()}-${Math.floor(Math.random() * 1e6)}.${wav ? "wav" : "webm"}`,
  );
  try {
    fs.writeFileSync(tmpPath, bytes);

    if (backend === "local") {
      const result = await localTranscribe(tmpPath, language);
      return NextResponse.json(result);
    }

    const auth = readCodexAuth();
    if (!auth) {
      return NextResponse.json(
        { error: "missing Codex ChatGPT token (~/.codex/auth.json) — sign in to Codex" },
        { status: 503 },
      );
    }
    const upstream = await callTranscribe(auth, tmpPath, mime, language);
    if (upstream.status === 401) {
      return NextResponse.json({ error: "ChatGPT token expired — open Codex so it can refresh the token" }, { status: 502 });
    }
    if (upstream.status !== 200) {
      return NextResponse.json({ error: `transcription backend: HTTP ${upstream.status || "0 (network)"}` }, { status: 502 });
    }
    const json = JSON.parse(upstream.body) as { text?: unknown };
    return NextResponse.json({ text: typeof json.text === "string" ? json.text : "" });
  } catch (error) {
    const label = backend === "local" ? "local STT" : "transcription backend";
    return NextResponse.json(
      { error: `${label}: ${error instanceof Error ? error.message : String(error)}` },
      { status: 502 },
    );
  } finally {
    fs.rmSync(tmpPath, { force: true });
  }
}
