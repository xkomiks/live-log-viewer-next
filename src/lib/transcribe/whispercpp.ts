import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { cacheEntryPath } from "@/lib/configDir";

import type { TranscribeResponse } from "./types";

/* whisper.cpp's whisper-cli, run against a ggml model on this machine. Handy
   (the desktop dictation app) ships its whisper model as .gguf, which
   whisper.cpp does not load, so the model normally lives in the Viewer's own
   cache (scripts/setup-whispercpp.sh downloads it there). Handy's HF cache is
   still searched for a ggml .bin in case a model of that format lands there.
   whisper-cli reads WAV, not the webm/opus MediaRecorder produces, so the
   browser encodes 16 kHz mono WAV for this backend (the token route's 409
   names the batch format). */
const WHISPERCPP_TIMEOUT_MS = 120_000;

/** LLV_WHISPERCPP_TIMEOUT_MS (a positive integer), else 120 s. */
export function whisperCppTimeoutMs(): number {
  const env = Number(process.env.LLV_WHISPERCPP_TIMEOUT_MS);
  return Number.isInteger(env) && env > 0 ? env : WHISPERCPP_TIMEOUT_MS;
}
export const WHISPERCPP_DEFAULT_MODEL = "ggml-medium-q8_0.bin";
const BINARY = "whisper-cli";
/** Where Homebrew puts whisper-cli (Apple silicon first, then Intel). */
export const FALLBACK_BIN_DIRS: readonly string[] = ["/opt/homebrew/bin", "/usr/local/bin"];
/* whisper.cpp's own model format; Handy's .gguf files fail to load. The
   Silero VAD model shares the ggml-*.bin naming and the cache dir, so it is
   excluded here and resolved on its own. */
const MODEL_FILE_RE = /^(?:ggml|whisper)(?!.*silero).*\.bin$/i;
const VAD_FILE_RE = /^ggml-silero.*\.bin$/i;
export const WHISPERCPP_DEFAULT_VAD_MODEL = "ggml-silero-v5.1.2.bin";
const HANDY_HF_PREFIX = "models--handy-computer--";

/** Viewer-owned model dir; the setup script downloads into it. */
export function whisperCppModelDir(): string {
  return cacheEntryPath("whispercpp");
}

function isExecutable(file: string): boolean {
  try {
    fs.accessSync(file, fs.constants.X_OK);
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

function isFile(file: string): boolean {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

/** LLV_WHISPERCPP_BIN, else whisper-cli on PATH, else the Homebrew bin dirs
    (`fallbackDirs`, a parameter so a test can run as a host without them). */
export function resolveWhisperCppBinary(fallbackDirs: readonly string[] = FALLBACK_BIN_DIRS): string | null {
  const env = process.env.LLV_WHISPERCPP_BIN?.trim();
  if (env) return isExecutable(env) ? env : null;
  const dirs = [...(process.env.PATH ?? "").split(path.delimiter).filter(Boolean), ...fallbackDirs];
  for (const dir of dirs) {
    const candidate = path.join(dir, BINARY);
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

/* Read per call: Bun's os.homedir() does not follow a HOME set after start. */
function homeDir(): string {
  return process.env.HOME?.trim() || os.homedir();
}

function hfHubDir(): string {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (process.env.HF_HOME) return path.join(process.env.HF_HOME, "hub");
  return path.join(homeDir(), ".cache", "huggingface", "hub");
}

function handySettingsPath(): string {
  return path.join(homeDir(), "Library", "Application Support", "com.pais.handy", "settings_store.json");
}

/** Handy's selected_model ("<org>/<repo>/<file>") resolved through the HF cache. */
function handySelectedModel(): string | null {
  let selected: unknown;
  try {
    const json = JSON.parse(fs.readFileSync(handySettingsPath(), "utf8")) as {
      selected_model?: unknown;
      settings?: { selected_model?: unknown };
    };
    selected = json.settings?.selected_model ?? json.selected_model;
  } catch {
    return null;
  }
  if (typeof selected !== "string") return null;
  const parts = selected.split("/");
  if (parts.length < 3) return null;
  const [org, repo, ...rest] = parts;
  const fileName = rest.join("/");
  if (!MODEL_FILE_RE.test(path.basename(fileName))) return null;
  const snapshots = path.join(hfHubDir(), `models--${org}--${repo}`, "snapshots");
  return newest(listDirs(snapshots).map((dir) => path.join(dir, fileName)).filter(isFile));
}

function listDirs(dir: string): string[] {
  try {
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(dir, entry.name));
  } catch {
    return [];
  }
}

function modelFilesIn(dir: string, pattern: RegExp = MODEL_FILE_RE): string[] {
  try {
    return fs
      .readdirSync(dir)
      .filter((name) => pattern.test(name))
      .map((name) => path.join(dir, name))
      .filter(isFile);
  } catch {
    return [];
  }
}

function newest(files: string[]): string | null {
  let best: { file: string; mtime: number } | null = null;
  for (const file of files) {
    /* A file can vanish after it was listed (the setup script renames its
       .part download into place); it is skipped, never a resolution error. */
    let mtime: number;
    try {
      mtime = fs.statSync(file).mtimeMs;
    } catch {
      continue;
    }
    if (!best || mtime > best.mtime) best = { file, mtime };
  }
  return best?.file ?? null;
}

/**
 * LLV_WHISPERCPP_MODEL, else the newest ggml model in the Viewer cache, else
 * Handy's selected model when it is a loadable ggml file, else the newest ggml
 * file in any handy-computer HF cache snapshot.
 */
export function resolveWhisperCppModel(): string | null {
  const env = process.env.LLV_WHISPERCPP_MODEL?.trim();
  if (env) return isFile(env) ? env : null;
  const cached = newest(modelFilesIn(whisperCppModelDir()));
  if (cached) return cached;
  const handy = handySelectedModel();
  if (handy) return handy;
  const hub = hfHubDir();
  const handyRepos = listDirs(hub).filter((dir) => path.basename(dir).startsWith(HANDY_HF_PREFIX));
  return newest(handyRepos.flatMap((repo) => listDirs(path.join(repo, "snapshots")).flatMap((dir) => modelFilesIn(dir))));
}

/**
 * The Silero VAD model: LLV_WHISPERCPP_VAD_MODEL, else the newest
 * ggml-silero*.bin in the Viewer cache. Optional — without it whisper-cli
 * still runs, but it hallucinates on silence (" you" for an accidental press),
 * which is why the setup script downloads it beside the model.
 */
export function resolveWhisperCppVadModel(): string | null {
  const env = process.env.LLV_WHISPERCPP_VAD_MODEL?.trim();
  if (env) return isFile(env) ? env : null;
  return newest(modelFilesIn(whisperCppModelDir(), VAD_FILE_RE));
}

export interface WhisperCppStatus {
  available: boolean;
  binary: string | null;
  model: string | null;
  /** Silero VAD model handed to whisper-cli when present; not required. */
  vadModel: string | null;
  /** What to fix, copyable: the missing binary's or model's expected path. */
  keyPath: string;
  /** Plain-words reason naming what is missing; empty when available. */
  hint: string;
}

export function whisperCppStatus(fallbackDirs: readonly string[] = FALLBACK_BIN_DIRS): WhisperCppStatus {
  const binary = resolveWhisperCppBinary(fallbackDirs);
  const model = resolveWhisperCppModel();
  const missing: string[] = [];
  if (!binary) {
    missing.push(
      process.env.LLV_WHISPERCPP_BIN?.trim()
        ? "LLV_WHISPERCPP_BIN is not an executable"
        : "whisper-cli is not installed (brew install whisper-cpp)",
    );
  }
  if (!model) {
    missing.push(
      process.env.LLV_WHISPERCPP_MODEL?.trim()
        ? "LLV_WHISPERCPP_MODEL does not name a file"
        : "no ggml whisper model found (Handy's .gguf does not load in whisper.cpp)",
    );
  }
  const keyPath = !binary
    ? process.env.LLV_WHISPERCPP_BIN?.trim() || path.join(FALLBACK_BIN_DIRS[0], BINARY)
    : !model
      ? process.env.LLV_WHISPERCPP_MODEL?.trim() || path.join(whisperCppModelDir(), WHISPERCPP_DEFAULT_MODEL)
      : model;
  return {
    available: missing.length === 0,
    binary,
    model,
    vadModel: resolveWhisperCppVadModel(),
    keyPath,
    hint: missing.length ? `${missing.join("; ")} — run scripts/setup-whispercpp.sh` : "",
  };
}

export function whisperCppReady(): boolean {
  return whisperCppStatus().available;
}

/** RIFF....WAVE: the only container whisper-cli is handed. */
export function isWav(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;
  const tag = (offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  return tag(0) === "RIFF" && tag(8) === "WAVE";
}

/* Non-speech markers whisper-cli prints in place of words: [BLANK_AUDIO],
   [MUSIC], (silence)… */
const MARKER_RE = /\[\s*(?:BLANK_AUDIO|SILENCE|MUSIC|NOISE|INAUDIBLE|NO_SPEECH)\s*\]|\(\s*(?:silence|music|noise|inaudible)\s*\)/gi;
/* A line made only of bracketed or starred fragments ("[Ukraїner Експедиція]",
   "(upbeat music)", "*sighs*") is a caption-style hallucination, never speech. */
const BRACKETED_ONLY_RE = /^(?:\s*(?:\[[^\]]*\]|\([^)]*\)|\*[^*]*\*)\s*)+$/;

/**
 * whisper-cli's stdout (one line per segment under -nt) as dictation text:
 * known non-speech markers and bracketed-only segments are dropped. The VAD
 * model is what keeps silence from producing words at all; this is the net
 * for a run without it and for the markers VAD still lets through.
 */
export function cleanWhisperCppOutput(stdout: string): string {
  return stdout
    .split("\n")
    .map((line) => line.replace(MARKER_RE, " ").trim())
    .filter((line) => line && !BRACKETED_ONLY_RE.test(line))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

export interface WhisperCppRunOptions {
  /** Silero VAD model; when set, whisper-cli runs with --vad -vm <model>. */
  vadModel?: string | null;
  timeoutMs?: number;
}

export function whisperCppTranscribe(
  binary: string,
  model: string,
  audioPath: string,
  language: string,
  { vadModel = null, timeoutMs = whisperCppTimeoutMs() }: WhisperCppRunOptions = {},
): Promise<TranscribeResponse> {
  /* whisper-cli takes a bare language code ("en", not "en-US"). */
  const args = ["-m", model, "-f", audioPath, "-l", language.split("-")[0] || "auto", "-nt", "-np"];
  if (vadModel) args.push("--vad", "-vm", vadModel);
  return new Promise((resolve, reject) => {
    execFile(
      binary,
      args,
      { maxBuffer: 4 * 1024 * 1024, timeout: timeoutMs },
      (error, stdout, stderr) => {
        if (error) {
          const timedOut = error.killed || (error as NodeJS.ErrnoException).code === "ETIMEDOUT";
          const detail = timedOut
            ? `timed out after ${timeoutMs / 1000} s`
            : String(stderr).trim().split("\n").at(-1) || error.message;
          reject(new Error(detail));
          return;
        }
        resolve({ text: cleanWhisperCppOutput(stdout) });
      },
    );
  });
}
