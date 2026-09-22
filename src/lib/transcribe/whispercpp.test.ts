import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import {
  cleanWhisperCppOutput,
  isWav,
  resolveWhisperCppBinary,
  resolveWhisperCppModel,
  resolveWhisperCppVadModel,
  whisperCppStatus,
  whisperCppTranscribe,
} from "./whispercpp";

const ENV_KEYS = [
  "HOME",
  "XDG_CACHE_HOME",
  "HF_HOME",
  "HF_HUB_CACHE",
  "PATH",
  "LLV_WHISPERCPP_BIN",
  "LLV_WHISPERCPP_MODEL",
  "LLV_WHISPERCPP_TIMEOUT_MS",
  "LLV_WHISPERCPP_VAD_MODEL",
];
const saved = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));
let root = "";

function setEnv(name: string, value: string | undefined): void {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

function touch(file: string, mtimeSeconds?: number): string {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, "model");
  if (mtimeSeconds !== undefined) fs.utimesSync(file, mtimeSeconds, mtimeSeconds);
  return file;
}

function script(file: string, body: string): string {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `#!/bin/sh\n${body}\n`);
  fs.chmodSync(file, 0o755);
  return file;
}

const cacheDir = () => path.join(root, "cache", "agent-log-viewer", "whispercpp");
const hub = () => path.join(root, "home", ".cache", "huggingface", "hub");
const handySettings = () =>
  path.join(root, "home", "Library", "Application Support", "com.pais.handy", "settings_store.json");

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "llv-whispercpp-"));
  setEnv("HOME", path.join(root, "home"));
  setEnv("XDG_CACHE_HOME", path.join(root, "cache"));
  for (const key of ["HF_HOME", "HF_HUB_CACHE", "LLV_WHISPERCPP_BIN", "LLV_WHISPERCPP_MODEL", "LLV_WHISPERCPP_TIMEOUT_MS", "LLV_WHISPERCPP_VAD_MODEL"]) {
    setEnv(key, undefined);
  }
  setEnv("PATH", path.join(root, "bin"));
});

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
  for (const [key, value] of saved) setEnv(key, value);
});

describe("whisper.cpp model resolution", () => {
  test("LLV_WHISPERCPP_MODEL wins, and a missing file reads as no model", () => {
    const override = touch(path.join(root, "custom", "ggml-small.bin"));
    touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"));
    setEnv("LLV_WHISPERCPP_MODEL", override);
    expect(resolveWhisperCppModel()).toBe(override);
    setEnv("LLV_WHISPERCPP_MODEL", path.join(root, "custom", "absent.bin"));
    expect(resolveWhisperCppModel()).toBeNull();
  });

  test("a model that vanishes between listing and stat is skipped, not thrown", () => {
    const vanishing = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"), 2_000);
    const kept = touch(path.join(cacheDir(), "ggml-small.bin"), 1_000);
    /* The listing's isFile() sees it; the mtime read right after does not. */
    const realStat = fs.statSync;
    const seen = new Map<string, number>();
    const stat = spyOn(fs, "statSync").mockImplementation(((file: fs.PathLike, options?: fs.StatSyncOptions) => {
      const key = String(file);
      const calls = (seen.get(key) ?? 0) + 1;
      seen.set(key, calls);
      if (key === vanishing && calls > 1) throw Object.assign(new Error("ENOENT: no such file"), { code: "ENOENT" });
      return realStat(file, options);
    }) as typeof fs.statSync);
    try {
      expect(resolveWhisperCppModel()).toBe(kept);
      expect(() => whisperCppStatus()).not.toThrow();
    } finally {
      stat.mockRestore();
    }
  });

  test("the agreed order: a model in the Viewer cache wins over Handy's ggml selection", () => {
    /* Operator decision 2026-09-22: the backend's own ggml download comes
       before Handy's selected model (docs/transcription.md). */
    fs.mkdirSync(path.dirname(handySettings()), { recursive: true });
    fs.writeFileSync(handySettings(), JSON.stringify({ settings: { selected_model: "handy-computer/whisper-small/ggml-small.bin" } }));
    touch(path.join(hub(), "models--handy-computer--whisper-small", "snapshots", "rev1", "ggml-small.bin"), 2_000);
    const cached = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"), 1_000);
    expect(resolveWhisperCppModel()).toBe(cached);
  });

  test("the newest ggml model in the Viewer cache is used", () => {
    touch(path.join(cacheDir(), "ggml-small.bin"), 1_000);
    const newer = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"), 2_000);
    touch(path.join(cacheDir(), "notes.txt"), 3_000);
    expect(resolveWhisperCppModel()).toBe(newer);
  });

  test("Handy's selected model resolves through the HF cache when it is a ggml file", () => {
    fs.mkdirSync(path.dirname(handySettings()), { recursive: true });
    fs.writeFileSync(handySettings(), JSON.stringify({ settings: { selected_model: "handy-computer/whisper-small/ggml-small.bin" } }));
    const selected = touch(path.join(hub(), "models--handy-computer--whisper-small", "snapshots", "rev1", "ggml-small.bin"), 1_000);
    touch(path.join(hub(), "models--handy-computer--whisper-large", "snapshots", "rev1", "ggml-large.bin"), 2_000);
    expect(resolveWhisperCppModel()).toBe(selected);
  });

  test("Handy's .gguf selection is skipped (whisper.cpp cannot load it); a ggml file in its HF cache is found", () => {
    fs.mkdirSync(path.dirname(handySettings()), { recursive: true });
    fs.writeFileSync(
      handySettings(),
      JSON.stringify({ settings: { selected_model: "handy-computer/whisper-medium-gguf/whisper-medium-Q8_0.gguf" } }),
    );
    const repo = path.join(hub(), "models--handy-computer--whisper-medium-gguf", "snapshots", "rev1");
    touch(path.join(repo, "whisper-medium-Q8_0.gguf"));
    expect(resolveWhisperCppModel()).toBeNull();

    touch(path.join(hub(), "models--handy-computer--a", "snapshots", "r", "ggml-a.bin"), 1_000);
    const newest = touch(path.join(hub(), "models--handy-computer--b", "snapshots", "r", "ggml-b.bin"), 2_000);
    touch(path.join(hub(), "models--someone-else--c", "snapshots", "r", "ggml-c.bin"), 3_000);
    expect(resolveWhisperCppModel()).toBe(newest);
  });
});

describe("Silero VAD model", () => {
  test("resolves from the Viewer cache and is never taken for the whisper model", () => {
    const vad = touch(path.join(cacheDir(), "ggml-silero-v5.1.2.bin"), 3_000);
    expect(resolveWhisperCppVadModel()).toBe(vad);
    expect(resolveWhisperCppModel()).toBeNull();
    const model = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"), 1_000);
    expect(resolveWhisperCppModel()).toBe(model);
  });

  test("LLV_WHISPERCPP_VAD_MODEL wins, a missing file reads as none, and none still leaves the backend available", () => {
    touch(path.join(cacheDir(), "ggml-silero-v5.1.2.bin"));
    const override = touch(path.join(root, "vad", "silero.bin"));
    setEnv("LLV_WHISPERCPP_VAD_MODEL", override);
    expect(resolveWhisperCppVadModel()).toBe(override);
    setEnv("LLV_WHISPERCPP_VAD_MODEL", path.join(root, "vad", "absent.bin"));
    expect(resolveWhisperCppVadModel()).toBeNull();

    setEnv("LLV_WHISPERCPP_VAD_MODEL", undefined);
    fs.rmSync(path.join(cacheDir(), "ggml-silero-v5.1.2.bin"));
    setEnv("LLV_WHISPERCPP_BIN", script(path.join(root, "bin", "whisper-cli"), "exit 0"));
    touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"));
    expect(whisperCppStatus()).toMatchObject({ available: true, vadModel: null });
  });

  test("whisper-cli gets --vad -vm <model> exactly when the VAD model exists", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), 'printf "%s\\n" "$*"');
    setEnv("LLV_WHISPERCPP_BIN", bin);
    const model = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"));
    const run = async () => {
      const status = whisperCppStatus();
      return (await whisperCppTranscribe(bin, status.model!, "/a.wav", "", { vadModel: status.vadModel })).text;
    };
    const without = await run();
    expect(without).toBe(`-m ${model} -f /a.wav -l auto -nt -np`);
    expect(without).not.toContain("--vad");

    const vad = touch(path.join(cacheDir(), "ggml-silero-v5.1.2.bin"));
    expect(await run()).toBe(`-m ${model} -f /a.wav -l auto -nt -np --vad -vm ${vad}`);
  });
});

describe("non-speech output filter", () => {
  test("drops known markers and bracketed-only segments, keeps speech", () => {
    expect(cleanWhisperCppOutput("\n [BLANK_AUDIO]\n")).toBe("");
    expect(cleanWhisperCppOutput("\n [Ukraїner Експедиція]\n")).toBe("");
    expect(cleanWhisperCppOutput(" (upbeat music)\n *sighs*\n [ Silence ]\n")).toBe("");
    expect(cleanWhisperCppOutput(" Привіт, це перевірка.\n [BLANK_AUDIO]\n Друге речення.\n")).toBe(
      "Привіт, це перевірка. Друге речення.",
    );
    expect(cleanWhisperCppOutput(" Open the file [BLANK_AUDIO] now.")).toBe("Open the file now.");
    /* Brackets inside speech are speech. */
    expect(cleanWhisperCppOutput(" Call f(x) with [1, 2].")).toBe("Call f(x) with [1, 2].");
  });

  test("a whisper-cli run that prints only a marker transcribes to empty text", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), 'printf "\\n [BLANK_AUDIO]\\n"');
    expect((await whisperCppTranscribe(bin, "/m.bin", "/a.wav", "")).text).toBe("");
  });
});

describe("whisper.cpp binary resolution and status", () => {
  test("nothing installed anywhere: not available, the hint says to install, keyPath names Homebrew's path", () => {
    /* No env override, an empty PATH dir, and no fallback dirs: the host's own
       whisper-cli (if any) cannot leak into the answer. */
    touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"));
    expect(resolveWhisperCppBinary([])).toBeNull();
    const status = whisperCppStatus([]);
    expect(status.available).toBe(false);
    expect(status.binary).toBeNull();
    expect(status.hint).toBe("whisper-cli is not installed (brew install whisper-cpp) — run scripts/setup-whispercpp.sh");
    expect(status.keyPath).toBe("/opt/homebrew/bin/whisper-cli");
  });

  test("the fallback dirs are searched after PATH", () => {
    const fallback = script(path.join(root, "brew", "whisper-cli"), "exit 0");
    expect(resolveWhisperCppBinary([path.join(root, "brew")])).toBe(fallback);
    const onPath = script(path.join(root, "bin", "whisper-cli"), "exit 0");
    expect(resolveWhisperCppBinary([path.join(root, "brew")])).toBe(onPath);
  });

  test("LLV_WHISPERCPP_BIN wins over PATH; whisper-cli on PATH is found otherwise", () => {
    const onPath = script(path.join(root, "bin", "whisper-cli"), "exit 0");
    expect(resolveWhisperCppBinary()).toBe(onPath);
    const override = script(path.join(root, "other", "whisper-cli"), "exit 0");
    setEnv("LLV_WHISPERCPP_BIN", override);
    expect(resolveWhisperCppBinary()).toBe(override);
    setEnv("LLV_WHISPERCPP_BIN", path.join(root, "nope"));
    expect(resolveWhisperCppBinary()).toBeNull();
  });

  test("status is available only with binary AND model, and names what is missing", () => {
    setEnv("LLV_WHISPERCPP_BIN", path.join(root, "nope"));
    let status = whisperCppStatus();
    expect(status.available).toBe(false);
    expect(status.hint).toContain("LLV_WHISPERCPP_BIN");
    expect(status.hint).toContain("no ggml whisper model");
    expect(status.keyPath).toBe(path.join(root, "nope"));

    setEnv("LLV_WHISPERCPP_BIN", script(path.join(root, "bin", "whisper-cli"), "exit 0"));
    status = whisperCppStatus();
    expect(status.available).toBe(false);
    expect(status.hint).not.toContain("LLV_WHISPERCPP_BIN");
    expect(status.keyPath).toBe(path.join(cacheDir(), "ggml-medium-q8_0.bin"));

    const model = touch(path.join(cacheDir(), "ggml-medium-q8_0.bin"));
    status = whisperCppStatus();
    expect(status).toMatchObject({ available: true, model, hint: "" });
  });
});

describe("whisper-cli invocation", () => {
  test("passes model, file and language (auto when empty, bare code for a region tag)", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), 'printf " %s\\n" "$@"');
    const text = await whisperCppTranscribe(bin, "/m.bin", "/a.wav", "");
    expect(text.text).toBe("-m /m.bin -f /a.wav -l auto -nt -np");
    expect((await whisperCppTranscribe(bin, "/m.bin", "/a.wav", "en-US")).text).toContain("-l en ");
    expect((await whisperCppTranscribe(bin, "/m.bin", "/a.wav", "uk")).text).toContain("-l uk ");
  });

  test("a run past the timeout is killed and rejects with the timeout", async () => {
    const pidFile = path.join(root, "pid");
    const bin = script(path.join(root, "bin", "whisper-cli"), `echo $$ > "${pidFile}"; exec /bin/sleep 30`);
    const started = Date.now();
    await expect(whisperCppTranscribe(bin, "/m.bin", "/a.wav", "", { timeoutMs: 300 })).rejects.toThrow("timed out after 0.3 s");
    expect(Date.now() - started).toBeLessThan(10_000);
    const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
    expect(() => process.kill(pid, 0)).toThrow();
  });

  test("output past maxBuffer is reported as such, not as a timeout", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), 'i=0; while [ $i -lt 200 ]; do echo "a long transcript line"; i=$((i+1)); done');
    const run = whisperCppTranscribe(bin, "/m.bin", "/a.wav", "", { maxBufferBytes: 1024, timeoutMs: 10_000 });
    await expect(run).rejects.toThrow("output exceeded 1024 bytes");
    await expect(run).rejects.not.toThrow("timed out");
  });

  test("LLV_WHISPERCPP_TIMEOUT_MS sets the default timeout", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), "exec /bin/sleep 30");
    setEnv("LLV_WHISPERCPP_TIMEOUT_MS", "200");
    await expect(whisperCppTranscribe(bin, "/m.bin", "/a.wav", "")).rejects.toThrow("timed out after 0.2 s");
  });

  test("a failing run rejects with the last stderr line", async () => {
    const bin = script(path.join(root, "bin", "whisper-cli"), 'echo "error: failed to read audio" >&2; exit 3');
    await expect(whisperCppTranscribe(bin, "/m.bin", "/a.wav", "")).rejects.toThrow("failed to read audio");
  });

  test("isWav sniffs RIFF/WAVE only", () => {
    const enc = new TextEncoder();
    expect(isWav(enc.encode("RIFF\0\0\0\0WAVEfmt "))).toBe(true);
    expect(isWav(enc.encode("\x1aE\xdf\xa3 webm bytes"))).toBe(false);
    expect(isWav(enc.encode("RIFF"))).toBe(false);
  });
});
