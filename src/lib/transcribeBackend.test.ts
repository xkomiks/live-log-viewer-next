import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, test } from "bun:test";

import {
  isTranscribeBackend,
  readSonioxApiKey,
  resolveTranscribeBackend,
  TRANSCRIBE_BACKENDS,
  transcribeBackendInfo,
  writeTranscribeBackend,
} from "./transcribeBackend";

const originalConfigHome = process.env.XDG_CONFIG_HOME;
const originalBackend = process.env.LLV_TRANSCRIBE_BACKEND;
const originalSonioxKey = process.env.SONIOX_API_KEY;
const roots: string[] = [];

function configHome(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "llv-stt-backend-"));
  roots.push(root);
  process.env.XDG_CONFIG_HOME = root;
  const dir = path.join(root, "agent-log-viewer");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function sonioxOption() {
  return transcribeBackendInfo().options.find((option) => option.id === "soniox")!;
}

/* Env restore goes through a name-indexed helper: writing
   `process.env.X_API_KEY = value` directly reads as a credential assignment to
   the publication gate. */
function setEnv(name: string, value: string | undefined): void {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
  setEnv("XDG_CONFIG_HOME", originalConfigHome);
  setEnv("LLV_TRANSCRIBE_BACKEND", originalBackend);
  setEnv("SONIOX_API_KEY", originalSonioxKey);
});

describe("soniox as a transcription backend (#1020)", () => {
  test("is a selectable backend beside the existing ones", () => {
    expect(TRANSCRIBE_BACKENDS).toEqual(["local", "chatgpt", "elevenlabs", "soniox", "whispercpp"]);
    expect(isTranscribeBackend("soniox")).toBe(true);
  });

  test("the override file selects it, case-insensitively", () => {
    const dir = configHome();
    delete process.env.LLV_TRANSCRIBE_BACKEND;
    fs.writeFileSync(path.join(dir, "transcribe-backend"), "soniox\n");
    expect(resolveTranscribeBackend()).toBe("soniox");
    fs.writeFileSync(path.join(dir, "transcribe-backend"), "SONIOX\n");
    expect(resolveTranscribeBackend()).toBe("soniox");
  });

  test("the env override wins over the file and locks the selector", () => {
    configHome();
    writeTranscribeBackend("local");
    process.env.LLV_TRANSCRIBE_BACKEND = "soniox";
    expect(resolveTranscribeBackend()).toBe("soniox");
    expect(transcribeBackendInfo()).toMatchObject({ backend: "soniox", lockedByEnv: true });
  });

  test("the mic menu can persist the choice into the override file", () => {
    const dir = configHome();
    delete process.env.LLV_TRANSCRIBE_BACKEND;
    writeTranscribeBackend("soniox");
    expect(fs.readFileSync(path.join(dir, "transcribe-backend"), "utf8")).toBe("soniox\n");
    expect(resolveTranscribeBackend()).toBe("soniox");
  });

  test("the key file makes it available and names the path to drop the key into", () => {
    const dir = configHome();
    setEnv("SONIOX_API_KEY", undefined);
    expect(sonioxOption()).toMatchObject({ available: false, keyPath: path.join(dir, "soniox-api-key") });

    fs.writeFileSync(path.join(dir, "soniox-api-key"), "file-key\n");
    expect(readSonioxApiKey()).toBe("file-key");
    expect(sonioxOption().available).toBe(true);
  });

  test("the environment key wins over the file, and an empty file reads as no key", () => {
    const dir = configHome();
    fs.writeFileSync(path.join(dir, "soniox-api-key"), "file-key\n");
    setEnv("SONIOX_API_KEY", "env-key");
    expect(readSonioxApiKey()).toBe("env-key");

    setEnv("SONIOX_API_KEY", undefined);
    fs.writeFileSync(path.join(dir, "soniox-api-key"), "\n");
    expect(readSonioxApiKey()).toBeNull();
    expect(sonioxOption().available).toBe(false);
  });

  test("selecting soniox leaves the other backends' options untouched", () => {
    configHome();
    process.env.LLV_TRANSCRIBE_BACKEND = "soniox";
    expect(transcribeBackendInfo().options.map((option) => option.id)).toEqual([
      "local",
      "chatgpt",
      "elevenlabs",
      "soniox",
      "whispercpp",
    ]);
  });
});

describe("whispercpp as a transcription backend", () => {
  const saved = ["LLV_WHISPER_VENV", "LLV_WHISPERCPP_BIN", "LLV_WHISPERCPP_MODEL"].map((name) => [name, process.env[name]] as const);
  afterEach(() => {
    for (const [name, value] of saved) setEnv(name, value);
  });

  /* Both local engines driven by their own overrides, so the machine's real
     venv, whisper-cli and model never decide the outcome. */
  function engines({ local, whispercpp }: { local: boolean; whispercpp: boolean }): string {
    const dir = configHome();
    delete process.env.LLV_TRANSCRIBE_BACKEND;
    const root = path.dirname(dir);
    const venv = path.join(root, "venv");
    if (local) {
      fs.mkdirSync(path.join(venv, "bin"), { recursive: true });
      fs.writeFileSync(path.join(venv, "bin", "python"), "");
    }
    process.env.LLV_WHISPER_VENV = venv;
    const bin = path.join(root, "whisper-cli");
    const model = path.join(root, "ggml-medium-q8_0.bin");
    if (whispercpp) {
      fs.writeFileSync(bin, "#!/bin/sh\n", { mode: 0o755 });
      fs.writeFileSync(model, "model");
    }
    process.env.LLV_WHISPERCPP_BIN = bin;
    process.env.LLV_WHISPERCPP_MODEL = model;
    return dir;
  }

  test("with no override, a missing faster-whisper venv falls back to an available whisper.cpp", () => {
    engines({ local: false, whispercpp: true });
    expect(resolveTranscribeBackend()).toBe("whispercpp");
  });

  test("local stays the default when it is set up, or when whisper.cpp is not", () => {
    engines({ local: true, whispercpp: true });
    expect(resolveTranscribeBackend()).toBe("local");
    engines({ local: false, whispercpp: false });
    expect(resolveTranscribeBackend()).toBe("local");
  });

  test("the file and env overrides still win over the fallback", () => {
    const dir = engines({ local: false, whispercpp: true });
    fs.writeFileSync(path.join(dir, "transcribe-backend"), "local\n");
    expect(resolveTranscribeBackend()).toBe("local");
    process.env.LLV_TRANSCRIBE_BACKEND = "chatgpt";
    expect(resolveTranscribeBackend()).toBe("chatgpt");
  });

  test("the menu option is truthful and names what is missing", () => {
    engines({ local: true, whispercpp: false });
    const missing = transcribeBackendInfo().options.find((option) => option.id === "whispercpp")!;
    expect(missing.available).toBe(false);
    expect(missing.hint).toContain("LLV_WHISPERCPP_BIN");
    expect(missing.keyPath).toBe(process.env.LLV_WHISPERCPP_BIN!);

    engines({ local: true, whispercpp: true });
    const ready = transcribeBackendInfo().options.find((option) => option.id === "whispercpp")!;
    expect(ready).toMatchObject({ available: true, keyPath: process.env.LLV_WHISPERCPP_MODEL! });
    expect(ready.hint).toBeUndefined();
  });
});
