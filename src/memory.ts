import fs from "node:fs";
import path from "node:path";

type MemoryEntry = { time: string; task: string; result: string; url: string };

export class Memory {
  private entries: MemoryEntry[] = [];
  constructor(private readonly file: string, private readonly limit = 30) {
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(file, "utf8"));
      if (Array.isArray(parsed)) this.entries = parsed.filter(isMemoryEntry).slice(-limit);
    } catch {
      this.entries = [];
    }
  }
  context(): string {
    return this.entries.length ? JSON.stringify(this.entries.slice(-this.limit)) : "No previous task memory.";
  }
  add(task: string, result: string, url: string): void {
    this.entries = [...this.entries, { time: new Date().toISOString(), task, result, url }].slice(-this.limit);
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify(this.entries, null, 2));
  }
}
function isMemoryEntry(value: unknown): value is MemoryEntry {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return ["time", "task", "result", "url"].every((key) => typeof item[key] === "string");
}
