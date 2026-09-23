import "dotenv/config";
import path from "node:path";
import { createInterface } from "node:readline/promises";
import process from "node:process";
import { chromium } from "playwright";
import { BrowserAgent } from "./agent.js";
import { Memory } from "./memory.js";

const apiKey = process.env.CUSTOM_API_KEY || process.env.ANTHROPIC_API_KEY;
if (!apiKey) {
  console.error("Missing API credentials. Set CUSTOM_API_KEY or ANTHROPIC_API_KEY in .env.");
  process.exit(1);
}

const profileDir = path.resolve(process.env.BROWSER_PROFILE_DIR ?? ".browser-profile");
const context = await chromium.launchPersistentContext(profileDir, {
  headless: process.env.HEADLESS === "true",
  slowMo: 80,
  viewport: { width: 1440, height: 900 }
});
const page = context.pages()[0] ?? (await context.newPage());
for (const extraPage of context.pages().slice(1)) await extraPage.close();
await page.goto(process.env.START_URL ?? "https://www.google.com", { waitUntil: "domcontentloaded" });
console.log(`\nBrowser ready at ${page.url()}\n`);
const rl = createInterface({ input: process.stdin, output: process.stdout });

const agent = new BrowserAgent(page, {
  apiKey,
  baseURL: process.env.CUSTOM_API_BASE_URL,
  model: process.env.CUSTOM_API_MODEL || process.env.ANTHROPIC_MODEL,
  maxSteps: Number(process.env.MAX_STEPS ?? 40),
  memory: new Memory(path.resolve(process.env.MEMORY_FILE ?? ".browser-memory.json")),
  timeoutMs: Number(process.env.API_TIMEOUT_SECONDS ?? 90) * 1000
});
const memory = new Memory(path.resolve(process.env.MEMORY_FILE ?? ".browser-memory.json"));

try {
  console.log("Commands: setup (manual login), exit (close browser)\n");
  let setupMode = false;
  while (true) {
    const task = (await rl.question("Task (or exit): ")).trim();
    if (!task) continue;
    if (["exit", "quit", "выход"].includes(task.toLowerCase())) break;
    if (setupMode) {
      if (task.toLowerCase() === "done") {
        setupMode = false;
        console.log("✅ Setup saved. You can send a task now.");
      } else {
        console.log("Finish login in the browser, then type `done`.");
      }
      continue;
    }
    if (task.toLowerCase() === "setup") {
      console.log("\n🔐 Setup mode: use the visible browser to register/login and complete 2FA.");
      console.log("When finished, type `done` at the next prompt. The session is saved in the profile.\n");
      setupMode = true;
      continue;
    }

    try {
      const result = await agent.run(task, (event) => {
        if (event.type === "thought") console.log(`\n🤔 ${event.text}`);
        if (event.type === "action") console.log(`🔧 ${event.name}`, event.input);
        if (event.type === "question") console.log(`\n❓ ${event.text}`);
        if (event.type === "result") console.log(`\n✅ ${event.text}`);
      }, undefined, async (question) => {
        return rl.question(`\nAnswer (${question}): `);
      });
      if (result) {
        console.log(`\n${result}`);
        memory.add(task, result, page.url());
      }
    } catch (error) {
      console.error("\nAgent failed:", error instanceof Error ? error.message : error);
    }
  }
} catch (error) {
  console.error("\nSession failed:", error instanceof Error ? error.message : error);
  process.exitCode = 1;
} finally {
  await rl.close();
  await context.close();
}
