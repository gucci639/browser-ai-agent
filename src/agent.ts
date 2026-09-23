import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import type { Page } from "playwright";
import { Memory } from "./memory.js";

export type AgentEvent =
  | { type: "thought"; text: string }
  | { type: "action"; name: string; input: unknown }
  | { type: "result"; text: string }
  | { type: "question"; text: string };

type ToolHandler = (input: Record<string, unknown>) => Promise<string>;
type QuestionHandler = (question: string) => Promise<string>;

const keyAliases: Record<string, string> = {
  ENTER: "Enter",
  RETURN: "Enter",
  ESC: "Escape",
  ESCAPE: "Escape",
  BACKSPACE: "Backspace",
  TAB: "Tab",
  SPACE: " ",
  CTRL: "Control",
  CMD: "Meta",
  COMMAND: "Meta",
  ALT: "Alt",
  SHIFT: "Shift",
  ARROWUP: "ArrowUp",
  ARROWDOWN: "ArrowDown",
  ARROWLEFT: "ArrowLeft",
  ARROWRIGHT: "ArrowRight"
};

const tools: Anthropic.Tool[] = [
  {
    name: "click_text",
    description: "Click a visible native button or link by its exact visible text from inspect_page.",
    input_schema: {
      type: "object",
      properties: { text: { type: "string" } },
      required: ["text"]
    }
  },
  {
    name: "close_overlay",
    description: "Close the visible modal, popup, drawer, or overlay using Escape.",
    input_schema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "go_back",
    description: "Go back one browser history entry when the current page is not the intended flow.",
    input_schema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "search_web",
    description:
      "Search the web for a website, product, restaurant, or other destination. Use this when the target is not already open.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"]
    }
  },
  {
    name: "inspect_page",
    description:
      "Read the current page. Returns URL, title, visible text and an accessibility-oriented list of interactive elements with stable generated selectors. Always inspect before acting when the page changed.",
    input_schema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "scroll",
    description: "Scroll the current page or a scrollable element to reveal more content.",
    input_schema: {
      type: "object",
      properties: {
        selector: { type: "string" },
        direction: { type: "string", enum: ["down", "up", "bottom", "top"] },
        amount: { type: "number", minimum: 200, maximum: 2000 }
      },
      required: ["direction"]
    }
  },
  {
    name: "select_option",
    description: "Select an option in a native HTML select using its visible label.",
    input_schema: {
      type: "object",
      properties: { selector: { type: "string" }, option: { type: "string" } },
      required: ["selector", "option"]
    }
  },
  {
    name: "navigate",
    description: "Navigate to a URL discovered from the page or required by the task.",
    input_schema: {
      type: "object",
      properties: { url: { type: "string", description: "Absolute URL." } },
      required: ["url"]
    }
  },
  {
    name: "click",
    description: "Click an element using a selector from inspect_page. Never invent a selector.",
    input_schema: {
      type: "object",
      properties: { selector: { type: "string" } },
      required: ["selector"]
    }
  },
  {
    name: "type",
    description: "Fill an input or textarea selected from inspect_page.",
    input_schema: {
      type: "object",
      properties: { selector: { type: "string" }, text: { type: "string" } },
      required: ["selector", "text"]
    }
  },
  {
    name: "press",
    description: "Press a keyboard key on an element selected from inspect_page.",
    input_schema: {
      type: "object",
      properties: { selector: { type: "string" }, key: { type: "string" } },
      required: ["selector", "key"]
    }
  },
  {
    name: "wait",
    description: "Wait for the page to settle after an action.",
    input_schema: {
      type: "object",
      properties: { milliseconds: { type: "number", minimum: 100, maximum: 10000 } },
      required: []
    }
  },
  {
    name: "finish",
    description: "Call when the user's task is complete. Include a concise result.",
    input_schema: {
      type: "object",
      properties: { summary: { type: "string" } },
      required: ["summary"]
    }
  },
  {
    name: "ask_user",
    description: "Ask the user for missing information or approval that cannot be inferred safely.",
    input_schema: {
      type: "object",
      properties: { question: { type: "string" } },
      required: ["question"]
    }
  }
];

const system = `You are BrowserPilot, an autonomous browser agent.
Solve the user's task by reasoning from the live page, not from hardcoded site knowledge.
Rules:
- Inspect the page before your first action and after navigation/clicks that change content.
- When the requested website or product is not already open, use search_web first.
- Do not type search terms into an unknown page; search_web opens real results.
- Use scroll when controls or cards are below the fold. Use select_option for native dropdowns;
  use inspect_page then click for custom radio, checkbox, or card choices.
- Prefer native buttons and cards on the current page. Keep one active tab and do not open
  search results in new tabs.
- After adding an item, inspect again, close the product modal if it blocks the page, find the
  visible cart/basket control and open it. In the cart, inspect again and continue to checkout;
  do not stop after merely adding an item.
- If the cart is a visible side panel, treat its primary button (often containing a total or
  checkout wording) as the next step. Use click_text when the visible button label is clear.
- Use only selectors returned by inspect_page. Never assume CSS selectors, URLs, or button labels.
- Prefer the smallest safe action and verify its outcome.
- Do not claim success without visible evidence.
- Never submit purchases, send messages, delete data, or perform other irreversible actions without explicit user confirmation.
- If a critical value or confirmation is missing, call ask_user and stop.
- Explain each meaningful step briefly. When complete call finish.
- Do not end with normal prose. Make at least one real browser action, verify it, then call finish.
- If a response is truncated or has no tool call, continue using tools rather than closing the task.`;

export class BrowserAgent {
  private readonly client: Anthropic;
  private readonly customClient?: OpenAI;
  private readonly model: string;
  private readonly maxSteps: number;
  private readonly handlers: Record<string, ToolHandler>;

  constructor(
    private page: Page,
    options: { apiKey: string; baseURL?: string; model?: string; maxSteps?: number; memory?: Memory; timeoutMs?: number }
  ) {
    this.timeoutMs = options.timeoutMs ?? 90_000;
    this.client = new Anthropic({
      apiKey: options.apiKey,
      baseURL: options.baseURL && options.baseURL.includes("anthropic") ? options.baseURL : undefined,
      timeout: this.timeoutMs
    });
    if (options.baseURL && !options.baseURL.includes("anthropic")) {
      this.customClient = new OpenAI({ apiKey: options.apiKey, baseURL: options.baseURL });
    }
    this.model = options.model ?? (this.customClient ? "gpt-4o-mini" : "claude-sonnet-4-5");
    this.maxSteps = options.maxSteps ?? 40;
    this.memory = options.memory;
    this.handlers = {
      inspect_page: async () => inspectPage(this.page),
      search_web: async ({ query }) => {
        const value = String(query).trim();
        if (!value) return "TOOL_ERROR: search query is empty";
        await this.page.goto(`https://www.google.com/search?q=${encodeURIComponent(value)}`, {
          waitUntil: "domcontentloaded",
          timeout: 30_000
        });
        return `Search results opened for: ${value}. Current URL: ${this.page.url()}`;
      },
      navigate: async ({ url }) => {
        await this.page.goto(String(url), { waitUntil: "domcontentloaded", timeout: 30_000 });
        return `Navigated to ${this.page.url()}`;
      },
      click: async ({ selector }) => {
        const before = new Set(this.page.context().pages());
        await this.page.locator(`${String(selector)}:visible`).first().click({ timeout: 10_000 });
        await this.page.waitForTimeout(250);
        const pages = this.page.context().pages();
        const newPages = pages.filter((candidate) => !before.has(candidate));
        const openedPage = newPages.at(-1);
        if (openedPage) this.page = openedPage;
        for (const candidate of this.page.context().pages()) {
          if (candidate !== this.page) await candidate.close();
        }
        if (newPages.length > 0) {
          await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
          return `Clicked and switched to the opened page: ${this.page.url()}. Extra tabs closed.`;
        }
        return "Clicked successfully.";
      },
      click_text: async ({ text }) => {
        const value = String(text).trim();
        if (!value) return "TOOL_ERROR: button text is empty";
        let locator = this.page.getByRole("button", { name: value, exact: true });
        if (!(await locator.count())) locator = this.page.getByRole("link", { name: value, exact: true });
        if (!(await locator.count())) {
          locator = this.page
            .locator("button:visible,a:visible,[role='button']:visible,[role='link']:visible,[role='menuitem']:visible")
            .filter({ hasText: value });
        }
        if (!(await locator.count())) locator = this.page.getByText(value, { exact: true }).locator("..");
        if (!(await locator.count())) return `TOOL_ERROR: no visible button or link named ${value}`;
        await locator.first().click({ timeout: 10_000 });
        return `Clicked visible control named ${value}.`;
      },
      close_overlay: async () => {
        await this.page.keyboard.press("Escape");
        await this.page.waitForTimeout(300);
        const close = this.page.locator(
          '[data-agent-overlay="active"] button:visible,' +
          '[data-agent-overlay="active"] [aria-label*="close" i]:visible,' +
          '[data-agent-overlay="active"] [title*="close" i]:visible'
        ).first();
        if (await close.count()) {
          await close.click({ timeout: 5_000 });
          await this.page.waitForTimeout(300);
          return "Closed the active product overlay with its close control.";
        }
        return "Escape sent; inspect the page to verify whether the overlay closed.";
      },
      go_back: async () => {
        await this.page.goBack({ waitUntil: "domcontentloaded", timeout: 15_000 });
        return `Went back to ${this.page.url()}`;
      },
      scroll: async ({ selector, direction, amount }) => {
        const dir = String(direction).toLowerCase();
        const size = Math.min(Math.max(Number(amount ?? 700), 200), 2000);
        if (!["down", "up", "bottom", "top"].includes(dir)) return "TOOL_ERROR: invalid scroll direction";
        const delta = dir === "down" ? size : -size;
        if (selector) {
          await this.page.locator(String(selector)).evaluate((element, data) => {
            const target = element as HTMLElement;
            if (data.direction === "bottom" || data.direction === "top") {
              target.scrollTo({ top: data.direction === "bottom" ? target.scrollHeight : 0, behavior: "instant" });
            } else {
              target.scrollBy({ top: data.delta, behavior: "instant" });
            }
          }, { direction: dir, delta });
        } else {
          await this.page.evaluate(({ direction, delta }) => {
            if (direction === "bottom" || direction === "top") {
              window.scrollTo({ top: direction === "bottom" ? document.body.scrollHeight : 0, behavior: "instant" });
            } else {
              window.scrollBy({ top: delta, behavior: "instant" });
            }
          }, { direction: dir, delta });
        }
        await this.page.waitForTimeout(300);
        return `Scrolled ${dir}; inspect the page again for newly visible content.`;
      },
      select_option: async ({ selector, option }) => {
        await this.page.locator(String(selector)).selectOption({ label: String(option) });
        return `Selected option: ${String(option)}`;
      },
      type: async ({ selector, text }) => {
        await this.page.locator(String(selector)).fill(String(text));
        return "Filled successfully.";
      },
      press: async ({ selector, key }) => {
        await this.page.locator(String(selector)).press(normalizeKey(String(key)));
        return "Key pressed successfully.";
      },
      wait: async ({ milliseconds }) => {
        await this.page.waitForTimeout(Math.min(Math.max(Number(milliseconds ?? 500), 100), 10_000));
        return "Wait complete.";
      },
      finish: async ({ summary }) => `TASK_COMPLETE: ${String(summary)}`,
      ask_user: async ({ question }) => `USER_INPUT_REQUIRED: ${String(question)}`
    };
  }
  private readonly memory?: Memory;
  private readonly timeoutMs: number;

  async run(
    task: string,
    onEvent: (event: AgentEvent) => void,
    answer?: string,
    askUser?: QuestionHandler
  ): Promise<string> {
    if (this.customClient) return this.runOpenAI(task, onEvent, answer, askUser);
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: `${this.memory?.context() ?? "No previous task memory."}\n\nCurrent task: ${answer ? `Original task: ${task}\nUser clarification: ${answer}` : task}` }
    ];
    let progress = false;

    for (let step = 1; step <= this.maxSteps; step += 1) {
      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: 2048,
        system,
        tools,
        messages
      }, {
        timeout: this.timeoutMs
      });
      const text = response.content
        .filter((block): block is Anthropic.TextBlock => block.type === "text")
        .map((block) => block.text)
        .join("\n");
      if (text) onEvent({ type: "thought", text });
      messages.push({ role: "assistant", content: response.content });

      const calls = response.content.filter(
        (block): block is Anthropic.ToolUseBlock => block.type === "tool_use"
      );
      if (calls.length === 0) {
        messages.push({ role: "user", content: "Continue with a useful browser tool call. Do not finish with prose." });
        continue;
      }

      const results: Anthropic.ToolResultBlockParam[] = [];
      for (const call of calls) {
        const handler = this.handlers[call.name];
        if (!handler) throw new Error(`Unknown tool requested: ${call.name}`);
        if (call.name === "ask_user") {
          onEvent({ type: "question", text: String((call.input as Record<string, unknown>).question ?? "") });
        } else {
          onEvent({ type: "action", name: call.name, input: call.input });
        }
        if (!["inspect_page", "wait", "finish", "ask_user"].includes(call.name)) progress = true;
        let result: string;
        try {
          result = await handler(call.input as Record<string, unknown>);
        } catch (error) {
          result = `TOOL_ERROR: ${error instanceof Error ? error.message : String(error)}`;
          onEvent({ type: "thought", text: `Tool ${call.name} failed: ${result}` });
        }
        results.push({ type: "tool_result", tool_use_id: call.id, content: result });
        if (result.startsWith("TASK_COMPLETE:")) {
          if (!progress) {
            messages.push({ role: "user", content: "No real browser action has happened yet. Continue with a useful action before finishing." });
            continue;
          }
          const summary = result.slice("TASK_COMPLETE:".length).trim();
          onEvent({ type: "result", text: summary });
          return summary;
        }
        if (result.startsWith("USER_INPUT_REQUIRED:")) {
          const question = result.slice("USER_INPUT_REQUIRED:".length).trim();
          onEvent({ type: "question", text: question });
          if (!askUser) return question;
          messages.push({ role: "user", content: `User clarification: ${await askUser(question)}` });
          break;
        }
      }

      messages.push({ role: "user", content: results });
    }
    throw new Error(`Agent exceeded the ${this.maxSteps}-step safety limit.`);
  }

  private async runOpenAI(
    task: string,
    onEvent: (event: AgentEvent) => void,
    answer?: string,
    askUser?: QuestionHandler
  ): Promise<string> {
    if (!this.customClient) throw new Error("Custom client is not configured.");
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: system },
      { role: "user", content: `${this.memory?.context() ?? "No previous task memory."}\n\nCurrent task: ${answer ? `Original task: ${task}\nUser clarification: ${answer}` : task}` }
    ];
    const openAITools: OpenAI.Chat.ChatCompletionTool[] = tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema as OpenAI.FunctionParameters
      }
    }));

    let progress = false;
    for (let step = 1; step <= this.maxSteps; step += 1) {
      const response = await this.customClient.chat.completions.create({
        model: this.model,
        messages,
        tools: openAITools,
        tool_choice: "auto"
      }, {
        timeout: this.timeoutMs
      });
      const message = response.choices[0]?.message;
      if (!message) throw new Error("Custom API returned an empty response.");
      if (message.content) onEvent({ type: "thought", text: message.content });
      messages.push(message);
      if (!message.tool_calls?.length) {
        messages.push({ role: "user", content: "Continue with a useful browser tool call. Do not finish with prose." });
        continue;
      }

      for (const call of message.tool_calls) {
        if (call.type !== "function") throw new Error(`Unsupported custom tool call type: ${call.type}`);
        const input = JSON.parse(call.function.arguments) as Record<string, unknown>;
        if (!["inspect_page", "wait", "finish", "ask_user"].includes(call.function.name)) progress = true;
        const handler = this.handlers[call.function.name];
        if (!handler) throw new Error(`Unknown tool requested: ${call.function.name}`);
        if (call.function.name === "ask_user") {
          onEvent({ type: "question", text: String(input.question ?? "") });
        } else {
          onEvent({ type: "action", name: call.function.name, input });
        }
        const result = await handler(input);
        messages.push({ role: "tool", tool_call_id: call.id, content: result });
        if (result.startsWith("TASK_COMPLETE:")) {
          if (!progress) {
            messages.push({ role: "user", content: "No real browser action has happened yet. Continue with a useful action before finishing." });
            continue;
          }
          const summary = result.slice("TASK_COMPLETE:".length).trim();
          onEvent({ type: "result", text: summary });
          return summary;
        }
        if (result.startsWith("USER_INPUT_REQUIRED:")) {
          const question = result.slice("USER_INPUT_REQUIRED:".length).trim();
          onEvent({ type: "question", text: question });
          if (!askUser) return question;
          messages.push({ role: "user", content: `User clarification: ${await askUser(question)}` });
          break;
        }
      }
    }
    throw new Error(`Agent exceeded the ${this.maxSteps}-step safety limit.`);
  }
}

function normalizeKey(key: string): string {
  return key
    .replaceAll("-", "+")
    .split("+")
    .map((part) => keyAliases[part.trim().toUpperCase()] ?? part.trim())
    .join("+");
}

async function inspectPage(page: Page): Promise<string> {
  const snapshot = await page.locator("body").evaluate((body) => {
    const visible = (element: Element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" &&
        Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
    };
    const layers = [...body.querySelectorAll<HTMLElement>("body *")]
      .filter((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        const z = Number.parseInt(style.zIndex, 10);
        return visible(element) && ["fixed", "absolute"].includes(style.position) &&
          Number.isFinite(z) && z >= 10 && rect.width > 280 && rect.height > 180;
      })
      .sort((a, b) => Number.parseInt(getComputedStyle(b).zIndex, 10) -
        Number.parseInt(getComputedStyle(a).zIndex, 10));
    const activeLayer = layers[0];
    for (const layer of layers) layer.removeAttribute("data-agent-overlay");
    activeLayer?.setAttribute("data-agent-overlay", "active");
    const scope = activeLayer ?? body;
    const elements = [...scope.querySelectorAll<HTMLElement>(
      "a,button,input,textarea,select,[role='button'],[role='link'],[role='checkbox'],[role='radio'],[contenteditable='true']"
    )].filter(visible).slice(0, 120);
    const interactive = elements.map((element, index) => {
      const tag = element.tagName.toLowerCase();
      const role = element.getAttribute("role") ?? tag;
      const text = (element.textContent || element.getAttribute("aria-label") || element.getAttribute("placeholder") || "").trim();
      const selector = `[data-agent-id="${index}"]`;
      element.setAttribute("data-agent-id", String(index));
      const rect = element.getBoundingClientRect();
      const options = tag === "select"
        ? [...(element as HTMLSelectElement).options].slice(0, 30).map((option) => ({
            label: option.text,
            value: option.value
          }))
        : undefined;
      const checked = (element as HTMLInputElement).checked;
      return {
        selector,
        role,
        text: text.slice(0, 160),
        type: element.getAttribute("type"),
        visibleInViewport: rect.bottom > 0 && rect.top < window.innerHeight,
        checked: element.getAttribute("aria-checked") ?? (checked ? true : undefined),
        options
      };
    });
    const dialogs = [...body.querySelectorAll<HTMLElement>("[role='dialog'],[aria-modal='true'],[data-agent-overlay='active']")]
      .filter(visible)
      .map((dialog) => ({ text: (dialog.innerText || "").slice(0, 8_000) }));
    const scrollables = [...body.querySelectorAll<HTMLElement>("*")]
      .filter((element) => {
        const style = getComputedStyle(element);
        return visible(element) && element.scrollHeight > element.clientHeight + 40 &&
          ["auto", "scroll"].includes(style.overflowY);
      })
      .slice(0, 10)
      .map((element, index) => {
        element.setAttribute("data-agent-scroll-id", String(index));
        return {
          selector: `[data-agent-scroll-id="${index}"]`,
          height: element.clientHeight,
          scrollHeight: element.scrollHeight
        };
      });
    const landmarks = [...body.querySelectorAll<HTMLElement>("button,a,[role='button'],[role='link'],[role='menuitem']")]
      .filter(visible)
      .map((element) => ({
        text: (element.innerText || element.getAttribute("aria-label") || "").trim().slice(0, 120),
        role: element.getAttribute("role") || element.tagName.toLowerCase(),
        inActiveOverlay: Boolean(activeLayer?.contains(element))
      }))
      .filter((item) => item.text)
      .slice(0, 80);
    return {
      text: (scope.textContent ?? body.textContent ?? "").slice(0, 12_000),
      interactive,
      landmarks,
      dialogs,
      scrollables,
      activeOverlay: Boolean(activeLayer)
    };
  });
  return JSON.stringify({ url: page.url(), title: await page.title(), ...snapshot });
}
