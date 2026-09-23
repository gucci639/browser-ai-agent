from __future__ import annotations

import json
from urllib.parse import quote_plus
from collections.abc import Awaitable, Callable
from typing import Any, cast

from openai import AsyncOpenAI
from playwright.async_api import Page
from .memory import Memory

EventHandler = Callable[[str, dict[str, Any]], None]
QuestionHandler = Callable[[str], Awaitable[str]]
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

KEY_ALIASES = {
    "ENTER": "Enter",
    "RETURN": "Enter",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "TAB": "Tab",
    "SPACE": " ",
    "ARROWUP": "ArrowUp",
    "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight",
    "CTRL": "Control",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "ALT": "Alt",
    "SHIFT": "Shift",
}

SYSTEM_PROMPT = """You are BrowserPilot, an autonomous browser agent.
Solve the user's task from the live browser, never from hardcoded site knowledge.
When the requested website or product is not already open, use search_web first.
Do not try to type search terms into an unknown page; search_web opens real results.
Use scroll when controls or cards are below the fold. Use select_option for native dropdowns;
use inspect_page then click for custom radio, checkbox, or card choices. Prefer buttons and
cards on the current page. Do not open search results in new tabs and keep one active tab.
After adding an item, inspect again, close the product modal if it blocks the page, then
find the visible cart/basket control and open it. In the cart, inspect again and continue
to checkout; do not stop after merely adding an item.
If the cart is a visible side panel, treat its primary button (often containing a total or
checkout wording) as the next step. Use click_text when the visible button label is clear.
Inspect before acting and after any action that may change the page.
Use only selectors returned by inspect_page. Never invent selectors or URLs.
Prefer small, reversible actions and verify outcomes with inspect_page.
Never submit purchases, send messages, delete data, or do other irreversible actions
without explicit user confirmation. If a critical value or approval is missing, ask_user.
Call finish only when the task is visibly complete.
Do not end with a normal text response. You must use at least one real browser action
(navigate, click, type, press, or wait) and then verify the result before finish.
If an answer is long or gets truncated, continue the task with tool calls; never treat
a truncated response as completion."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_page",
            "description": "Return the current URL, title, visible text and interactive elements.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_overlay",
            "description": "Close the visible modal, popup, drawer, or overlay using Escape or its visible close button.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Go back one browser history entry when the current page is not the intended flow.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the current page or a scrollable element to reveal more content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Optional selector from inspect_page; omit for the page."},
                    "direction": {"type": "string", "enum": ["down", "up", "bottom", "top"]},
                    "amount": {"type": "integer", "minimum": 200, "maximum": 2000},
                },
                "required": ["direction"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select an option in a native HTML select using its visible label or value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "option": {"type": "string"},
                },
                "required": ["selector", "option"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a website, product, restaurant, or other destination. Use this when the target is not already open.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to an absolute URL discovered from the page.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_text",
            "description": "Click a visible native button or link by its exact/near-exact visible text from inspect_page.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click a selector returned by inspect_page.",
            "parameters": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Fill an input or textarea returned by inspect_page.",
            "parameters": {
                "type": "object",
                "properties": {"selector": {"type": "string"}, "text": {"type": "string"}},
                "required": ["selector", "text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press",
            "description": "Press a keyboard key on a selector returned by inspect_page.",
            "parameters": {
                "type": "object",
                "properties": {"selector": {"type": "string"}, "key": {"type": "string"}},
                "required": ["selector", "key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for dynamic page content to settle.",
            "parameters": {
                "type": "object",
                "properties": {"milliseconds": {"type": "integer", "minimum": 100, "maximum": 10000}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish only after the task is visibly complete.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask for missing information or explicit approval, then stop.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]


class BrowserAgent:
    def __init__(self, page: Page, client: AsyncOpenAI, model: str, max_steps: int, memory: Memory, timeout: float = 90) -> None:
        self.page = page
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.memory = memory
        self.timeout = timeout
        self.handlers: dict[str, ToolHandler] = {
            "inspect_page": self.inspect_page,
            "search_web": self.search_web,
            "navigate": self.navigate,
            "click": self.click,
            "click_text": self.click_text,
            "close_overlay": self.close_overlay,
            "go_back": self.go_back,
            "scroll": self.scroll,
            "select_option": self.select_option,
            "type": self.type_text,
            "press": self.press,
            "wait": self.wait,
            "finish": self.finish,
            "ask_user": self.ask_user,
        }

    async def run(
        self,
        task: str,
        on_event: EventHandler,
        ask_user: QuestionHandler | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Previous memory:\n{self.memory.context()}\n\nCurrent task:\n{task}"},
        ]
        progress = False
        for _ in range(self.max_steps):
            response = await self._completion(messages)
            message = response.choices[0].message
            if message.content:
                on_event("thought", {"text": message.content})
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                messages.append({"role": "user", "content": "Continue. Do not answer with prose; use a browser tool, verify progress, and call finish only when done."})
                continue

            for call in message.tool_calls:
                if call.type != "function":
                    raise RuntimeError(f"Unsupported tool call type: {call.type}")
                arguments = json.loads(call.function.arguments)
                handler = self.handlers.get(call.function.name)
                if handler is None:
                    raise RuntimeError(f"Unknown tool: {call.function.name}")
                on_event("action", {"name": call.function.name, "input": arguments})
                if call.function.name not in {"inspect_page", "wait", "finish", "ask_user"}:
                    progress = True
                try:
                    result = await handler(arguments)
                except Exception as error:
                    result = f"TOOL_ERROR: {type(error).__name__}: {error}"
                    on_event("tool_error", {"name": call.function.name, "error": str(error)})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if result.startswith("TASK_COMPLETE:"):
                    if not progress:
                        messages.append({"role": "user", "content": "You have not made a real browser action yet. Continue with a useful action before finishing."})
                        continue
                    summary = result.removeprefix("TASK_COMPLETE:").strip()
                    on_event("result", {"text": summary})
                    return summary
                if result.startswith("USER_INPUT_REQUIRED:"):
                    question = result.removeprefix("USER_INPUT_REQUIRED:").strip()
                    on_event("question", {"text": question})
                    if ask_user is None:
                        return question
                    answer = await ask_user(question)
                    messages.append({"role": "user", "content": f"User clarification: {answer}"})
                    break
        raise RuntimeError(f"Agent exceeded the {self.max_steps}-step safety limit.")

    async def _completion(self, messages: list[dict[str, Any]]) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, messages),
                    tools=cast(Any, TOOLS),
                    tool_choice="auto",
                    timeout=self.timeout,
                )
            except Exception as error:
                last_error = error
                if attempt < 2:
                    await self.page.wait_for_timeout(500 * (attempt + 1))
        raise RuntimeError(f"Model API failed after 3 attempts: {last_error}")

    async def inspect_page(self, _: dict[str, Any]) -> str:
        snapshot = await self.page.locator("body").evaluate(
            """body => {
              const visible = element => {
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' &&
                  Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
              };
              const layers = [...body.querySelectorAll('body *')].filter(element => {
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                const z = Number.parseInt(style.zIndex, 10);
                return visible(element) && ['fixed', 'absolute'].includes(style.position) &&
                  Number.isFinite(z) && z >= 10 && rect.width > 280 && rect.height > 180;
              }).sort((a, b) => {
                const za = Number.parseInt(getComputedStyle(a).zIndex, 10);
                const zb = Number.parseInt(getComputedStyle(b).zIndex, 10);
                return zb - za;
              });
              const activeLayer = layers[0] || null;
              for (const layer of layers) layer.removeAttribute('data-agent-overlay');
              if (activeLayer) activeLayer.setAttribute('data-agent-overlay', 'active');
              const scope = activeLayer || body;
              const elements = [...scope.querySelectorAll(
                'a,button,input,textarea,select,[role="button"],[role="link"],[role="checkbox"],[role="radio"],[contenteditable="true"]'
              )].filter(visible).slice(0, 120);
              const interactive = elements.map((element, index) => {
                element.setAttribute('data-agent-id', String(index));
                const rect = element.getBoundingClientRect();
                return {
                  selector: `[data-agent-id="${index}"]`,
                  role: element.getAttribute('role') || element.tagName.toLowerCase(),
                  text: (element.innerText || element.getAttribute('aria-label') ||
                    element.getAttribute('placeholder') || '').trim().slice(0, 160),
                  type: element.getAttribute('type'),
                  visibleInViewport: rect.bottom > 0 && rect.top < window.innerHeight,
                  checked: element.getAttribute('aria-checked') || element.checked || undefined,
                  options: element.tagName.toLowerCase() === 'select'
                    ? [...element.options].map(option => ({label: option.text, value: option.value})).slice(0, 30)
                    : undefined,
                };
              });
              const dialogs = [...body.querySelectorAll('[role="dialog"],[aria-modal="true"],[data-agent-overlay="active"]')]
                .filter(visible)
                .map(dialog => ({
                  text: (dialog.innerText || '').slice(0, 8000),
                  selector: dialog.getAttribute('data-agent-id')
                }));
              const scrollables = [...body.querySelectorAll('*')].filter(element => {
                const style = getComputedStyle(element);
                return visible(element) && element.scrollHeight > element.clientHeight + 40 &&
                  ['auto', 'scroll'].includes(style.overflowY);
              }).slice(0, 10).map((element, index) => {
                element.setAttribute('data-agent-scroll-id', String(index));
                return {selector: `[data-agent-scroll-id="${index}"]`, height: element.clientHeight, scrollHeight: element.scrollHeight};
              });
              const landmarks = [...body.querySelectorAll('button,a,[role="button"],[role="link"],[role="menuitem"]')]
                .filter(visible)
                .map(element => ({
                  text: (element.innerText || element.getAttribute('aria-label') || '').trim().slice(0, 120),
                  role: element.getAttribute('role') || element.tagName.toLowerCase(),
                  inActiveOverlay: Boolean(activeLayer && activeLayer.contains(element))
                })).filter(item => item.text).slice(0, 80);
              return {text: (scope.innerText || body.innerText || '').slice(0, 12000), interactive, landmarks, dialogs, scrollables,
                activeOverlay: Boolean(activeLayer)};
            }"""
        )
        return json.dumps(
            {"url": self.page.url, "title": await self.page.title(), **snapshot},
            ensure_ascii=False,
        )

    async def navigate(self, args: dict[str, Any]) -> str:
        await self.page.goto(str(args["url"]), wait_until="domcontentloaded", timeout=30_000)
        return f"Navigated to {self.page.url}"

    async def search_web(self, args: dict[str, Any]) -> str:
        query = str(args["query"]).strip()
        if not query:
            return "TOOL_ERROR: search query is empty"
        await self.page.goto(
            f"https://www.google.com/search?q={quote_plus(query)}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        return f"Search results opened for: {query}. Current URL: {self.page.url}"

    async def click(self, args: dict[str, Any]) -> str:
        before = set(self.page.context.pages)
        locator = self.page.locator(str(args["selector"])).first
        await locator.click(timeout=10_000)
        await self.page.wait_for_timeout(250)
        pages = self.page.context.pages
        new_pages = [page for page in pages if page not in before]
        if new_pages:
            self.page = new_pages[-1]
        for page in list(self.page.context.pages):
            if page is not self.page:
                await page.close()
        if new_pages:
            await self.page.wait_for_load_state("domcontentloaded", timeout=10_000)
            return f"Clicked and switched to the opened page: {self.page.url}. Extra tabs closed."
        return "Clicked successfully."

    async def click_text(self, args: dict[str, Any]) -> str:
        text = str(args["text"]).strip()
        if not text:
            return "TOOL_ERROR: button text is empty"
        locator = self.page.get_by_role("button", name=text, exact=True)
        if not await locator.count():
            locator = self.page.get_by_role("link", name=text, exact=True)
        if not await locator.count():
            locator = self.page.locator(
                "button:visible,a:visible,[role='button']:visible,[role='link']:visible"
            ).filter(has_text=text)
        if not await locator.count():
            locator = self.page.get_by_text(text, exact=True).locator("..")
        if not await locator.count():
            return f"TOOL_ERROR: no visible button or link named {text!r}"
        await locator.first.click(timeout=10_000)
        return f"Clicked visible control named {text!r}."

    async def close_overlay(self, _: dict[str, Any]) -> str:
        await self.page.keyboard.press("Escape")
        await self.page.wait_for_timeout(300)
        if await self.page.locator('[data-agent-overlay="active"]').count():
            close = self.page.locator(
                '[data-agent-overlay="active"] button:visible,'
                '[data-agent-overlay="active"] [aria-label*="close" i]:visible,'
                '[data-agent-overlay="active"] [title*="close" i]:visible'
            ).first
            if await close.count():
                await close.click(timeout=5_000)
                await self.page.wait_for_timeout(300)
                return "Closed the active product overlay with its close control."
        return "Escape sent; inspect the page to verify whether the overlay closed."

    async def go_back(self, _: dict[str, Any]) -> str:
        await self.page.go_back(wait_until="domcontentloaded", timeout=15_000)
        return f"Went back to {self.page.url}"

    async def scroll(self, args: dict[str, Any]) -> str:
        direction = str(args["direction"]).lower()
        amount = min(max(int(args.get("amount", 700)), 200), 2000)
        delta = amount if direction == "down" else -amount
        if direction not in {"down", "up", "bottom", "top"}:
            return "TOOL_ERROR: direction must be down, up, bottom, or top"
        selector = args.get("selector")
        if selector:
            locator = self.page.locator(str(selector))
            await locator.evaluate(
                """(element, data) => {
                  if (data.direction === "bottom" || data.direction === "top") {
                    element.scrollTo({ top: data.direction === "bottom" ? element.scrollHeight : 0, behavior: "instant" });
                  } else {
                    element.scrollBy({ top: data.delta, behavior: "instant" });
                  }
                }""",
                {"direction": direction, "delta": delta},
            )
        else:
            await self.page.evaluate(
                """data => {
                  if (data.direction === "bottom" || data.direction === "top") {
                    window.scrollTo({ top: data.direction === "bottom" ? document.body.scrollHeight : 0, behavior: "instant" });
                  } else {
                    window.scrollBy({ top: data.delta, behavior: "instant" });
                  }
                }""",
                {"direction": direction, "delta": delta},
            )
        await self.page.wait_for_timeout(300)
        return f"Scrolled {direction}; inspect the page again for newly visible content."

    async def select_option(self, args: dict[str, Any]) -> str:
        locator = self.page.locator(str(args["selector"]))
        option = str(args["option"])
        await locator.select_option(label=option)
        return f"Selected option: {option}"

    async def type_text(self, args: dict[str, Any]) -> str:
        await self.page.locator(str(args["selector"])).fill(str(args["text"]))
        return "Filled successfully."

    async def press(self, args: dict[str, Any]) -> str:
        key = normalize_key(str(args["key"]))
        await self.page.locator(str(args["selector"])).press(key)
        return "Key pressed successfully."

    async def wait(self, args: dict[str, Any]) -> str:
        milliseconds = min(max(int(args.get("milliseconds", 500)), 100), 10_000)
        await self.page.wait_for_timeout(milliseconds)
        return "Wait complete."

    async def finish(self, args: dict[str, Any]) -> str:
        return f"TASK_COMPLETE: {args['summary']}"

    async def ask_user(self, args: dict[str, Any]) -> str:
        return f"USER_INPUT_REQUIRED: {args['question']}"


def normalize_key(key: str) -> str:
    """Accept common model spellings while preserving Playwright key syntax."""
    parts = key.replace("-", "+").split("+")
    normalized = [KEY_ALIASES.get(part.strip().upper(), part.strip()) for part in parts]
    return "+".join(normalized)
