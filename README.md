# BrowserPilot

> Universal AI browser agent for multi-step web tasks.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-2EAD33?logo=playwright&logoColor=white)](https://playwright.dev/python/)
[![AI](https://img.shields.io/badge/AI-Tool%20Calling-111827)](https://platform.openai.com/docs/guides/function-calling)

**Портфолио:** [тестовое задание для CityCar](CITYCAR_TEST_TASK.md)

An autonomous browser agent for multi-step web tasks. It operates a real, visible Chromium
session, uses live page state instead of site-specific scripts, and keeps working until the
task is verified or human input is required.

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m playwright install chromium
cp .env.example .env
browserpilot
```

Configure an OpenAI-compatible gateway with `CUSTOM_API_KEY`, `CUSTOM_API_BASE_URL`, and
`CUSTOM_API_MODEL`. The browser profile (`BROWSER_PROFILE_DIR`) persists cookies and sessions.
Use `setup` in the CLI for manual registration, login, 2FA, or CAPTCHA; `done` returns control
to the agent. The process accepts multiple tasks and exits with `exit`.

## Technical solution

### Automation library

**Playwright for Python** was selected over Selenium and Puppeteer because it provides reliable
async browser control, auto-waiting, persistent contexts, popup/page events, and first-class
support for Chromium. A persistent context makes authentication state reusable without copying
cookies or credentials into the application.

### AI SDK and language

The runtime is **Python 3.12+**, using the official **OpenAI Python SDK** against any
OpenAI-compatible API. Python gives a compact async orchestration layer, strong typing through
modern annotations, and a straightforward path to LangGraph or local-model integrations later.
The agent does not depend on LangChain: the control loop is intentionally explicit and easy to
audit.

### Page information extraction

The `inspect_page` tool builds a bounded, structured snapshot from the live DOM:

- current URL and title;
- visible text;
- visible interactive elements with generated `data-agent-id` selectors;
- roles, labels, input types, checked state, and native select options;
- active dialogs/overlays and scrollable containers;
- visible landmarks such as buttons, links, menu items, and cart controls.

Hidden elements are filtered by computed style and geometry. When a modal or fixed layer is
present, the snapshot is scoped to the top visible layer so stale catalog DOM cannot compete
with the active product card or checkout panel.

### Tool-calling architecture

The agent follows **observe → reason → act → verify**. The model can call generic tools for
search, navigation, inspection, scrolling, clicking, text entry, keyboard input, native option
selection, overlay closing, history navigation, waiting, clarification, and completion.
Selectors are accepted only from the latest inspection snapshot. Tool failures are returned to
the model as structured results so it can recover instead of terminating the task.

Completion is guarded: a normal or truncated model response cannot finish a task. `finish` is
accepted only after a real browser action and explicit progress. API calls have timeouts and
retries. Recent task summaries and URLs are persisted in `MEMORY_FILE` and supplied to later
tasks.

### Dynamic pages, popups, and forms

The browser uses one active tab. Newly opened pages become active and extra tabs are closed.
Visible overlays are detected by ARIA semantics or fixed/absolute geometry and can be closed
with Escape or a visible close control. Internal modal scrolling is exposed as a tool target.
Forms use live selectors; native `<select>` controls use `select_option`, while custom radio,
checkbox, card, cart, and checkout controls are selected from visible labels and landmarks.
CAPTCHA, 2FA, irreversible purchases, deletion, and message submission remain human-controlled.

### MCP decision

**MCP is not used in the core runtime.** The agent already has a narrow, typed browser tool
surface and does not need a remote tool registry. MCP would be useful when composing multiple
external systems (calendar, CRM, payments, or internal APIs); it can be added as an adapter
without changing the browser or memory layers.

## Research and engineering process

1. Compared Playwright, Selenium, and Puppeteer for persistent sessions, dynamic UI behavior,
   and debugging ergonomics; selected Playwright.
2. Compared provider SDKs and chose OpenAI-compatible tool calling to support hosted gateways,
   GLM proxies, OpenRouter, and local servers without changing agent logic.
3. Prototyped a minimal observe/act loop, then tested it against search results, product
   modals, lazy scrolling, cart drawers, and checkout controls.
4. Investigated failures from stale DOM, modal overlays, popup tabs, invalid key names,
   truncated responses, and lost clarification context.
5. Added visibility-scoped snapshots, single-tab recovery, normalized keyboard input,
   persistent memory, retries, completion guards, and human-in-the-loop safety boundaries.

## Safety

Keep `.env`, `.browser-profile`, and `.browser-memory.json` private. The agent does not bypass
CAPTCHA or 2FA, and it must ask for confirmation before irreversible actions.
