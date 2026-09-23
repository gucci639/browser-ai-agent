from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI
from playwright.async_api import async_playwright

from .agent import BrowserAgent
from .config import Settings
from .memory import Memory


async def run() -> None:
    settings = Settings()
    settings.browser_profile_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=settings.custom_api_key, base_url=settings.custom_api_base_url)

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            str(settings.browser_profile_dir),
            headless=settings.headless,
            slow_mo=80,
            viewport={"width": 1440, "height": 900},
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            for extra_page in context.pages[1:]:
                await extra_page.close()
            await page.goto(settings.start_url, wait_until="domcontentloaded")
            memory = Memory(settings.memory_file)
            agent = BrowserAgent(page, client, settings.custom_api_model, settings.max_steps, memory, settings.api_timeout_seconds)
            print(f"\nBrowser ready at {page.url}\nCommands: setup (manual login), exit (close browser)\n")
            setup_mode = False
            while True:
                task = (await asyncio.to_thread(input, "Task (or exit): ")).strip()
                if not task:
                    continue
                if task.lower() in {"exit", "quit", "выход"}:
                    break
                if setup_mode:
                    if task.lower() == "done":
                        setup_mode = False
                        print("Setup saved. You can send a task now.")
                    else:
                        print("Finish login in the browser, then type `done`.")
                    continue
                if task.lower() == "setup":
                    setup_mode = True
                    print("Use the visible browser to register/login and complete 2FA.")
                    print("When finished, type `done`; the agent will not act in setup mode.")
                    continue
                try:
                    result = await agent.run(
                        task,
                        lambda kind, data: print_event(kind, data),
                        ask_user=lambda question: ask_for_clarification(question),
                    )
                    print(f"\n{result}\n")
                    memory.add(task, result, page.url)
                except Exception as error:
                    print(f"\nAgent failed: {error}\n")
        finally:
            try:
                await context.close()
            except Exception as error:
                print(f"Browser cleanup warning: {error}")
            await client.close()


def print_event(kind: str, data: dict[str, Any]) -> None:
    prefixes = {"thought": "🤔", "action": "🔧", "question": "❓", "result": "✅", "tool_error": "⚠️"}
    print(f"\n{prefixes.get(kind, '•')} {data.get('text', data)}")


async def ask_for_clarification(question: str) -> str:
    print("\nThe agent needs more information. Its context is preserved.")
    return await asyncio.to_thread(input, f"Answer ({question}): ")


def main() -> None:
    asyncio.run(run())
