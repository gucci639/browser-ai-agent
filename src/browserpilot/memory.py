from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Memory:
    def __init__(self, path: Path, limit: int = 30) -> None:
        self.path = path
        self.limit = limit
        self.entries: list[dict[str, Any]] = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.entries = [item for item in data if isinstance(item, dict)][-limit:]
            except (OSError, json.JSONDecodeError):
                self.entries = []

    def context(self) -> str:
        if not self.entries:
            return "No previous task memory."
        return json.dumps(self.entries[-self.limit :], ensure_ascii=False)

    def add(self, task: str, result: str, url: str) -> None:
        self.entries.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "result": result,
                "url": url,
            }
        )
        self.entries = self.entries[-self.limit :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")
