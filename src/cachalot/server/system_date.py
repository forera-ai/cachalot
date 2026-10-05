"""
Keep an agent's saved system block usable across days (HANDOFF section 18.66).

Hermes writes "Conversation started: <weekday, month day, year> (<time zone>)" into its system prompt at about token
6.2k of 22.5k, in front of 16k tokens of tool schemas. A new day changes one token there, every KV row after it
depends on that token, and the first message of the day prefills the whole block cold (227 s at 99 tok/s). With Hamed's
go (2026-10-05) the server shows the model the date of the first request it saw in the last few days instead of the
true one, so the saved block matches. The model reads a date up to `max_days` old; after that window the true date is
used (and recorded), which starts the next window. Only the date text of the leading system message is replaced.
Switch: CACHALOT_SYSTEM_DATE_REUSE=0 turns it off; CACHALOT_SYSTEM_DATE_REUSE_DAYS (default 7) is the window.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

_DATE_LINE = re.compile(r"(Conversation started: )([A-Z][a-z]+, [A-Z][a-z]+ \d{1,2}, \d{4})")
_FORMAT = "%A, %B %d, %Y"


def _parse(text: str) -> date | None:
    try:
        return datetime.strptime(text, _FORMAT).date()
    except ValueError:
        return None


class SystemDateReuse:
    """Remembers the true dates seen (a small JSON file next to the snapshots) and picks the one to show."""

    def __init__(self, path: Path | None, max_days: int = 7, enabled: bool = True):
        self.path = Path(path) if path else None
        self.max_days = max_days
        self.enabled = enabled
        self.reused = 0
        self.seen: list[str] = self._read()

    @classmethod
    def from_env(cls, path: Path | None, env=os.environ) -> SystemDateReuse:
        enabled = str(env.get("CACHALOT_SYSTEM_DATE_REUSE", "1")).strip().lower() not in ("0", "off", "false", "no")
        try:
            days = int(env.get("CACHALOT_SYSTEM_DATE_REUSE_DAYS", "7"))
        except ValueError:
            days = 7
        return cls(path, max(0, days), enabled)

    def _read(self) -> list[str]:
        if self.path is None:
            return []
        try:
            return [d for d in json.loads(self.path.read_text()) if _parse(d)]
        except (OSError, ValueError, TypeError):
            return []

    def _write(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.seen[-16:]))
            os.replace(tmp, self.path)
        except OSError:
            pass  # bookkeeping only

    def apply(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """`messages` with the leading system message's date line replaced where a recent saved date exists.

        Returns (messages, note); note is the log line when a date was replaced, else None.
        """
        if not self.enabled or not messages or messages[0].get("role") != "system":
            return messages, None
        content = messages[0].get("content")
        if not isinstance(content, str):
            return messages, None
        m = _DATE_LINE.search(content)
        if m is None:
            return messages, None
        true_text = m.group(2)
        today = _parse(true_text)
        if today is None:
            return messages, None
        # the oldest date still inside the window is the one the longest-lived saved block carries
        for old_text in self.seen:
            old = _parse(old_text)
            if old is not None and 0 <= (today - old).days <= self.max_days:
                if old_text == true_text:
                    return messages, None
                head = dict(messages[0], content=content[: m.start(2)] + old_text + content[m.end(2):])
                self.reused += 1
                return [head] + list(messages[1:]), f"system date: showing {old_text} for {true_text}"
        if true_text not in self.seen:
            self.seen.append(true_text)
            self._write()
        return messages, None
