from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    name: str
    content: str


class ConversationHistory:
    def __init__(self):
        self._messages: list[Message] = []

    def add(self, role: str, name: str, content: str) -> None:
        self._messages.append(Message(role=role, name=name, content=content))

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def get_window(self, max_messages: int) -> list[Message]:
        return self._messages[-max_messages:]

    def format_for_openai(self, max_messages: int) -> list[dict]:
        window = self.get_window(max_messages)
        result = []
        for msg in window:
            content = f"[{msg.name} said]: {msg.content}"
            result.append({"role": "user", "content": content})
        return result

    def format_for_anthropic(self, max_messages: int) -> list[dict]:
        window = self.get_window(max_messages)
        result = []
        for i, msg in enumerate(window):
            content = f"[{msg.name} said]: {msg.content}"
            role = "user" if i % 2 == 0 else "assistant"
            result.append({"role": role, "content": content})
        if result and result[-1]["role"] == "assistant":
            result.append({"role": "user", "content": "[Moderator]: Please share your perspective."})
        return result

    def format_for_gemini(self, max_messages: int) -> list[dict]:
        window = self.get_window(max_messages)
        result = []
        for msg in window:
            content = f"[{msg.name} said]: {msg.content}"
            result.append({"role": "user", "parts": [{"text": content}]})
        return result

    def to_transcript(self) -> str:
        lines = []
        for msg in self._messages:
            lines.append(f"--- {msg.name} ---\n{msg.content}\n")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._messages)
