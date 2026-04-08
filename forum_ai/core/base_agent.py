from abc import ABC, abstractmethod
from core.history import ConversationHistory


class BaseAgent(ABC):
    def __init__(self, name: str, persona: str, model: str):
        self.name = name
        self.persona = persona
        self.model = model

    @abstractmethod
    def respond(
        self,
        history: ConversationHistory,
        topic: str,
        round_prompt: str = "",
    ) -> str:
        ...

    def build_system_prompt(self, topic_framing: str) -> str:
        return (
            f"You are {self.name}. {self.persona}\n\n"
            f"{topic_framing}"
        )

    def build_turn_instruction(self, topic: str, round_prompt: str) -> str:
        """Tạo instruction cụ thể cho từng lượt nói."""
        return (
            f"Topic: {topic}\n\n"
            f"Your instruction for this turn: {round_prompt}\n\n"
            f"Please respond now."
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.model!r}>"