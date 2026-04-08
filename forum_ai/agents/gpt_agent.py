from openai import OpenAI
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from config.settings import DEFAULT_CONFIG


class GPTAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ChatGPT",
        model: str = "gpt-4o",
        persona: str = (
            "You are ChatGPT, an AI assistant made by OpenAI. "
            "You are analytical, structured, and present balanced perspectives."
        ),
    ):
        super().__init__(name=name, persona=persona, model=model)
        self._client = OpenAI()

    def respond(self, history: ConversationHistory, topic: str) -> str:
        messages = [
            {"role": "system", "content": self.build_system_prompt(DEFAULT_CONFIG.topic_framing)},
        ]
        messages += history.format_for_openai(DEFAULT_CONFIG.max_context_messages)
        messages.append({"role": "user", "content": "Please share your perspective on the discussion so far."})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_CONFIG.max_tokens_per_turn,
            temperature=DEFAULT_CONFIG.temperature,
        )
        return response.choices[0].message.content.strip()