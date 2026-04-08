from openai import OpenAI
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from config.settings import DEFAULT_CONFIG
import os


class DeepSeekAgent(BaseAgent):
    def __init__(
        self,
        name: str = "DeepSeek",
        model: str = "deepseek-chat",
        persona: str = (
            "You are DeepSeek, an AI by DeepSeek AI. "
            "You are known for rigorous logical reasoning and technical depth."
        ),
    ):
        super().__init__(name=name, persona=persona, model=model)
        self._client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

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