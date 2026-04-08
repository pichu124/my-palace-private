from mistralai import Mistral
import os
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from config.settings import DEFAULT_CONFIG


class MistralAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Mistral",
        model: str = "mistral-large-latest",
        persona: str = (
            "You are Mistral, an AI assistant made by Mistral AI. "
            "You are known for sharp logical reasoning and precise arguments. "
            "You cut through vague claims and demand evidence and specificity."
        ),
    ):
        super().__init__(name=name, persona=persona, model=model)
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def respond(self, history: ConversationHistory, topic: str, round_prompt: str = "") -> str:
        messages = [
            {"role": "system", "content": self.build_system_prompt(DEFAULT_CONFIG.topic_framing)},
        ]
        messages += history.format_for_openai(DEFAULT_CONFIG.max_context_messages)
        messages.append({
            "role": "user",
            "content": self.build_turn_instruction(topic, round_prompt),
        })

        response = self._client.chat.complete(
    model=self.model,
    messages=messages,
    max_tokens=1000,        # Tăng từ 500 lên 800
    temperature=DEFAULT_CONFIG.temperature,
)
        return response.choices[0].message.content.strip()