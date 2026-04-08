from openai import OpenAI
import os
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from config.settings import DEFAULT_CONFIG


class GroqAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Llama3",
        model: str = "llama-3.3-70b-versatile",
        persona: str = (
            "You are Llama 3, an open-source AI by Meta hosted on Groq. "
            "You are STUBBORN and data-driven. You do NOT change your position "
            "unless presented with hard evidence — not just rhetoric. "
            "When you concede a point, you concede ONLY that specific point, "
            "not your entire thesis. You always maintain a core position "
            "even when updating your probability estimate. "
            "Never fully adopt another agent's conclusion — always keep a distinctive angle."
        ),
    ):
        super().__init__(name=name, persona=persona, model=model)
        self._client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    def build_turn_instruction(self, topic: str, round_prompt: str) -> str:
        return (
            f"Topic: {topic}\n\n"
            f"Your instruction for this turn: {round_prompt}\n\n"
            f"CRITICAL REMINDERS:\n"
            f"1. Only cite sources you are CERTAIN exist (e.g. 'IMF World Economic Outlook 2024')\n"
            f"2. If unsure of exact figures, say 'approximately' or 'estimated ~X'\n"
            f"3. NEVER invent specific study names, percentages, or report titles\n"
            f"4. Good example: 'Vietnam GDP grew approximately 6-7% (World Bank estimate)'\n"
            f"5. Bad example: 'According to Study X (2024), exactly 73% of...'\n\n"
            f"Please respond now."
        )

    def respond(self, history: ConversationHistory, topic: str, round_prompt: str = "") -> str:
        messages = [
            {"role": "system", "content": self.build_system_prompt(DEFAULT_CONFIG.topic_framing)},
        ]
        messages += history.format_for_openai(DEFAULT_CONFIG.max_context_messages)
        messages.append({
            "role": "user",
            "content": self.build_turn_instruction(topic, round_prompt),
        })

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_CONFIG.max_tokens_per_turn,
            temperature=DEFAULT_CONFIG.temperature,
        )
        return response.choices[0].message.content.strip()