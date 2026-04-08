from google import genai
from google.genai import types
import os
import time
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from config.settings import DEFAULT_CONFIG


class GeminiAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Gemini",
        model: str = "gemini-2.5-flash-lite",
        persona: str = (
            "You are Gemini, an AI assistant made by Google. "
            "You are CONTRARIAN by nature — your job is to find the angle others miss. "
            "You bring cross-disciplinary data: macro economics, geopolitics, satellite data, "
            "shipping flows. You never simply agree — you always add a twist or a challenge. "
            "Be bold, specific, and cite unconventional sources."
        ),
    ):
        super().__init__(name=name, persona=persona, model=model)
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def respond(
        self,
        history: ConversationHistory,
        topic: str,
        round_prompt: str = "",
    ) -> str:
        window = history.get_window(DEFAULT_CONFIG.max_context_messages)
        conversation = "\n".join(
            f"[{msg.name} said]: {msg.content}" for msg in window
        )
        full_prompt = (
            f"{self.build_system_prompt(DEFAULT_CONFIG.topic_framing)}\n\n"
            f"Conversation so far:\n{conversation}\n\n"
            f"{self.build_turn_instruction(topic, round_prompt)}\n\n"
            f"IMPORTANT: Write your COMPLETE response. Do not stop mid-sentence."
        )

        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=DEFAULT_CONFIG.temperature,
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                        ],
                    ),
                )

                if not response.text:
                    return "[Gemini: Response was blocked by safety filter.]"

                text = response.text.strip()
                if not text.endswith((".", "!", "?", "\n")):
                    text += "..."
                return text

            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    wait = (attempt + 1) * 10
                    print(f"⚙  Gemini 503 error, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise e