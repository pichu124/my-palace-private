from mistralai import Mistral
import os
from core.history import ConversationHistory
from config.settings import ForumConfig, DEFAULT_CONFIG
from utils.display import print_moderator


class DiscussionSynthesizer:
    SYNTHESIS_PROMPT = """You are a senior academic moderator writing the final summary of an AI forum.

Your synthesis must be sharp, specific, and actionable. Follow this exact structure:

**1. Key Arguments by Participant**
For each participant, summarize their UNIQUE contribution in 2-3 sentences. 
Do not repeat points covered by others.

**2. Points of Consensus**
What did all participants ultimately agree on? Be specific — avoid vague generalities.

**3. Core Disagreements**
What were the sharpest points of conflict? Who disagreed with whom and why?

**4. What Was Left Unresolved**
What important questions did the forum fail to answer? What would need further research?

**5. Recommendations for the User**
Based on the full discussion, give 3-5 concrete, prioritized action items.
Format as: [Priority: High/Medium/Low] Recommendation — reason in one sentence.

Be direct. Cut anything vague or redundant. The user needs clarity, not padding.
"""

    def __init__(self, config: ForumConfig = DEFAULT_CONFIG):
        self.config = config
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def synthesize(self, history: ConversationHistory, topic: str) -> str:
        print_moderator("Synthesizing discussion...")
        transcript = history.to_transcript()
        user_message = (
            f"Original topic: {topic}\n\n"
            f"Forum transcript:\n\n{transcript}"
        )
        response = self._client.chat.complete(
            model="mistral-large-latest",
            max_tokens=2500,
            messages=[
                {"role": "system", "content": self.SYNTHESIS_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()