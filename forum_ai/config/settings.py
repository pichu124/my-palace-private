from dataclasses import dataclass


@dataclass
class ForumConfig:
    max_turns: int = 9
    max_context_messages: int = 20
    synthesis_model: str = "mistral-large-latest"
    temperature: float = 0.75
    max_tokens_per_turn: int = 500
    
    # Prompt chung cho tất cả agents
    topic_framing: str = (
    "You are participating in a high-quality AI research forum. "
    "Rules you MUST follow:\n"
    "1. Be concise but substantive (3-4 paragraphs max)\n"
    "2. NEVER repeat what previous speakers already said\n"
    "3. Always ADD new information, examples, or angles\n"
    "4. If you AGREE with someone, say why briefly then extend their point\n"
    "5. If you DISAGREE, be direct and explain your reasoning clearly\n"
    "6. Use concrete examples, data, or analogies when possible\n"
    "7. ALWAYS cite your data source (e.g. IMF, Bloomberg, Glassnode)\n"
    "8. NEVER fabricate statistics — if unsure, say 'estimated' or 'approximately'\n"
    "9. End each message with ONE sharp question or challenge for others\n"
    "10. Sign off with your name on the last line"
),

    # Prompt riêng cho từng lượt — thay đổi theo vòng
    round_prompts: tuple = (
    "This is the OPENING round. Introduce your core perspective on the topic. "
    "Be bold and take a clear stance. Do not sit on the fence.",

    "This is the DEBATE round. You MUST directly respond to at least ONE specific "
    "point made by another participant. Quote them briefly then challenge or extend it. "
    "Be sharp and specific.",

    "This is the REBUTTAL round. Defend your position or concede specific points with evidence. "
    "Do NOT repeat arguments already made. Add NEW data or angles.",

    "This is the CLOSING round. Give your UNIQUE final verdict with a probability estimate. "
    "IMPORTANT: Your conclusion MUST differ from other participants — check what they said "
    "and make sure your final position, probability, and reasoning are distinctly your own. "
    "What specifically changed your mind? What remains unresolved?",
)


DEFAULT_CONFIG = ForumConfig()