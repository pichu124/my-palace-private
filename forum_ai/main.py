from dotenv import load_dotenv
load_dotenv()

from config.settings import ForumConfig
from core.moderator import ForumModerator
from core.synthesizer import DiscussionSynthesizer
from agents.groq_agent import GroqAgent
from agents.gemini_agent import GeminiAgent
from agents.mistral_agent import MistralAgent
# from agents.deepseek_agent import DeepSeekAgent  # Bật lại khi nạp tiền
from utils.display import print_synthesis, console


def build_forum(config: ForumConfig) -> ForumModerator:
    moderator = ForumModerator(config=config)
    moderator.register_agent(GroqAgent())      # Llama3
    moderator.register_agent(GeminiAgent())    # Gemini 2.5 Flash
    moderator.register_agent(MistralAgent())   # Mistral Large
    return moderator


def main():
    config = ForumConfig(
    max_turns=12,         # 4 rounds x 3 agents
    max_tokens_per_turn=500,
    temperature=0.8,      # Tăng nhẹ để debate sôi nổi hơn
)

    topic = input("\n Enter forum topic: ").strip()
    if not topic:
        topic = "What are the most important unsolved problems in AI alignment?"

    moderator = build_forum(config)
    history = moderator.run(topic)

    synthesizer = DiscussionSynthesizer(config=config)
    conclusion = synthesizer.synthesize(history, topic)
    print_synthesis(conclusion)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\n\n")
        f.write(history.to_transcript())
        f.write("\n\n=== SYNTHESIS ===\n\n")
        f.write(conclusion)

    console.print("\n[dim]Transcript saved to transcript.txt[/dim]")


if __name__ == "__main__":
    main()