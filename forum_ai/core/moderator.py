from config.settings import ForumConfig, DEFAULT_CONFIG
from core.base_agent import BaseAgent
from core.history import ConversationHistory
from utils.display import print_message, print_moderator


class ForumModerator:
    def __init__(self, config: ForumConfig = DEFAULT_CONFIG, randomise_order: bool = False):
        self.config = config
        self.randomise_order = randomise_order
        self._agents: list[BaseAgent] = []
        self.history = ConversationHistory()

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents.append(agent)
        print_moderator(f"Registered agent: {agent.name} ({agent.model})")

    def _get_round_prompt(self, turn: int) -> str:
        """Trả về prompt theo vòng thảo luận."""
        n_agents = len(self._agents)
        round_index = turn // n_agents  # 0, 1, 2...
        prompts = self.config.round_prompts
        # Dùng prompt cuối nếu vượt quá số vòng định nghĩa
        idx = min(round_index, len(prompts) - 1)
        return prompts[idx]

    def _get_round_label(self, turn: int) -> str:
        n_agents = len(self._agents)
        round_num = turn // n_agents + 1
        labels = {1: "Opening", 2: "Debate", 3: "Closing"}
        return labels.get(round_num, f"Round {round_num}")

    def run(self, topic: str) -> ConversationHistory:
        if not self._agents:
            raise RuntimeError("No agents registered.")

        print_moderator(f"\n Forum opened. Topic: {topic}\n")
        self.history.add(role="user", name="User", content=topic)

        turn_count = 0
        while turn_count < self.config.max_turns:
            agent = self._agents[turn_count % len(self._agents)]
            round_label = self._get_round_label(turn_count)
            round_prompt = self._get_round_prompt(turn_count)

            print_moderator(
                f"Turn {turn_count + 1}/{self.config.max_turns} "
                f"[{round_label}] → {agent.name}"
            )

            try:
                response = agent.respond(
                    history=self.history,
                    topic=topic,
                    round_prompt=round_prompt,   # Truyền prompt theo vòng
                )
                self.history.add(role="assistant", name=agent.name, content=response)
                print_message(agent.name, round_label, response)

            except Exception as exc:
                error_msg = f"[Error from {agent.name}: {exc}]"
                print_moderator(error_msg)
                self.history.add(role="assistant", name=agent.name, content=error_msg)

            turn_count += 1

        print_moderator(f"\n Discussion complete after {turn_count} turns.\n")
        return self.history