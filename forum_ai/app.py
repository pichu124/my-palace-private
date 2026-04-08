"""
Forum AI — Streamlit UI
Multi-agent debate forum with real-time streaming, per-agent colors, and synthesis.
Run: streamlit run app.py
"""

import os
import time
import threading
import queue
from pathlib import Path

from dotenv import load_dotenv
# Load .env from the forum_ai directory (not cwd)
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

# ── Imports from project ─────────────────────────────────────────────────────

from config.settings import ForumConfig
from core.history import ConversationHistory, Message
from core.base_agent import BaseAgent
from agents.groq_agent import GroqAgent
from agents.gemini_agent import GeminiAgent
from agents.mistral_agent import MistralAgent
from agents.deepseek_agent import DeepSeekAgent

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forum AI",
    page_icon="🎙️",
    layout="wide",
)

# ── Agent colors (CSS-safe) ─────────────────────────────────────────────────

AGENT_COLORS = {
    "Llama3": "#22c55e",    # green
    "Gemini": "#3b82f6",    # blue
    "Mistral": "#a855f7",   # purple
    "DeepSeek": "#eab308",  # yellow
    "Moderator": "#6b7280", # gray
    "Synthesis": "#f59e0b", # amber
}

ROUND_LABELS = {0: "Opening", 1: "Debate", 2: "Rebuttal", 3: "Closing"}


def get_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#9ca3af")


def escape_dollars(text: str) -> str:
    """Escape $ signs so Streamlit doesn't render them as LaTeX."""
    return text.replace("$", "&#36;")


# ── Session state init ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {name, role, content, round_label}
if "running" not in st.session_state:
    st.session_state.running = False
if "synthesis" not in st.session_state:
    st.session_state.synthesis = ""
if "history" not in st.session_state:
    st.session_state.history = None


# ── Sidebar settings ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Forum Settings")

    n_agents = 3  # Llama3, Gemini, Mistral
    n_rounds = st.slider("Number of rounds", 1, 6, 3)
    max_turns = n_rounds * n_agents

    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    max_tokens = st.slider("Max tokens per turn", 200, 1500, 500, 50)

    st.divider()
    st.subheader("Active Agents")
    use_llama = st.checkbox("Llama3 (Groq)", value=True)
    use_gemini = st.checkbox("Gemini (Google)", value=True)
    use_mistral = st.checkbox("Mistral", value=True)
    use_deepseek = st.checkbox("DeepSeek", value=False)

    st.divider()
    env_status = []
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY"]:
        val = os.environ.get(key, "")
        status = "✅" if val else "❌"
        st.text(f"{status} {key}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.synthesis = ""
        st.session_state.running = False
        st.session_state.history = None
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────

st.title("🎙️ Forum AI")
st.caption("Multi-agent debate — Llama3 vs Gemini vs Mistral")

topic = st.text_input(
    "Enter debate topic",
    placeholder="e.g. Will AI replace most white-collar jobs within 10 years?",
    key="topic_input",
)


def build_agents():
    """Build list of active agents based on sidebar checkboxes."""
    agents = []
    if use_llama:
        agents.append(GroqAgent())
    if use_gemini:
        agents.append(GeminiAgent())
    if use_mistral:
        agents.append(MistralAgent())
    if use_deepseek:
        agents.append(DeepSeekAgent())
    return agents


def get_round_prompt(turn: int, n_agents: int, config: ForumConfig) -> str:
    round_index = turn // n_agents
    idx = min(round_index, len(config.round_prompts) - 1)
    return config.round_prompts[idx]


def get_round_label(turn: int, n_agents_count: int) -> str:
    round_index = turn // n_agents_count
    return ROUND_LABELS.get(round_index, f"Round {round_index + 1}")


def run_synthesis(history: ConversationHistory, topic_str: str, config: ForumConfig) -> str:
    """Run synthesis using Mistral."""
    from mistralai import Mistral as MistralClient

    SYNTHESIS_PROMPT = """You are a senior academic moderator writing the final summary of an AI forum.

Your synthesis must be sharp, specific, and actionable. Follow this exact structure:

**1. Key Arguments by Participant**
For each participant, summarize their UNIQUE contribution in 2-3 sentences.

**2. Points of Consensus**
What did all participants ultimately agree on?

**3. Core Disagreements**
What were the sharpest points of conflict?

**4. What Was Left Unresolved**
What important questions remain?

**5. Recommendations for the User**
3-5 concrete, prioritized action items.
Format: [Priority: High/Medium/Low] Recommendation — reason.

Be direct. Cut anything vague or redundant."""

    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    transcript = history.to_transcript()

    response = client.chat.complete(
        model="mistral-large-latest",
        max_tokens=2500,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"Original topic: {topic_str}\n\nForum transcript:\n\n{transcript}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Start button ─────────────────────────────────────────────────────────────

start_btn = st.button(
    "🚀 Start Forum",
    type="primary",
    disabled=st.session_state.running,
)

if start_btn and topic:
    agents = build_agents()
    if not agents:
        st.error("Select at least one agent.")
    else:
        st.session_state.running = True
        st.session_state.messages = []
        st.session_state.synthesis = ""

        config = ForumConfig(
            max_turns=max_turns,
            max_tokens_per_turn=max_tokens,
            temperature=temperature,
        )

        history = ConversationHistory()
        history.add(role="user", name="User", content=topic)

        progress_bar = st.progress(0, text="Starting forum...")
        message_container = st.container()

        for turn in range(max_turns):
            agent = agents[turn % len(agents)]
            round_label = get_round_label(turn, len(agents))
            round_prompt = get_round_prompt(turn, len(agents), config)
            color = get_color(agent.name)

            progress_bar.progress(
                (turn) / (max_turns + 1),
                text=f"Turn {turn + 1}/{max_turns} [{round_label}] → {agent.name}",
            )

            try:
                response = agent.respond(
                    history=history,
                    topic=topic,
                    round_prompt=round_prompt,
                )
            except Exception as e:
                response = f"[Error: {e}]"

            history.add(role="assistant", name=agent.name, content=response)

            msg = {
                "name": agent.name,
                "content": response,
                "round_label": round_label,
                "turn": turn + 1,
            }
            st.session_state.messages.append(msg)

            with message_container:
                st.markdown(
                    f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
                    f'margin: 8px 0; border-radius: 4px; '
                    f'background-color: rgba(0,0,0,0.02);">'
                    f'<strong style="color: {color};">{agent.name}</strong> '
                    f'<span style="color: #999; font-size: 0.85em;">· {round_label} · Turn {turn + 1}</span>'
                    f'<br><br>{escape_dollars(response)}</div>',
                    unsafe_allow_html=True,
                )

        # Synthesis
        progress_bar.progress(
            max_turns / (max_turns + 1),
            text="Synthesizing discussion...",
        )

        try:
            synthesis = run_synthesis(history, topic, config)
            st.session_state.synthesis = synthesis
        except Exception as e:
            synthesis = f"[Synthesis error: {e}]"
            st.session_state.synthesis = synthesis

        progress_bar.progress(1.0, text="Forum complete!")
        st.session_state.running = False
        st.session_state.history = history

elif start_btn and not topic:
    st.warning("Please enter a topic first.")


# ── Display saved messages (after rerun) ─────────────────────────────────────

if st.session_state.messages and not start_btn:
    for msg in st.session_state.messages:
        color = get_color(msg["name"])
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
            f'margin: 8px 0; border-radius: 4px; '
            f'background-color: rgba(0,0,0,0.02);">'
            f'<strong style="color: {color};">{msg["name"]}</strong> '
            f'<span style="color: #999; font-size: 0.85em;">· {msg["round_label"]} · Turn {msg["turn"]}</span>'
            f'<br><br>{escape_dollars(msg["content"])}</div>',
            unsafe_allow_html=True,
        )


# ── Synthesis display ────────────────────────────────────────────────────────

if st.session_state.synthesis:
    st.divider()
    st.subheader("📋 Forum Synthesis")
    st.markdown(escape_dollars(st.session_state.synthesis))


# ── Download transcript ──────────────────────────────────────────────────────

if st.session_state.messages:
    st.divider()

    transcript_lines = []
    if topic:
        transcript_lines.append(f"Topic: {topic}\n")
    for msg in st.session_state.messages:
        transcript_lines.append(
            f"--- {msg['name']} [{msg['round_label']}] (Turn {msg['turn']}) ---\n"
            f"{msg['content']}\n"
        )
    if st.session_state.synthesis:
        transcript_lines.append("\n=== SYNTHESIS ===\n")
        transcript_lines.append(st.session_state.synthesis)

    transcript_text = "\n".join(transcript_lines)

    st.download_button(
        label="📥 Download Transcript",
        data=transcript_text,
        file_name="forum_transcript.txt",
        mime="text/plain",
    )
