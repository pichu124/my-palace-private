from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

AGENT_COLORS = {
    "Llama3":   "green",
    "Gemini":   "blue",
    "Mistral":  "magenta",
    "DeepSeek": "yellow",
    "User":     "cyan",
}

def print_message(agent_name: str, round_label: str, content: str) -> None:
    color = AGENT_COLORS.get(agent_name, "white")
    console.print(Panel(
        content,
        title=f"[bold {color}]{agent_name}[/bold {color}] [dim]· {round_label}[/dim]",
        border_style=color,
        padding=(1, 2),
    ))

def print_moderator(message: str) -> None:
    console.print(f"[bold dim]⚙  {message}[/bold dim]")

def print_synthesis(synthesis: str) -> None:
    console.print(Rule("[bold magenta]Forum Synthesis[/bold magenta]"))
    console.print(synthesis)
    console.print(Rule())