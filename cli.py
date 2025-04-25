import os
import sys

import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(add_completion=False)
console = Console()

API_URL = os.getenv("MEMIR_API_URL", "http://127.0.0.1:8000")

@app.command()
def chat(message: str = typer.Argument(..., help="Message to send to M.E.M.I.R. chatbot")):
    """Send a message to the M.E.M.I.R. chatbot and print the reply."""
    _chat_and_print(message)

def _chat_and_print(message, history=None):
    try:
        payload = {"message": message}
        if history is not None:
            payload["history"] = history
        resp = requests.post(f"{API_URL}/chat/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        console.print(Panel(data["response"], title="[bold green]M.E.M.I.R.[/bold green]", expand=False))
        return data.get("history", [])
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return history or []

def interactive_shell():
    console.print("[bold cyan]Welcome to the M.E.M.I.R. CLI shell![/bold cyan]")
    console.print("Type [bold yellow]chat <message>[/bold yellow] or just enter a message to chat. Type [bold]exit[/bold] or [bold]quit[/bold] to leave.")
    history = []
    while True:
        try:
            user_input = Prompt.ask("[bold magenta]memir>[/bold magenta]")
            if user_input.strip().lower() in {"exit", "quit"}:
                console.print("[bold green]Goodbye![/bold green]")
                break
            elif user_input.strip().startswith("chat "):
                message = user_input.strip()[5:].strip()
                if message:
                    history = _chat_and_print(message, history)
            elif user_input.strip():
                # Default: treat as chat message
                history = _chat_and_print(user_input.strip(), history)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye![/bold green]")
            break

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_shell()
    else:
        app()
