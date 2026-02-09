"""
ClawAI CLI

Design principles (ClawAI style):
- Strong modular boundaries
- Predictable lifecycle (start/stop)
- Centralized path & config handling
- Clear async orchestration
- Clean CLI UX
"""

from __future__ import annotations

import asyncio
import atexit
import os
import select
import signal
import sys
from pathlib import Path
from typing import Final

import typer
from rich.console import Console
from rich.table import Table

from clawai import __version__, __logo__


# ============================================================================
# CLI App
# ============================================================================

APP_NAME: Final[str] = "clawai"

app = typer.Typer(
    name=APP_NAME,
    help=f"{__logo__} ClawAI - Personal AI Agent Runtime",
    no_args_is_help=True,
)

console = Console()

# ============================================================================
# Readline / Terminal Helpers
# ============================================================================

_READLINE = None
_HISTORY_FILE: Path | None = None
_HISTORY_HOOK_REGISTERED = False
_USING_LIBEDIT = False
_SAVED_TERM_ATTRS = None


# ------------------------------
# Terminal utils
# ------------------------------

def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model is generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _save_history() -> None:
    if _READLINE and _HISTORY_FILE:
        try:
            _READLINE.write_history_file(str(_HISTORY_FILE))
        except Exception:
            pass


def _restore_terminal() -> None:
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


# ------------------------------
# Readline setup
# ------------------------------

def _enable_line_editing() -> None:
    """Enable readline with history and arrow-key support."""
    global _READLINE, _HISTORY_FILE, _HISTORY_HOOK_REGISTERED, _USING_LIBEDIT, _SAVED_TERM_ATTRS

    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".clawai" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_FILE = history_file

    try:
        import readline
    except ImportError:
        return

    _READLINE = readline
    _USING_LIBEDIT = "libedit" in (readline.__doc__ or "").lower()

    try:
        if _USING_LIBEDIT:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set editing-mode emacs")
    except Exception:
        pass

    try:
        readline.read_history_file(str(history_file))
    except Exception:
        pass

    if not _HISTORY_HOOK_REGISTERED:
        atexit.register(_save_history)
        _HISTORY_HOOK_REGISTERED = True


# ------------------------------
# Prompt
# ------------------------------

def _prompt_text() -> str:
    if _READLINE is None:
        return "You: "
    if _USING_LIBEDIT:
        return "\033[1;34mYou:\033[0m "
    return "\001\033[1;34m\002You:\001\033[0m\002 "


async def _read_interactive_input_async() -> str:
    try:
        return await asyncio.to_thread(input, _prompt_text())
    except EOFError as exc:
        raise KeyboardInterrupt from exc


# ============================================================================
# Version
# ============================================================================

def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} clawai v{__version__}")
        raise typer.Exit()


@app.callback()

def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """ClawAI - Personal AI Agent Runtime."""
    pass


# ============================================================================
# Onboard
# ============================================================================


@app.command()

def onboard():
    """Initialize ClawAI configuration and workspace."""
    from clawai.config.loader import get_config_path, save_config
    from clawai.config.schema import Config
    from clawai.runtime.paths import RUNTIME_PATHS

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit()

    config = Config()
    save_config(config)

    console.print(f"[green]✓[/green] Created config at {config_path}")
    console.print(f"[green]✓[/green] Workspace ready at {RUNTIME_PATHS.workspace}")

    _create_workspace_templates(RUNTIME_PATHS.workspace)

    console.print(f"\n{__logo__} ClawAI is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.clawai/config.json[/cyan]")
    console.print("  2. Chat: [cyan]clawai agent -m \"Hello!\"[/cyan]")


# ============================================================================
# Workspace bootstrap
# ============================================================================


def _create_workspace_templates(workspace: Path):
    templates = {
        "AGENTS.md": """# Agent Instructions

- Be concise, accurate, and transparent
- Explain intent before actions
- Ask for clarification when ambiguous
""",
        "SOUL.md": """# Soul

I am ClawAI, a lightweight autonomous agent.

Values:
- Accuracy
- Safety
- Transparency
""",
        "USER.md": """# User

User preferences & habits.
""",
    }

    for name, content in templates.items():
        path = workspace / name
        if not path.exists():
            path.write_text(content)
            console.print(f"  [dim]Created {name}[/dim]")

    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)

    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("# Long-term Memory\n")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")


# ============================================================================
# Provider
# ============================================================================


def _make_provider(config):
    from clawai.llm.litellm import LiteLLMProvider

    p = config.get_provider()
    model = config.agents.defaults.model

    if not (p and p.api_key) and not model.startswith("bedrock/"):
        console.print("[red]Error: No API key configured.[/red]")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )


# ============================================================================
# Gateway
# ============================================================================


@app.command()

def gateway(
    port: int = typer.Option(0, "--port", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start ClawAI gateway runtime."""

    from clawai.config.loader import load_config, get_data_dir
    from clawai.bus.queue import MessageBus
    from clawai.agent.loop import AgentLoop
    from clawai.channels.manager import ChannelManager
    from clawai.session.manager import SessionManager
    from clawai.scheduler.service import CronService
    from clawai.scheduler.types import CronJob
    from clawai.heartbeat.service import HeartbeatService

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    console.print(f"{__logo__} Starting ClawAI gateway on port {port}...")

    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)

    session_manager = SessionManager(config.workspace_path)

    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
    )

    async def on_cron_job(job: CronJob) -> str | None:
        return await agent.process_direct(job.payload.message, session_key=f"cron:{job.id}")

    cron.on_job = on_cron_job

    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        on_heartbeat=lambda p: agent.process_direct(p, session_key="heartbeat"),
        interval_s=30 * 60,
        enabled=True,
    )

    channels = ChannelManager(config, bus, session_manager=session_manager)

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(agent.run(), channels.start_all())
        except KeyboardInterrupt:
            console.print("\nShutting down...")
            await heartbeat.stop()
            await cron.stop()
            await agent.stop()
            await channels.stop_all()

    asyncio.run(run())


# ============================================================================
# Agent
# ============================================================================


@app.command()

def agent(
    message: str = typer.Option(None, "--message", "-m"),
    session_id: str = typer.Option("cli:default", "--session", "-s"),
):
    """Chat with ClawAI agent."""

    from clawai.config.loader import load_config
    from clawai.bus.queue import MessageBus
    from clawai.agent.loop import AgentLoop

    config = load_config()

    bus = MessageBus()
    provider = _make_provider(config)

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
    )

    if message:
        async def run_once():
            response = await agent_loop.process_direct(message, session_id)
            console.print(f"\n{__logo__} {response}")

        asyncio.run(run_once())
        return

    _enable_line_editing()
    console.print(f"{__logo__} Interactive mode (Ctrl+C to exit)\n")

    def _exit_on_sigint(signum, frame):
        _save_history()
        _restore_terminal()
        console.print("\nGoodbye!")
        os._exit(0)

    signal.signal(signal.SIGINT, _exit_on_sigint)

    async def run_interactive():
        while True:
            try:
                _flush_pending_tty_input()
                user_input = await _read_interactive_input_async()
                if not user_input.strip():
                    continue

                response = await agent_loop.process_direct(user_input, session_id)
                console.print(f"\n{__logo__} {response}\n")
            except KeyboardInterrupt:
                break

    asyncio.run(run_interactive())


# ============================================================================
# Status
# ============================================================================


@app.command()

def status():
    """Show ClawAI runtime status."""

    from clawai.config.loader import load_config, get_config_path
    from clawai.llm.registry import PROVIDERS

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} ClawAI Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    console.print(f"Model: {config.agents.defaults.model}")

    for spec in PROVIDERS:
        p = getattr(config.providers, spec.name, None)
        if not p:
            continue
        if spec.is_local:
            console.print(f"{spec.label}: {p.api_base or '[dim]not set[/dim]'}")
        else:
            console.print(f"{spec.label}: {'[green]✓[/green]' if p.api_key else '[dim]not set[/dim]'}")


if __name__ == "__main__":
    app()
