# Setup wizard. Asks for a workspace path and the user name, creates the
# workspace with a user anchor node, initializes git when available, and
# prints an MCP config block for pasting into any agent.

from __future__ import annotations

import json
from pathlib import Path

import click

from .store import Store


def _mcp_block(server_bin: str, root: str, tier: str = "write") -> str:
    return json.dumps({
        "miea-memory": {
            "type": "local",
            "command": [server_bin, "--root", root, "--tier", tier],
        }
    }, indent=2)


def _detect_agents() -> list[tuple[str, Path]]:
    # Known agent config paths that exist on this machine.
    home = Path.home()
    known = [
        ("OpenCode", home / ".config/opencode/opencode.jsonc"),
        ("Claude Code", home / ".claude.json"),
        ("Cursor", home / ".cursor/mcp.json"),
        ("Gemini CLI", home / ".gemini/settings.json"),
    ]
    return [(name, p) for name, p in known if p.exists()]


def _find_server_bin() -> str:
    from shutil import which

    found = which("miea-server")
    if found:
        return str(Path(found).resolve())
    fallback = Path.home() / ".local/bin/miea-server"
    if fallback.exists():
        return str(fallback)
    return "miea-server"


@click.command("setup")
@click.option("--root", default=None, help="Skip the location prompt.")
@click.option("--name", default=None, help="Skip the name prompt.")
def setup_cmd(root: str | None, name: str | None):
    """Interactive setup: create a workspace and print MCP config."""
    click.echo("miea setup\n" + "=" * 40)

    # Workspace location.
    if not root:
        default_root = Path.home() / "Documents" / "my-memory"
        root = click.prompt(
            "Workspace location",
            default=str(default_root),
            type=click.Path(),
        )
    assert root is not None
    root_path = Path(root).expanduser().resolve()

    store = Store(root_path)
    fresh = not store.exists()
    if fresh:
        manifest = store.init_workspace("Memory")
        click.echo(f"workspace created at {root_path}")
    else:
        manifest = store.load_manifest()
        click.echo(f"using existing workspace at {root_path}")

    # User name.
    if not name:
        name = click.prompt("Your name")
    assert name is not None
    name = name.strip()

    # User anchor node so wakeup and provenance work from day one.
    existing = [n for n in store.all_nodes()
                if n.label.lower() == name.lower()]
    if not existing:
        from .model import Node, new_id

        user = Node(id=new_id(), label=name, type="anchor",
                    content=f"{name}'s memory workspace")
        store.save_node(user)
        g = store.load_graph(manifest.root_graph_id)
        assert g is not None
        g.node_ids.add(user.id)
        store.save_graph(g)
        click.echo(f"user node [{name}] created")

    # Git tracking. Automatic when git exists, silent skip otherwise.
    import shutil
    import subprocess

    if shutil.which("git") is None:
        click.echo("git not found, skipping version history")
    elif not (root_path / ".git").exists():
        try:
            subprocess.run(["git", "init", "-q"], cwd=root_path, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root_path, check=True)
            subprocess.run(
                ["git", "commit", "-q",
                 "-m", f"seed: {name}'s memory workspace"],
                cwd=root_path, check=True,
                env={"GIT_AUTHOR_NAME": name or "miea",
                     "GIT_AUTHOR_EMAIL": "miea@localhost",
                     "GIT_COMMITTER_NAME": name or "miea",
                     "GIT_COMMITTER_EMAIL": "miea@localhost"})
            click.echo("git history initialized")
        except subprocess.CalledProcessError:
            click.echo("git init failed, continuing without history")
    else:
        click.echo("git already tracks this workspace")

    # Semantic search availability, informational only.
    try:
        from .semantic import try_load_embedder

        if try_load_embedder() is not None:
            click.echo("semantic search available")
        else:
            click.echo(
                "semantic search off, model not installed. Keyword search "
                "still works. Enable later with:\n"
                "  uv tool install --force --with sentence-transformers "
                "<repo>")
    except Exception:
        pass

    # Detected agents and the paste block.
    server_bin = _find_server_bin()
    block = _mcp_block(server_bin, str(root_path))
    detected = _detect_agents()
    if detected:
        click.echo("\nDetected agents:")
        for agent_name, cfg in detected:
            click.echo(f"  {agent_name}: {cfg}")
        click.echo("\nMerge this block into the config you use:")

    click.echo("\nMCP config:\n")
    click.echo(block)
    click.echo(
        f"\nDone. Point any MCP client at {root_path}.\n"
        "The workspace is empty. Memory can only recall what was stored,\n"
        'so seed it by telling your agent things: "remember that I love X".')
