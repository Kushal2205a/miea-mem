"""Setup wizard: `agent-mem setup`.

Prompted flow — workspace path, user name, agent detection, paste-ready MCP
config. Never overwrites existing configs; merging is left to the user.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .store import Store


def _mcp_block(server_bin: str, root: str, tier: str = "write") -> str:
    return json.dumps({
        "kushal-memory": {
            "type": "local",
            "command": [server_bin, "--root", root, "--tier", tier],
        }
    }, indent=2)


def _detect_agents() -> list[tuple[str, Path]]:
    """Known-agent config paths that exist on this machine."""
    home = Path.home()
    known = [
        ("OpenCode", home / ".config/opencode/opencode.jsonc"),
        ("Claude Code", home / ".claude.json"),
        ("Cursor", home / ".cursor/mcp.json"),
        ("Gemini CLI", home / ".gemini/settings.json"),
    ]
    return [(name, p) for name, p in known if p.exists()]


def _find_server_bin() -> str:
    """Path to the installed agent-mem-server executable."""
    from shutil import which

    found = which("agent-mem-server")
    if found:
        return str(Path(found).resolve())
    # uv tool default location
    fallback = Path.home() / ".local/bin/agent-mem-server"
    if fallback.exists():
        return str(fallback)
    return "agent-mem-server"


@click.command("setup")
@click.option("--root", default=None, help="Skip the location prompt.")
@click.option("--name", default=None, help="Skip the name prompt.")
def setup_cmd(root: str | None, name: str | None):
    """Interactive setup: create a workspace and print MCP config."""
    click.echo("agent-mem setup\n" + "=" * 40)

    # 1. workspace location
    if not root:
        default_root = Path.home() / "Documents" / "kushal-memory"
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
        click.echo(f"✓ Workspace created at {root_path}")
    else:
        manifest = store.load_manifest()
        click.echo(f"• Using existing workspace at {root_path}")

    # 2. user name
    if not name:
        name = click.prompt("Your name")
    assert name is not None
    name = name.strip()

    # record the user node so wakeup()/provenance have an anchor from day one
    existing = [n for n in mem_all_nodes(store) if n.label.lower() == name.lower()]
    if not existing:
        from .model import Node, new_id

        user = Node(id=new_id(), label=name, type="anchor",
                    content=f"{name}'s memory workspace")
        store.save_node(user)
        g = store.load_graph(manifest.root_graph_id)
        assert g is not None
        g.node_ids.add(user.id)
        store.save_graph(g)
        click.echo(f"✓ User node [{name}] created")

    # 3. semantic search availability (informational only)
    try:
        from .semantic import try_load_embedder

        if try_load_embedder() is not None:
            click.echo("✓ Semantic search available")
        else:
            click.echo(
                "• Semantic search: model not installed — keyword search "
                "still works. Enable later with:\n"
                "  uv tool install --force --with sentence-transformers "
                "<repo>"
            )
    except Exception:
        pass

    # 4. detected agents
    server_bin = _find_server_bin()
    block = _mcp_block(server_bin, str(root_path))
    detected = _detect_agents()
    if detected:
        click.echo("\nDetected agents:")
        for agent_name, cfg in detected:
            click.echo(f"  • {agent_name}: {cfg}")
        click.echo(
            "\nMerge this block into the config(s) you use "
            "(not done automatically):")

    click.echo("\nMCP config — paste into your agent's config file:\n")
    click.echo(block)
    click.echo(
        f"\nDone. Point any MCP client at {root_path}.\n"
        'Test: ask your agent "What do you know about me?"')


def mem_all_nodes(store: Store):
    return store.all_nodes()
