"""agent_mem CLI: the shell door into the memory core."""

from __future__ import annotations

import json
import sys

import click

from .core import Memory
from .store import Store


def _mem(root: str) -> Memory:
    try:
        return Memory(root)
    except FileNotFoundError as e:
        raise click.ClickException(
            f"not a workspace: {root} ({e})"
        ) from e


@click.group()
@click.option("--root", default=".", envvar="AGENT_MEM_ROOT",
              help="Workspace directory.")
@click.pass_context
def cli(ctx: click.Context, root: str):
    """agent_mem — graph-based agent memory."""
    ctx.ensure_object(dict)
    ctx.obj["root"] = root


@cli.command()
@click.argument("name", default="Memory")
@click.option("--root", default=".")
def init(name: str, root: str):
    """Create a new memory workspace."""
    store = Store(root)
    if store.exists():
        raise click.ClickException(f"{root} is already a workspace")
    m = store.init_workspace(name)
    click.echo(f"initialized {name!r} at {root} (root graph {m.root_graph_id})")


@cli.command("search")
@click.argument("query")
@click.option("--limit", default=5)
@click.pass_context
def search(ctx: click.Context, query: str, limit: int):
    """Find entry points (ranked)."""
    mem = _mem(ctx.obj["root"])
    for node, score in mem.search(query, limit=limit):
        click.echo(f"({score:.2f}) [{node.label}] {node.type} — {node.id}")


@cli.command()
@click.argument("ref")
@click.pass_context
def land(ctx: click.Context, ref: str):
    """Land on a node; print its rideable payload + signpost."""
    click.echo(_mem(ctx.obj["root"]).land(ref).render())


@cli.command()
@click.argument("ref")
@click.argument("destination")
@click.pass_context
def steer(ctx: click.Context, ref: str, destination: str):
    """Ride one branch toward a signpost destination."""
    click.echo(_mem(ctx.obj["root"]).steer(ref, destination).render())


@cli.command("query-scoped")
@click.argument("graph_ref")
@click.argument("query")
@click.pass_context
def query_scoped(ctx: click.Context, graph_ref: str, query: str):
    """Mediated deep dive within one subtree."""
    for p in _mem(ctx.obj["root"]).query_scoped(graph_ref, query):
        click.echo(p.render())
        click.echo("---")


@cli.command("lca")
@click.argument("refs", nargs=-1, required=True)
@click.pass_context
def lca(ctx: click.Context, refs: tuple[str]):
    """Lowest common ancestor of nodes."""
    click.echo(json.dumps(_mem(ctx.obj["root"]).lca_context(list(refs)), indent=2))


@cli.command()
@click.argument("source")
@click.argument("verb")
@click.argument("target")
@click.pass_context
def link(ctx: click.Context, source: str, verb: str, target: str):
    """Add a named edge: SOURCE --VERB--> TARGET (creates missing nodes)."""
    e = _mem(ctx.obj["root"]).write_triple(source, verb, target, create_missing=True)
    click.echo(f"linked: [{source}] --{e.verb}--> [{target}]")


@cli.command()
@click.argument("label")
@click.option("--content", default="")
@click.option("--type", "type_", default="fact")
@click.option("--under-graph", default=None)
@click.pass_context
def add(ctx: click.Context, label: str, content: str, type_: str,
        under_graph: str | None):
    """Create a node."""
    n = _mem(ctx.obj["root"]).create_node(label, content, type_, under_graph)
    click.echo(f"added [{n.label}] {n.id}")


@cli.command()
@click.argument("ref")
@click.confirmation_option(prompt="Really delete?")
@click.pass_context
def forget(ctx: click.Context, ref: str):
    """Explicit deletion of a node (and its subtree)."""
    removed = _mem(ctx.obj["root"]).forget(ref)
    click.echo(f"forgot {removed} node(s)")


@cli.command()
@click.argument("ref")
@click.pass_context
def neighbors(ctx: click.Context, ref: str):
    """Full local neighborhood (write-tier inspection)."""
    for n in _mem(ctx.obj["root"]).neighbors(ref):
        arrow = f"--{n['verb']}-->" if n["direction"] == "out" else f"<--{n['verb']}--"
        click.echo(f"{arrow} [{n['other']}]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
