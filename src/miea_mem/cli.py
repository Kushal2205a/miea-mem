# Command line interface. Thin wrappers over the core operations, one
# command each.

from __future__ import annotations

import json

import click

from .core import Memory
from .store import Store


def _mem(root: str) -> Memory:
    try:
        return Memory(root)
    except FileNotFoundError as e:
        raise click.ClickException(f"not a workspace: {root} ({e})") from e


@click.group()
@click.option("--root", default=".", envvar="MIEA_ROOT",
              help="Workspace directory.")
@click.pass_context
def cli(ctx: click.Context, root: str):
    """miea: graph-based agent memory."""
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
    """Find entry points by keyword, ranked."""
    for node, score in _mem(ctx.obj["root"]).search(query, limit=limit):
        click.echo(f"({score:.2f}) [{node.label}] {node.type} -- {node.id}")


@cli.command()
@click.argument("ref")
@click.option("--page", default=0, help="Signpost page, seven destinations each.")
@click.pass_context
def land(ctx: click.Context, ref: str, page: int):
    """Read a node with its content and signpost."""
    click.echo(_mem(ctx.obj["root"]).land(ref, page=page).render())


@cli.command()
@click.argument("ref")
@click.argument("destination")
@click.pass_context
def steer(ctx: click.Context, ref: str, destination: str):
    """Move from ref to one of its signpost destinations."""
    click.echo(_mem(ctx.obj["root"]).steer(ref, destination).render())


@cli.command("query-scoped")
@click.argument("graph_ref")
@click.argument("query")
@click.pass_context
def query_scoped(ctx: click.Context, graph_ref: str, query: str):
    """Deep dive inside one subtree."""
    for p in _mem(ctx.obj["root"]).query_scoped(graph_ref, query):
        click.echo(p.render())
        click.echo("---")


@cli.command("lca")
@click.argument("refs", nargs=-1, required=True)
@click.pass_context
def lca(ctx: click.Context, refs: tuple[str]):
    """Lowest common ancestor of two or more nodes."""
    click.echo(json.dumps(_mem(ctx.obj["root"]).lca_context(list(refs)), indent=2))


@cli.command()
@click.argument("source")
@click.argument("verb")
@click.argument("target")
@click.pass_context
def link(ctx: click.Context, source: str, verb: str, target: str):
    """Add a named edge SOURCE --VERB--> TARGET, creating missing nodes."""
    e = _mem(ctx.obj["root"]).write_triple(source, verb, target,
                                           create_missing=True)
    click.echo(f"linked: [{source}] --{e.verb}--> [{target}]")


@cli.command()
@click.argument("label")
@click.option("--content", default="")
@click.option("--type", "type_",
              type=click.Choice(["fact", "preference", "procedure", "event",
                                 "claim", "anchor"]),
              default="fact")
@click.option("--under-graph", default=None)
@click.option("--tags", default="", help="Comma-separated category keywords.")
@click.pass_context
def add(ctx: click.Context, label: str, content: str, type_: str,
        under_graph: str | None, tags: str):
    """Create a node."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    n = _mem(ctx.obj["root"]).create_node(label, content, type_, under_graph)
    if tag_list:
        from .model import now_iso
        n.tags = tag_list
        n.updated_at = now_iso()
        m = _mem(ctx.obj["root"])
        m._index_node(n)
        m.store.save_node(n)
    click.echo(f"added [{n.label}] {n.id}")


@cli.command()
@click.argument("ref")
@click.confirmation_option(prompt="Really delete?")
@click.pass_context
def forget(ctx: click.Context, ref: str):
    """Delete a node and its subtree."""
    removed = _mem(ctx.obj["root"]).forget(ref)
    click.echo(f"forgot {removed} node(s)")


@cli.command()
@click.argument("ref")
@click.pass_context
def neighbors(ctx: click.Context, ref: str):
    """Show every edge touching a node."""
    for n in _mem(ctx.obj["root"]).neighbors(ref):
        arrow = f"--{n['verb']}-->" if n["direction"] == "out" else f"<--{n['verb']}--"
        click.echo(f"{arrow} [{n['other']}]")


@cli.command()
@click.argument("source")
@click.argument("target")
@click.pass_context
def placement(ctx: click.Context, source: str, target: str):
    """Suggest where to file a SOURCE to TARGET triple."""
    click.echo(json.dumps(
        _mem(ctx.obj["root"]).placement_hint(source, target), indent=2))


@cli.command("verify")
@click.option("--limit", default=None, type=int,
              help="Maximum claims to check this run.")
@click.option("--verifier", default="null",
              type=click.Choice(["null", "serp"]),
              help="null marks everything unverifiable. serp uses the "
                   "command in MIEA_SEARCH_CMD.")
@click.pass_context
def verify(ctx: click.Context, limit: int | None, verifier: str):
    """Run the epistemic annotation pass over pending world claims."""
    import os
    import shlex
    import subprocess

    from .epistemics import EpistemicPass, NullVerifier, make_serp_verifier

    if verifier == "serp":
        cmd = os.environ.get("MIEA_SEARCH_CMD")
        if not cmd:
            raise click.ClickException(
                "set MIEA_SEARCH_CMD to a command that takes a query and "
                'prints JSON like [{"title": "..."}]. Use {q} as the '
                "query placeholder.")

        def search_fn(q: str) -> list[dict]:
            argv = [part.replace("{q}", q) for part in shlex.split(cmd)]
            out = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=30)
            return json.loads(out.stdout)

        v = make_serp_verifier(search_fn)
    else:
        v = NullVerifier()

    report = EpistemicPass(_mem(ctx.obj["root"]), v).run(limit=limit)
    click.echo(json.dumps(report, indent=2) or "[]")


@cli.command("provenance")
@click.pass_context
def provenance(ctx: click.Context):
    """List claim nodes without a provenance edge."""
    click.echo(json.dumps(
        _mem(ctx.obj["root"]).provenance_report(), indent=2))


@cli.command()
@click.option("--tokens", default=250,
              help="Approximate token budget for the snapshot.")
@click.pass_context
def wakeup(ctx: click.Context, tokens: int):
    """Print a session-start snapshot of who the user is and what matters.
    Custom agents can inject this output at session start."""
    mem = _mem(ctx.obj["root"])
    click.echo(json.dumps(_wakeup_snapshot(mem, tokens), indent=2))


def _wakeup_snapshot(mem: Memory, budget: int) -> dict:
    # Finds the user as the source of most provenance edges, then lists
    # their neighbors grouped by verb, ranked, then recent nodes. Trims to
    # roughly four characters per token against the budget.
    from collections import Counter

    from .epistemics import PROVENANCE_VERBS

    prov_in = Counter()
    all_in = Counter()
    for e in mem.edges.values():
        all_in[e.target_id] += 1
        if e.verb in PROVENANCE_VERBS:
            prov_in[e.source_id] += 1

    user_id = None
    if prov_in:
        user_id = max(prov_in, key=lambda k: (prov_in[k], all_in.get(k, 0)))
    elif all_in:
        user_id = max(all_in, key=lambda k: all_in[k])

    sections: list[tuple[str, list[str]]] = []
    if user_id:
        user = mem.nodes[user_id]
        sections.append(("USER",
                         [f"{user.label} {user.content}".strip()]))

        grouped: dict[str, list[str]] = {}
        for d in sorted(mem._all_destinations(user),
                        key=lambda x: x.score, reverse=True):
            n = mem.nodes.get(d.node_id)
            if not n:
                continue
            entry = n.label + (f" ({n.content[:60]})" if n.content else "")
            grouped.setdefault(d.verb or "related", []).append(entry)
        for verb, items in grouped.items():
            sections.append((verb.upper(), items))

        dated = sorted(
            (n for n in mem.nodes.values()
             if n.id != user_id and n.type in ("fact", "event")),
            key=lambda n: n.updated_at or "", reverse=True)[:3]
        listed = {it for _, items in sections[1:] for it in items}
        recent = [f"{n.label} ({(n.updated_at or '')[:10]})"
                  for n in dated if n.label not in listed]
        if recent:
            sections.append(("RECENT", recent))

    used = 0
    kept_sections: list[tuple[str, list[str]]] = []
    for title, items in sections:
        kept = []
        for item in items:
            cost = len(item) // 4 + 6
            if used + cost > budget:
                break
            kept.append(item)
            used += cost
        if kept:
            kept_sections.append((title, kept))
        if used >= budget:
            break

    parts = [f"## {title}\n" + "\n".join(f"- {i}" for i in items)
             for title, items in kept_sections]
    return {
        "text": "\n\n".join(parts),
        "approx_tokens": used,
        "budget": budget,
        "hint": "Inject this at session start; refresh per prompt via search.",
    }


def main() -> None:
    from .setup import setup_cmd  # local import avoids setup and cli cycle

    cli.add_command(setup_cmd)
    cli()
