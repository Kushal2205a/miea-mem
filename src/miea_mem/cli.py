"""miea CLI: the shell door into the memory core."""

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
    """miea — graph-based agent memory."""
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
@click.option("--page", default=0, help="Signpost page (TOP_K per page).")
@click.pass_context
def land(ctx: click.Context, ref: str, page: int = 0):
    """Land on a node; print its rideable payload + signpost (use --page to walk it)."""
    click.echo(_mem(ctx.obj["root"]).land(ref, page=page).render())


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
@click.option("--type", "type_", default="fact",
              type=click.Choice(["fact", "preference", "procedure", "event",
                                 "claim", "anchor"]))
@click.option("--under-graph", default=None)
@click.option("--tags", default="", help="Comma-separated category keywords.")
@click.pass_context
def add(ctx: click.Context, label: str, content: str, type_: str,
        under_graph: str | None, tags: str):
    """Create a node (fact/preference/procedure/event/claim)."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    n = _mem(ctx.obj["root"]).create_node(label, content, type_, under_graph)
    if tag_list:
        n.tags = tag_list
        from .model import now_iso
        n.updated_at = now_iso()
        _mem(ctx.obj["root"])._index_node(n)
        _mem(ctx.obj["root"]).store.save_node(n)
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


@cli.command()
@click.argument("source")
@click.argument("target")
@click.pass_context
def placement(ctx: click.Context, source: str, target: str):
    """Where to file a SOURCE→TARGET triple: as deep as it's true, no deeper."""
    import json
    click.echo(json.dumps(
        _mem(ctx.obj["root"]).placement_hint(source, target), indent=2))


@cli.command("verify")
@click.option("--limit", default=None, type=int,
              help="Max claims to check this run.")
@click.option("--verifier", default="null",
              type=click.Choice(["null", "serp"]),
              help="null: mark unverifiable (offline). serp: search-engine "
                   "verdict via AGENT_MEM_SEARCH_CMD.")
@click.pass_context
def verify(ctx: click.Context, limit: int | None, verifier: str):
    """Run the epistemic annotation pass over pending world-claims."""
    from .epistemics import EpistemicPass, NullVerifier, make_serp_verifier
    import json
    import os
    import shlex
    import subprocess

    if verifier == "serp":
        cmd = os.environ.get("AGENT_MEM_SEARCH_CMD")
        if not cmd:
            raise click.ClickException(
                "set AGENT_MEM_SEARCH_CMD='<query placeholder supported>' "
                "to a command that takes the query and prints JSON "
                "[{title: ...}, ...]")

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
    """Audit: which claim nodes lack a provenance edge."""
    import json
    click.echo(json.dumps(
        _mem(ctx.obj["root"]).provenance_report(), indent=2))


@cli.command()
@click.option("--tokens", default=250, help="Rough token budget for the snapshot.")
@click.pass_context
def wakeup(ctx: click.Context, tokens: int):
    """Session-start snapshot: who the user is, hot preferences, recent
    projects. For custom-agent builders: run this at session start and
    inject stdout into context — push-style recall without hooks support."""
    import json

    mem = _mem(ctx.obj["root"])
    snapshot = _wakeup_snapshot(mem, budget=tokens)
    click.echo(json.dumps(snapshot, indent=2))


def _wakeup_snapshot(mem: Memory, budget: int = 250) -> dict:
    """Build the identity/preferences/projects snapshot.

    Selection is pure structure: user-node edges by breadth×recency, then
    top breadth nodes overall. Budget trims in that priority order.
    """
    # find the user node: prefer anchors with provenance-verb in-edges
    # (e.g. [Kushal] receiving user_asserts), else most-connected node
    from collections import Counter as C

    from .epistemics import PROVENANCE_VERBS

    prov_in = C()
    all_in = C()
    for e in mem.edges.values():
        all_in[e.target_id] += 1
        if e.verb in PROVENANCE_VERBS:
            prov_in[e.source_id] += 1  # asserter SENDS user_asserts
    if prov_in:
        # the asserter of most provenance edges is the user
        user_id = max(prov_in, key=lambda k: (prov_in[k], all_in.get(k, 0)))
    elif all_in:
        user_id = max(all_in, key=lambda k: all_in[k])
    else:
        user_id = None

    lines: list[str] = []
    sections: list[tuple[str, list[str]]] = []

    if user_id:
        user = mem.nodes[user_id]
        sections.append(("USER", [f"{user.label} — {user.content}".strip(" —")]))

        # neighbors of the user grouped by verb, ranked by breadth
        dests = mem._all_destinations(user)
        grouped: dict[str, list[str]] = {}
        for d in sorted(dests, key=lambda x: x.score, reverse=True):
            n = mem.nodes.get(d.node_id)
            if not n:
                continue
            verb = d.verb or "related"
            entry = f"{n.label}" + (f" ({n.content[:60]})" if n.content else "")
            grouped.setdefault(verb, []).append(entry)
        for verb, items in grouped.items():
            sections.append((verb.upper(), items))

        # recent projects/events by updatedAt
        dated = sorted(
            (n for n in mem.nodes.values()
             if n.id != user_id and n.type in ("fact", "event")),
            key=lambda n: n.updated_at or "", reverse=True)[:3]
        recent = [
            f"{n.label} ({(n.updated_at or '')[:10]})" for n in dated
            if all(n.label not in it for _, items in sections[1:] for it in items)
        ]
        if recent:
            sections.append(("RECENT", recent))

    # trim to rough token budget (~4 chars/token), priority = section order
    used = 0
    out_sections = []
    for title, items in sections:
        kept = []
        for it in items:
            cost = len(it) // 4 + 6
            if used + cost > budget:
                break
            kept.append(it)
            used += cost
        if kept:
            out_sections.append((title, kept))
        if used >= budget:
            break

    text_parts = []
    for title, items in out_sections:
        text_parts.append(f"## {title}\n" + "\n".join(f"- {i}" for i in items))
    text = "\n\n".join(text_parts)

    return {
        "text": text,
        "approx_tokens": used,
        "budget": budget,
        "hint": "Inject this at session start; refresh per prompt via search.",
    }


def main() -> None:
    from .setup import setup_cmd  # local import avoids setup↔cli cycle

    cli.add_command(setup_cmd)
    cli()


if __name__ == "__main__":
    main()
