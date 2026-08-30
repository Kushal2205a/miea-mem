# Benchmark harness v2. Builds synthetic workspaces with realistic
# topologies, times public operations with proper sample counts, and
# reports p50/p95/p99 plus resident memory.

import argparse
import gc
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miea_mem.core import Memory

# Verb mix weighted like real agent sessions: lots of preference and
# project edges, rare contradictions.
VERB_WEIGHTS = [
    ("likes", 15), ("uses", 14), ("works_on", 13), ("knows_about", 12),
    ("prefers", 10), ("studied_at", 8), ("built", 8), ("supports", 6),
    ("relates_to", 5), ("contradicts", 2), ("supersedes", 2),
]
VERBS = [v for v, w in VERB_WEIGHTS for _ in range(w)]

FANOUT_CAP = 9          # matches split_if_overloaded default
CONTENT_WORDS = 12      # nodes carry real-ish content length


def weighted_verb(rng: random.Random) -> str:
    return rng.choice(VERBS)


def build_workspace(root: Path, n_nodes: int, seed: int = 42) -> None:
    # Topology: an anchor, then a balanced tree of group anchors created by
    # actually calling split_if_overloaded as the system would, so fanout
    # stays near FANOUT_CAP everywhere. Content is filler but realistic.
    rng = random.Random(seed)
    from miea_mem.store import Store

    Store(root).init_workspace("Bench")
    mem = Memory(str(root), embedder=None)
    user = mem.create_node("User", type="anchor",
                           content="workspace owner")

    def content(i: int) -> str:
        words = " ".join(rng.choice(
            ["database", "network", "compiler", "biryani", "memory",
             "kernel", "agent", "graph", "cache", "runtime"]) for _ in range(CONTENT_WORDS))
        return f"note {i} {words}"

    # create in verb-batches so signatures cluster, mirroring how a real
    # session writes several likes before moving to another relation
    batch = 50
    for start in range(0, n_nodes, batch):
        chunk = min(batch, n_nodes - start)
        verb = weighted_verb(rng)
        for i in range(start, start + chunk):
            label = f"topic_{i}"
            mem.create_node(label, content=content(i))
        # link this batch from whichever anchor currently holds them
        for i in range(start, start + chunk):
            mem.write_triple("User", verb if i == start else weighted_verb(rng),
                             f"topic_{i}")
        # inter-topic mesh: some topics reference their neighbors
        for i in range(start + 1, start + chunk, 7):
            mem.write_triple(f"topic_{i-1}", "relates_to", f"topic_{i}")

        # keep fanout honest: split when the anchor overflows
        while mem.fanout(user.id) > FANOUT_CAP * 4:
            groups = mem.split_if_overloaded(user.id)
            if not groups:
                break


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed(fn, iterations: int) -> dict:
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()

    def pct(p):
        idx = min(int(len(samples) * p / 100), len(samples) - 1)
        return round(samples[idx], 2)

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}


def pick_leaf(mem: Memory, rng: random.Random) -> str:
    # a random non-anchor node
    for _ in range(20):
        nid = rng.choice(list(mem.nodes.keys()))
        if mem.nodes[nid].type != "anchor":
            return mem.nodes[nid].label
    return next(n.label for n in mem.nodes.values() if n.type != "anchor")


def run_size(n_nodes: int, iterations_scale: float, seed: int) -> dict:
    cache = Path(tempfile.gettempdir()) / f"miea_bench_v2_{n_nodes}"
    fresh = not (cache / "manifest.json").exists()
    if fresh:
        print(f"building {n_nodes}-node workspace...", file=sys.stderr)
        cache.mkdir(parents=True, exist_ok=True)
        build_workspace(cache, n_nodes, seed)
    rng = random.Random(seed)

    results = {"nodes": n_nodes}

    # true cold load: fresh Memory instance, files not yet touched by this
    # process (page cache may still hold them; noted in output)
    gc.collect()
    t0 = time.perf_counter()
    mem = Memory(str(cache), embedder=None)
    results["cold_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    results["rss_mb"] = round(rss_mb(), 1)

    anchor_id = mem._resolve("User").id

    # realistic recall queries: category words, zero token overlap with labels
    recall_queries = ["what food does the user enjoy",
                      "database related memories",
                      "notes about runtime and kernel work",
                      "anything about compilers"]

    results["search_keyword"] = timed(
        lambda: mem.search(rng.choice(recall_queries)),
        max(30, int(150 * iterations_scale)))

    leaf = pick_leaf(mem, rng)
    dests = [d.label for d in mem._all_destinations(mem._resolve("User"))[:5]]

    results["land_anchor"] = timed(lambda: mem.land("User"),
                                   max(40, int(200 * iterations_scale)))
    results["land_leaf"] = timed(lambda: mem.land(leaf),
                                 max(40, int(200 * iterations_scale)))
    results["steer"] = timed(
        lambda: mem.steer("User", rng.choice(dests) if dests else leaf),
        max(40, int(200 * iterations_scale)))

    # direction picking over the divergence map: hybrid match across
    # branch entries, keyword-only here (embedder=None)
    results["route"] = timed(
        lambda: mem.route("User", rng.choice(recall_queries)),
        max(40, int(200 * iterations_scale)))

    # one-pass descent into a branch, deep rides on to its cue leaf
    branches = [e.label for e in mem.nodes[anchor_id].divergence_map]
    target = rng.choice(branches) if branches else leaf
    results["slide"] = timed(
        lambda: mem.slide("User", target, deep=True),
        max(40, int(200 * iterations_scale)))

    counter = [n_nodes * 10]
    def do_write():
        counter[0] += 1
        i = counter[0]
        mem.write_triple("User", weighted_verb(rng), f"fresh_{i}",
                         create_missing=True)
    results["write_new"] = timed(do_write, max(25, int(100 * iterations_scale)))

    existing_leaf = pick_leaf(mem, rng)
    def do_dedup_write():
        mem.write_triple("User", "likes", existing_leaf)
    results["write_duplicate"] = timed(do_dedup_write,
                                       max(25, int(100 * iterations_scale)))

    results["fanout"] = timed(lambda: mem.fanout(anchor_id), 300)
    return results


def fmt(r: dict, key: str) -> str:
    v = r.get(key)
    if v is None:
        return "-"
    return f"{v['p50']}/{v['p95']}/{v['p99']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int,
                        default=[1_000, 10_000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sizes = args.sizes or [1_000, 10_000]

    scale_map = {}
    for s in sizes:
        scale_map[s] = 1.0 if s <= 1_000 else (
            0.3 if s <= 10_000 else 0.1)

    all_results = []
    for s in sizes:
        all_results.append(run_size(s, scale_map[s], args.seed))

    ops = [("cold_load_ms", "cold load (ms)"),
           ("search_keyword", "search keyword"),
           ("land_anchor", "land anchor"),
           ("land_leaf", "land leaf"),
           ("steer", "steer"),
           ("route", "route (direction pick)"),
           ("slide", "slide (one-pass descent)"),
           ("write_new", "write new triple"),
           ("write_duplicate", "write duplicate"),
           ("fanout", "fanout check")]
    header = "| Operation | " + " | ".join(f"{r['nodes']:,} nodes" for r in all_results) + " |"
    print(header)
    print("|---|" + "---|" * len(all_results))
    print("| RSS memory | " + " | ".join(f"{r['rss_mb']} MB" for r in all_results) + " |")
    for key, name in ops:
        cells = []
        for r in all_results:
            if key.endswith("_ms"):
                cells.append(str(r[key]))
            else:
                cells.append(fmt(r, key))
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
