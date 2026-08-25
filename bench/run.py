# Benchmark harness. Builds synthetic workspaces at several sizes, times
# every public operation, and reports p50 and p95 latencies. Results go
# to stdout as a markdown table.

import json
import math
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miea_mem.core import Memory

SIZES = [1_000, 10_000, 100_000]
VERBS = ["likes", "studied_at", "uses", "knows_about", "prefers",
         "works_on", "contradicts", "supports", "relates_to", "built"]
NOUNS = ["topic_%d" % i for i in range(100_000)]
def iters(n_nodes):
    # Fewer iterations at bigger sizes; p50/p95 stay meaningful.
    scale = 1 if n_nodes <= 1_000 else (0.3 if n_nodes <= 10_000 else 0.1)
    return {k: max(30, int(v * scale)) for k, v in
            {"search": 200, "land": 500, "steer": 500, "write": 300}.items()}


def build_workspace(root: Path, n_nodes: int) -> Memory:
    # One user node plus n_nodes topic nodes, each linked from the user
    # with a random verb. Realistic-ish fanout on the anchor.
    store = __import__("miea_mem.store", fromlist=["Store"]).Store(root)
    manifest = store.init_workspace("Bench")
    mem = Memory(root, embedder=None)

    user = mem.create_node("User", type="anchor")
    for i in range(n_nodes):
        label = NOUNS[i]
        mem.create_node(label, content=f"detailed content about {label} "
                        "with enough words to make indexing realistic")
    for i in range(n_nodes):
        mem.write_triple(user.label, VERBS[i % len(VERBS)], NOUNS[i])

    # second hop edges between topics so steer has multi-level paths
    for i in range(0, n_nodes - 1, 5):
        mem.write_triple(NOUNS[i], "relates_to", NOUNS[i + 1])
    return mem


def bench(fn, iterations: int) -> dict:
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": round(statistics.median(times), 3),
        "p95": round(sorted(times)[int(len(times) * 0.95)], 3),
    }


def run_size(n_nodes: int):
    ITER = iters(n_nodes)
    # Build once and cache; the build itself is minutes of file writes at
    # 10k+ nodes and is not the thing being measured here.
    cache = Path(tempfile.gettempdir()) / f"miea_bench_ws_{n_nodes}"
    if not (cache / "manifest.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        build_workspace(cache, n_nodes)
    root = cache
    rows = []
    try:
        mem = Memory(str(root), embedder=None)

        # fresh load from disk (the real cold start cost)
        t0 = time.perf_counter()
        mem2 = Memory(root, embedder=None)
        load_ms = (time.perf_counter() - t0) * 1000

        rows.append({"op": "cold load (parse all files)", "p50": round(load_ms, 1),
                     "p95": "-"})
        _ = mem2  # keep alive

        rows.append({"op": "search (keyword)",
                     **bench(lambda: mem.search(f"content about topic_{random.randint(0, n_nodes)}"),
                             ITER["search"])})
        rows.append({"op": "land (anchor, high fanout)",
                     **bench(lambda: mem.land("User"), ITER["land"])})
        mid_label = NOUNS[n_nodes // 2]
        rows.append({"op": f"land (leaf, {mid_label})",
                     **bench(lambda: mem.land(mid_label), ITER["land"])})
        rows.append({"op": "steer (user to leaf)",
                     **bench(lambda: mem.steer("User", NOUNS[random.randint(0, n_nodes - 1)]),
                             ITER["steer"])})
        counter = [n_nodes]
        def do_write():
            counter[0] += 1
            mem.write_triple("User", "knows_about",
                             f"fresh_topic_{counter[0]}",
                             create_missing=True)
        rows.append({"op": "write triple (new node + edge)",
                     **bench(do_write, ITER["write"])})
        user_id = mem._resolve("User").id
        rows.append({"op": "fanout query on anchor",
                     **bench(lambda: mem.fanout(user_id), 500)})
        return rows
    finally:
        pass


def main():
    sizes = [int(s) for s in sys.argv[1:]] or SIZES
    print("| Operation |", " | ".join(f"{s:,} nodes" for s in sizes), "|")
    print("|---|" + "---|" * len(sizes))
    results = {}
    for size in sizes:
        print(f"# building {size} node workspace...", file=sys.stderr)
        rows = run_size(size)
        results[size] = {r["op"]: r for r in rows}
    first = results[sizes[0]] if sizes else {}
    ops = list(first.keys())
    for op in ops:
        cells = []
        for s in sizes:
            r = results[s].get(op, {})
            cells.append(f"{r.get('p50', '-')} / {r.get('p95', '-')}")
        print(f"| {op} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
