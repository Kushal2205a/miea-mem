# File storage. Reads and writes every node, edge and graph as its own
# JSON file under the workspace directory. Atomic writes. This is the
# only module that touches the files.

from __future__ import annotations

import json
from pathlib import Path

from .model import (
    Edge,
    Graph,
    Manifest,
    Node,
    edge_from_dict,
    edge_to_dict,
    graph_from_dict,
    graph_to_dict,
    manifest_from_dict,
    manifest_to_dict,
    new_id,
    node_from_dict,
    node_to_dict,
)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.nodes_dir = self.root / "nodes"
        self.edges_dir = self.root / "edges"
        self.graphs_dir = self.root / "graphs"

    def fingerprint(self) -> tuple:
        # Cheap change detector: file count plus newest modification time.
        # Both are needed; either alone misses some changes.
        latest = 0.0
        count = 0
        for d in (self.nodes_dir, self.edges_dir, self.graphs_dir):
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.suffix == ".json":
                    count += 1
                    try:
                        latest = max(latest, p.stat().st_mtime)
                    except OSError:
                        pass
        mf = self.root / "manifest.json"
        if mf.exists():
            count += 1
            latest = max(latest, mf.stat().st_mtime)
        return (count, round(latest, 3))

    def init_workspace(self, name: str = "Memory") -> Manifest:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.edges_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        root_graph = Graph(id=new_id(), name=name)
        manifest = Manifest(
            id=new_id(), name=name, root_graph_id=root_graph.id,
            graph_ids=[root_graph.id],
        )
        (self.graphs_dir / f"{root_graph.id}.json").write_text(
            json.dumps(graph_to_dict(root_graph), indent=2)
        )
        (self.root / "manifest.json").write_text(
            json.dumps(manifest_to_dict(manifest), indent=2)
        )
        return manifest

    def exists(self) -> bool:
        return (self.root / "manifest.json").exists()

    def load_manifest(self) -> Manifest:
        return manifest_from_dict(
            json.loads((self.root / "manifest.json").read_text())
        )

    def save_manifest(self, m: Manifest) -> None:
        path = self.root / "manifest.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest_to_dict(m), indent=2))
        tmp.replace(path)

    def save_node(self, n: Node) -> None:
        from .model import _write

        _write(self.nodes_dir / f"{n.id}.json", node_to_dict(n))

    def load_node(self, node_id: str) -> Node | None:
        p = self.nodes_dir / f"{node_id}.json"
        if not p.exists():
            return None
        return node_from_dict(json.loads(p.read_text()))

    def delete_node(self, node_id: str) -> None:
        (self.nodes_dir / f"{node_id}.json").unlink(missing_ok=True)

    def all_nodes(self) -> list[Node]:
        return [
            node_from_dict(json.loads(p.read_text()))
            for p in sorted(self.nodes_dir.glob("*.json"))
        ]

    def save_edge(self, e: Edge) -> None:
        from .model import _write

        _write(self.edges_dir / f"{e.id}.json", edge_to_dict(e))

    def load_edge(self, edge_id: str) -> Edge | None:
        p = self.edges_dir / f"{edge_id}.json"
        if not p.exists():
            return None
        return edge_from_dict(json.loads(p.read_text()))

    def delete_edge(self, edge_id: str) -> None:
        (self.edges_dir / f"{edge_id}.json").unlink(missing_ok=True)

    def all_edges(self) -> list[Edge]:
        return [
            edge_from_dict(json.loads(p.read_text()))
            for p in sorted(self.edges_dir.glob("*.json"))
        ]

    def save_graph(self, g: Graph) -> None:
        from .model import _write

        _write(self.graphs_dir / f"{g.id}.json", graph_to_dict(g))

    def load_graph(self, graph_id: str) -> Graph | None:
        p = self.graphs_dir / f"{graph_id}.json"
        if not p.exists():
            return None
        return graph_from_dict(json.loads(p.read_text()))

    def delete_graph(self, graph_id: str) -> None:
        (self.graphs_dir / f"{graph_id}.json").unlink(missing_ok=True)

    def all_graphs(self) -> list[Graph]:
        return [
            graph_from_dict(json.loads(p.read_text()))
            for p in sorted(self.graphs_dir.glob("*.json"))
        ]
