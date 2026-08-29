# Data model. Defines the four entities (Node, Edge, Graph, Manifest)
# and the dict translators used for JSON file persistence.

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Entities

@dataclass
class Breadth:
    access_count: int = 0
    traversal_count: int = 0  # times edges through this node were ridden
    last_accessed: str | None = None


@dataclass
class DivergenceEntry:
    # One fork entry in a parent's divergence map: a branch route, never a
    # summary. kind "anchor" routes to an intermediate node and carries a
    # cue to its hottest leaf; kind "leaf" is a singleton branch routing to
    # itself, cue fields left unset. Pointers and labels only, ordering is
    # computed live from breadth at read time.
    node_id: str
    label: str
    kind: str = "leaf"  # anchor | leaf
    cue_leaf_id: str | None = None
    cue_label: str | None = None


@dataclass
class Node:
    id: str
    label: str
    type: str = "fact"  # fact | preference | procedure | event | claim | anchor
    tags: list[str] = field(default_factory=list)
    content: str = ""  # plain text, the LLM interprets at read time
    child_graph_id: str | None = None
    # fork entries for the branch tier beneath this node. Derived but
    # persisted (a rebuildable cache in the node file): mark dirty on
    # structure writes, regenerate on access.
    divergence_map: list[DivergenceEntry] = field(default_factory=list)
    epistemic_status: str = "unverifiable"
    breadth: Breadth = field(default_factory=Breadth)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class Edge:
    id: str
    source_id: str
    target_id: str
    verb: str  # active form, passive derivable ("teaches" / "is-taught-by")
    created_at: str = field(default_factory=now_iso)


@dataclass
class Graph:
    # A nested graph. The root graph has parent_node_id=None.
    id: str
    name: str
    node_ids: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)
    parent_node_id: str | None = None


@dataclass
class Manifest:
    id: str
    name: str
    root_graph_id: str
    graph_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


# Persistence helpers

def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)  # atomic


# Node translation

def divergence_entry_to_dict(e: DivergenceEntry) -> dict:
    return {
        "nodeId": e.node_id,
        "label": e.label,
        "kind": e.kind,
        **({"cueLeafId": e.cue_leaf_id} if e.cue_leaf_id else {}),
        **({"cueLabel": e.cue_label} if e.cue_label else {}),
    }


def divergence_entry_from_dict(d: dict) -> DivergenceEntry:
    return DivergenceEntry(
        node_id=d["nodeId"],
        label=d.get("label", ""),
        kind=d.get("kind", "leaf"),
        cue_leaf_id=d.get("cueLeafId"),
        cue_label=d.get("cueLabel"),
    )


def node_to_dict(n: Node) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": n.id,
        "type": n.type,
        "label": n.label,
        "tags": n.tags,
        "content": n.content,
        **({"childGraphId": n.child_graph_id} if n.child_graph_id else {}),
        **({"divergenceMap": [divergence_entry_to_dict(e)
                              for e in n.divergence_map]}
           if n.divergence_map else {}),
        "epistemicStatus": n.epistemic_status,
        "breadth": {
            "accessCount": n.breadth.access_count,
            "traversalCount": n.breadth.traversal_count,
            "lastAccessed": n.breadth.last_accessed,
        },
        "createdAt": n.created_at,
        "updatedAt": n.updated_at,
    }


def node_from_dict(d: dict) -> Node:
    b = d.get("breadth", {})
    return Node(
        id=d["id"],
        label=d.get("label", ""),
        type=d.get("type", "fact"),
        tags=d.get("tags", []),
        content=d.get("content", ""),
        child_graph_id=d.get("childGraphId"),
        divergence_map=[divergence_entry_from_dict(e)
                        for e in d.get("divergenceMap", [])],
        epistemic_status=d.get("epistemicStatus", "unverifiable"),
        breadth=Breadth(
            access_count=b.get("accessCount", 0),
            traversal_count=b.get("traversalCount", 0),
            last_accessed=b.get("lastAccessed"),
        ),
        created_at=d.get("createdAt", now_iso()),
        updated_at=d.get("updatedAt", now_iso()),
    )


# Edge translation

def edge_to_dict(e: Edge) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": e.id,
        "sourceId": e.source_id,
        "targetId": e.target_id,
        "verb": e.verb,
        "createdAt": e.created_at,
    }


def edge_from_dict(d: dict) -> Edge:
    return Edge(
        id=d["id"],
        source_id=d["sourceId"],
        target_id=d["targetId"],
        verb=d.get("verb", "relates_to"),
        created_at=d.get("createdAt", now_iso()),
    )


# Graph translation

def graph_to_dict(g: Graph) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": g.id,
        "name": g.name,
        "parentNodeId": g.parent_node_id,
        "nodeIds": sorted(g.node_ids),
        "edgeIds": sorted(g.edge_ids),
    }


def graph_from_dict(d: dict) -> Graph:
    return Graph(
        id=d["id"],
        name=d.get("name", ""),
        node_ids=set(d.get("nodeIds", [])),
        edge_ids=set(d.get("edgeIds", [])),
        parent_node_id=d.get("parentNodeId"),
    )


# Manifest translation

def manifest_to_dict(m: Manifest) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": m.id,
        "name": m.name,
        "rootGraphId": m.root_graph_id,
        "graphIds": m.graph_ids,
        "createdAt": m.created_at,
        "updatedAt": m.updated_at,
    }


def manifest_from_dict(d: dict) -> Manifest:
    return Manifest(
        id=d["id"],
        name=d.get("name", "Memory"),
        root_graph_id=d["rootGraphId"],
        graph_ids=d.get("graphIds", []),
        created_at=d.get("createdAt", now_iso()),
        updated_at=d.get("updatedAt", now_iso()),
    )


__all__ = [
    "Breadth",
    "DivergenceEntry",
    "Edge",
    "Graph",
    "Manifest",
    "Node",
    "SCHEMA_VERSION",
    "divergence_entry_from_dict",
    "divergence_entry_to_dict",
    "edge_from_dict",
    "edge_to_dict",
    "graph_from_dict",
    "graph_to_dict",
    "manifest_from_dict",
    "manifest_to_dict",
    "new_id",
    "node_from_dict",
    "node_to_dict",
    "now_iso",
]
