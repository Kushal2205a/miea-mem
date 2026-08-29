"""Tests for the divergence map data model: JSON round-trip, tolerance of
old files without the field, and the pointer-only contract (labels and
ids, never generated text)."""

import json

from miea_mem.model import (
    DivergenceEntry,
    Node,
    node_from_dict,
    node_to_dict,
)


def test_divergence_entry_round_trips_through_json():
    n = Node(id="n1", label="Food", type="anchor")
    n.divergence_map = [
        DivergenceEntry(node_id="a1", label="Japanese cuisine",
                        kind="anchor", cue_leaf_id="s1", cue_label="Sushi"),
        DivergenceEntry(node_id="l1", label="Biryani", kind="leaf"),
    ]
    restored = node_from_dict(json.loads(json.dumps(node_to_dict(n))))
    assert restored.divergence_map == n.divergence_map
    assert restored.divergence_map[0].kind == "anchor"
    assert restored.divergence_map[1].cue_leaf_id is None


def test_node_without_map_field_loads_empty_and_writes_compact():
    # an old workspace file without divergenceMap must load with an
    # empty map, and a node with no map must not serialize the key at
    # all so existing files keep their byte shape
    old = {
        "schemaVersion": 1, "id": "n2", "type": "fact", "label": "WAL",
        "tags": [], "content": "write ahead log",
        "epistemicStatus": "unverifiable",
        "breadth": {"accessCount": 0, "traversalCount": 0,
                    "lastAccessed": None},
        "createdAt": "2026-01-01T00:00:00+00:00",
        "updatedAt": "2026-01-01T00:00:00+00:00",
    }
    n = node_from_dict(old)
    assert n.divergence_map == []
    assert "divergenceMap" not in node_to_dict(Node(id="n3", label="x"))


def test_map_stores_pointers_not_content():
    n = Node(id="n4", label="Food")
    n.divergence_map = [
        DivergenceEntry(node_id="a1", label="Japanese cuisine",
                        kind="anchor", cue_leaf_id="s1", cue_label="Sushi"),
    ]
    d = node_to_dict(n)
    # entry dicts carry identity and routing only, no free text beyond
    # the labels themselves
    assert all(set(e) <= {"nodeId", "label", "kind",
                          "cueLeafId", "cueLabel"}
               for e in d["divergenceMap"])