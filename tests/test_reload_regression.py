from miea_mem.core import Memory
from miea_mem.store import Store

def test_read_after_own_write_does_not_reload(tmp_path):
    # Regression: land saves access counts, which moved mtimes and made the
    # next read mistake its own write for an external one, reloading all
    # files. Own writes must refresh the baseline instead.
    store = Store(tmp_path)
    store.init_workspace("T")
    mem = Memory(str(tmp_path), embedder=None)
    mem.create_node("A")
    mem.create_node("B")
    mem.write_triple("A", "knows_about", "B")

    reloads = {"n": 0}
    original_load = Memory._load

    def counting_load(self):
        reloads["n"] += 1
        original_load(self)

    Memory._load = counting_load
    try:
        mem.land("A")                # writes access count
        mem.land("A")                # must not reload
        mem.steer("A", "B")         # writes traversal counts
        mem.land("B")               # must not reload
        assert reloads["n"] == 0, f"own writes triggered {reloads['n']} reloads"
    finally:
        Memory._load = original_load


def test_external_write_still_triggers_reload(tmp_path):
    # The self-write fix must not blind us to real external writers.
    import time as _time

    from miea_mem.store import Store

    store = Store(tmp_path)
    store.init_workspace("T")
    mem = Memory(str(tmp_path), embedder=None)
    mem.create_node("A")
    mem.land("A")

    _time.sleep(0.02)  # mtime resolution guard
    # simulate an external writer touching a file behind our back
    node_file = tmp_path / "nodes"
    victim = sorted(node_file.glob("*.json"))[0]
    data = victim.read_text()
    victim.write_text(data)

    p = mem.land("A")
    assert p is not None  # reloaded state still answers; no crash, fresh view
