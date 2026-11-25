"""Per-item checkpoints: a relaunched job resumes the same adapter version, never another."""

from decisioneval.remote.checkpoint import CHUNK_ITEMS, Checkpoint
from decisioneval.remote.storage import LocalStore

PREFIX = "dev/evaluations/1/5"


def item(index: int, dataset: str = "pool") -> dict:
    return {"dataset": dataset, "item_id": f"q{index}", "correct": index % 2 == 0}


def test_items_are_written_in_create_only_chunks_and_read_back(tmp_path):
    store = LocalStore(tmp_path)
    checkpoint = Checkpoint(store, PREFIX, "hash-a")
    for index in range(CHUNK_ITEMS + 3):
        checkpoint.add(item(index))
    assert len(store.list(f"{PREFIX}/_checkpoint/hash-a")) == 1  # one full chunk so far
    checkpoint.flush()

    chunks = store.list(f"{PREFIX}/_checkpoint/hash-a")
    assert [path.rsplit("/", 1)[-1] for path in chunks] == ["000000.jsonl", "000001.jsonl"]
    restored = Checkpoint(store, PREFIX, "hash-a").load()
    assert [record["item_id"] for record in restored] == [f"q{i}" for i in range(CHUNK_ITEMS + 3)]


def test_a_relaunch_appends_after_the_chunks_already_there(tmp_path):
    store = LocalStore(tmp_path)
    first = Checkpoint(store, PREFIX, "hash-a")
    first.add(item(0))
    first.flush()

    second = Checkpoint(store, PREFIX, "hash-a")
    assert len(second.load()) == 1
    second.add(item(1))
    second.flush()

    assert [r["item_id"] for r in Checkpoint(store, PREFIX, "hash-a").load()] == ["q0", "q1"]


def test_another_adapter_version_never_resumes(tmp_path):
    store = LocalStore(tmp_path)
    checkpoint = Checkpoint(store, PREFIX, "hash-a")
    checkpoint.add(item(0))
    checkpoint.flush()
    assert Checkpoint(store, PREFIX, "hash-b").load() == []


def test_an_empty_flush_writes_nothing(tmp_path):
    store = LocalStore(tmp_path)
    Checkpoint(store, PREFIX, "hash-a").flush()
    assert store.list(PREFIX) == []


def test_the_same_item_seen_twice_is_kept_once(tmp_path):
    store = LocalStore(tmp_path)
    checkpoint = Checkpoint(store, PREFIX, "hash-a")
    checkpoint.add(item(0))
    checkpoint.flush()
    again = Checkpoint(store, PREFIX, "hash-a")
    again.add(item(0))
    again.flush()
    assert len(Checkpoint(store, PREFIX, "hash-a").load()) == 1
