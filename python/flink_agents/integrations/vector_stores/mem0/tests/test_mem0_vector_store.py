################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
################################################################################
from __future__ import annotations

import hashlib
import os
from contextlib import suppress
from typing import Any, Dict, List

import pytest

from flink_agents.api.vector_stores.vector_store import (
    Document,
)


def _embed(text: str) -> List[float]:
    """Deterministic 8-dim vector derived from MD5 of the input."""
    digest = hashlib.md5(text.encode()).digest()
    return [b / 255.0 for b in digest[:8]]


def _doc(
    id_: str,
    content: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Document:
    """Build a ``Document`` with a pre-computed embedding — matches how
    Mem0 hands data to the reverse adapter in practice.
    """
    return Document(
        id=id_,
        content=content or id_,
        metadata=metadata or {},
        embedding=_embed(content or id_),
    )


try:
    import chromadb  # noqa: F401
    import mem0  # noqa: F401

    _backend_available = True
except ImportError:
    _backend_available = False

if _backend_available:
    from flink_agents.integrations.vector_stores.mem0.mem0_vector_store import (
        Mem0VectorStore,
    )
else:
    Mem0VectorStore = object  # type: ignore[assignment]

try:
    import pymilvus  # noqa: F401

    _milvus_available = True
except ImportError:
    _milvus_available = False

_milvus_uri = os.environ.get("MILVUS_URI")
_milvus_test_db = "flink_agents_mem0_test"
_milvus_strong_consistency = "Strong"
_mem0_provider = "chroma"
_created_stores: List[Any] = []

pytestmark = pytest.mark.skipif(
    not _backend_available, reason="mem0 / chromadb is not available"
)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
@pytest.fixture(params=["chroma", "milvus"], autouse=True)
def _vector_store_suite(request, monkeypatch) -> None:
    global _created_stores, _mem0_provider

    _mem0_provider = request.param
    _created_stores = []
    if _mem0_provider == "milvus":
        if not _milvus_available:
            pytest.skip("pymilvus is not available")
        if not _milvus_uri:
            pytest.skip("MILVUS_URI is not set")
        _clear_milvus_test_database()
        _use_strong_milvus_consistency(monkeypatch)

    try:
        yield
    finally:
        for store in _created_stores:
            for name in list(store._stores):
                with suppress(Exception):
                    store.delete_collection(name)
        if _mem0_provider == "milvus" and _milvus_available and _milvus_uri:
            _clear_milvus_test_database()
        _mem0_provider = "chroma"
        _created_stores = []


def _make_store(tmp_path, *, collection: str = "default_col") -> Mem0VectorStore:
    """Construct a :class:`Mem0VectorStore` backed by the active Mem0 provider.

    No ``embedding_model`` is configured: Mem0 always hands us pre-computed
    vectors, so the reverse adapter never needs to auto-embed.
    """
    if _mem0_provider == "chroma":
        provider_config = {"path": str(tmp_path)}
    elif _mem0_provider == "milvus":
        provider_config = {
            "url": _milvus_uri,
            "token": None,
            "embedding_model_dims": 8,
            "metric_type": "COSINE",
            "db_name": _milvus_test_db,
        }
    else:
        msg = f"Unknown Mem0 provider: {_mem0_provider}"
        raise ValueError(msg)

    vs = Mem0VectorStore(
        provider=_mem0_provider,
        provider_config=provider_config,
        collection=collection,
    )
    vs.open()
    _created_stores.append(vs)
    return vs


def _use_strong_milvus_consistency(monkeypatch) -> None:
    import mem0.vector_stores.milvus as mem0_milvus

    class StrongConsistencyMilvusClient(mem0_milvus.MilvusClient):
        def create_collection(self, *args: Any, **kwargs: Any) -> Any:
            # Test-only: use STRONG consistency so the existing contract tests can
            # assert read-after-write behavior without sleeps or retries.
            kwargs.setdefault("consistency_level", _milvus_strong_consistency)
            return super().create_collection(*args, **kwargs)

    monkeypatch.setattr(mem0_milvus, "MilvusClient", StrongConsistencyMilvusClient)


def _clear_milvus_test_database() -> None:
    from pymilvus import MilvusClient

    root_client = MilvusClient(uri=_milvus_uri, token=None)
    if _milvus_test_db not in root_client.list_databases():
        root_client.create_database(_milvus_test_db)

    client = MilvusClient(uri=_milvus_uri, token=None, db_name=_milvus_test_db)
    for name in client.list_collections():
        client.drop_collection(name)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------
def test_create_collection_if_not_exists_creates_lazily(tmp_path) -> None:
    vs = _make_store(tmp_path)
    assert len(vs._stores) == 0  # cache starts empty
    vs.create_collection_if_not_exists("fresh")
    assert "fresh" in vs._stores


def test_delete_collection_raises_when_missing(tmp_path) -> None:
    vs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        vs.delete_collection("never_created")


def test_delete_collection_removes_from_cache(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.create_collection_if_not_exists("some_col")
    vs.delete_collection("some_col")
    assert "some_col" not in vs._stores


# ---------------------------------------------------------------------------
# Read paths must not silently materialize a collection
# ---------------------------------------------------------------------------
def test_get_on_missing_raises_and_does_not_create(tmp_path) -> None:
    vs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        vs.get(ids="x", collection_name="nope")
    assert "nope" not in vs._stores


def test_delete_on_missing_raises_and_does_not_create(tmp_path) -> None:
    vs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        vs.delete(ids="x", collection_name="nope")
    assert "nope" not in vs._stores


def test_query_on_missing_raises_and_does_not_create(tmp_path) -> None:
    vs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        vs._query_embedding(embedding=[0.1] * 8, limit=5, collection_name="nope")
    assert "nope" not in vs._stores


def test_update_on_missing_raises_and_does_not_create(tmp_path) -> None:
    vs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        vs.update(documents=_doc("x"), collection_name="nope")
    assert "nope" not in vs._stores


# ---------------------------------------------------------------------------
# Write paths do lazy-create
# ---------------------------------------------------------------------------
def test_add_lazy_creates_collection(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("id_a", "hello world", {"k": "v"})],
        collection_name="fresh_col",
    )
    assert "fresh_col" in vs._stores


def test_add_uses_collection_when_none(tmp_path) -> None:
    vs = _make_store(tmp_path, collection="the_default")
    vs.add(documents=[_doc("x", "hi")])
    assert "the_default" in vs._stores


# ---------------------------------------------------------------------------
# CRUD round-trip
# ---------------------------------------------------------------------------
def test_add_and_get_by_ids(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[
            _doc("a", "alpha", {"kind": "x"}),
            _doc("b", "beta", {"kind": "y"}),
        ],
        collection_name="roundtrip",
    )
    got = vs.get(ids=["a"], collection_name="roundtrip")
    assert len(got) == 1
    assert got[0].id == "a"
    assert got[0].content == "alpha"
    assert got[0].metadata == {"kind": "x"}


def test_add_and_get_without_ids_returns_all(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("a", "alpha"), _doc("b", "beta")],
        collection_name="getall",
    )
    got = vs.get(collection_name="getall")
    assert {d.id for d in got} == {"a", "b"}


def test_get_respects_explicit_limit(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc(f"d{i}", f"c{i}") for i in range(5)],
        collection_name="limit_col",
    )
    got = vs.get(collection_name="limit_col", limit=2)
    assert len(got) == 2


def test_get_unbounded_when_limit_is_none(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc(f"d{i}", f"c{i}") for i in range(5)],
        collection_name="unbounded_col",
    )
    got = vs.get(collection_name="unbounded_col", limit=None)
    assert len(got) == 5


def test_query_embedding_returns_all_when_limit_covers(tmp_path) -> None:
    # Adapter-style usage: Mem0 holds pre-computed vectors and routes
    # straight through the protected embedding hook.
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("a", "alpha"), _doc("b", "beta")],
        collection_name="qtest",
    )
    docs = vs._query_embedding(
        embedding=_embed("alpha"),
        limit=5,
        collection_name="qtest",
    )
    assert {d.id for d in docs} == {"a", "b"}
    # scores should be populated by Mem0's backend (ChromaDB distances)
    assert all(d.score is not None for d in docs)


def test_update_overwrites_document(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("a", "old", {"k": "v"})],
        collection_name="upd",
    )
    vs.update(
        documents=_doc("a", "new", {"k": "w"}),
        collection_name="upd",
    )
    got = vs.get(ids=["a"], collection_name="upd")
    assert got[0].content == "new"
    assert got[0].metadata == {"k": "w"}


def test_delete_by_ids(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("a", "alpha"), _doc("b", "beta")],
        collection_name="del",
    )
    vs.delete(ids="a", collection_name="del")
    remaining = vs.get(collection_name="del")
    assert {d.id for d in remaining} == {"b"}


def test_delete_by_filters(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[
            _doc("a", "alpha", {"kind": "x"}),
            _doc("b", "beta", {"kind": "y"}),
        ],
        collection_name="delf",
    )
    vs.delete(filters={"kind": "x"}, collection_name="delf")
    remaining = vs.get(collection_name="delf")
    assert {d.id for d in remaining} == {"b"}


def test_delete_all_when_no_selector(tmp_path) -> None:
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("a", "alpha"), _doc("b", "beta")],
        collection_name="dall",
    )
    vs.delete(collection_name="dall")
    assert vs.get(collection_name="dall") == []


# ---------------------------------------------------------------------------
# Document <-> Mem0 payload round-trip
# ---------------------------------------------------------------------------
def test_data_payload_roundtrip(tmp_path) -> None:
    """``Document.content`` stores as Mem0's payload ``"data"`` key; reads
    extract it back into content while keeping other metadata keys intact.
    """
    vs = _make_store(tmp_path)
    vs.add(
        documents=[_doc("x", "my text", {"u": "u1"})],
        collection_name="roundtrip_data",
    )
    got = vs.get(ids=["x"], collection_name="roundtrip_data")
    assert got[0].content == "my text"
    # "data" must not leak back into metadata.
    assert got[0].metadata == {"u": "u1"}


# ---------------------------------------------------------------------------
# embedding_model-not-configured path
# ---------------------------------------------------------------------------
def test_auto_embed_without_model_raises(tmp_path) -> None:
    """Without an ``embedding_model`` configured, any path that tries to
    auto-embed raises a clear ``TypeError`` — callers must either set one
    or pre-compute vectors (the Mem0-style usage this adapter is built for).
    """
    vs = _make_store(tmp_path)
    vs.create_collection_if_not_exists("no_emb")
    with pytest.raises(TypeError, match="No embedding model configured"):
        vs.add(
            documents=[Document(id="x", content="no embedding")],
            collection_name="no_emb",
        )
