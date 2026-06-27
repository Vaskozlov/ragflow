#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


class TaskCanceledException(Exception):
    pass


class GraphChange:
    def __init__(self):
        self.added_updated_nodes = set()
        self.added_updated_edges = set()


async def _async_none(*_args, **_kwargs):
    return None


def _module(name: str, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _timeout(*_args, **_kwargs):
    return lambda fn: fn


def _load_index_module():
    path = Path(__file__).resolve().parents[4] / "rag/graphrag/general/index.py"
    spec = importlib.util.spec_from_file_location("graphrag_index_timeout_config_test", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "networkx": _module("networkx", Graph=object, node_link_graph=lambda *_a, **_k: None, node_link_data=lambda *_a, **_k: {}, pagerank=lambda *_a, **_k: {}),
        "api.db.services": _module("api.db.services"),
        "api.db.services.document_service": _module("api.db.services.document_service", DocumentService=SimpleNamespace()),
        "api.db.services.task_service": _module("api.db.services.task_service", has_canceled=lambda *_a, **_k: False),
        "common.connection_utils": _module("common.connection_utils", timeout=_timeout),
        "common.doc_store.doc_store_base": _module("common.doc_store.doc_store_base", OrderByExpr=object),
        "common.exceptions": _module("common.exceptions", TaskCanceledException=TaskCanceledException),
        "common.misc_utils": _module("common.misc_utils", thread_pool_exec=_async_none),
        "common.settings": _module("common.settings"),
        "rag.graphrag.checkpoints": _module(
            "rag.graphrag.checkpoints",
            COMMUNITY_CHECKPOINT="community",
            RESOLUTION_CHECKPOINT="resolution",
            cleanup_checkpoints=_async_none,
            load_checkpoints=_async_none,
            save_checkpoint=_async_none,
        ),
        "rag.graphrag.entity_resolution": _module("rag.graphrag.entity_resolution", EntityResolution=object),
        "rag.graphrag.general.community_reports_extractor": _module("rag.graphrag.general.community_reports_extractor", CommunityReportsExtractor=object),
        "rag.graphrag.general.extractor": _module("rag.graphrag.general.extractor", Extractor=object),
        "rag.graphrag.general.graph_extractor": _module("rag.graphrag.general.graph_extractor", GraphExtractor=object),
        "rag.graphrag.light.graph_extractor": _module("rag.graphrag.light.graph_extractor", GraphExtractor=object),
        "rag.graphrag.ner.graph_extractor": _module("rag.graphrag.ner.graph_extractor", GraphExtractor=object),
        "rag.graphrag.phase_markers": _module(
            "rag.graphrag.phase_markers",
            PHASE_COMMUNITY="community",
            PHASE_RESOLUTION="resolution",
            clear_phase_markers=lambda *_a, **_k: None,
            has_phase_marker=lambda *_a, **_k: False,
            set_phase_marker=lambda *_a, **_k: None,
        ),
        "rag.graphrag.utils": _module(
            "rag.graphrag.utils",
            GraphChange=GraphChange,
            chunk_id=lambda *_a, **_k: "chunk-id",
            does_graph_contains=_async_none,
            get_graph=_async_none,
            graph_merge=lambda old_graph, *_a, **_k: old_graph,
            insert_chunks_bounded=_async_none,
            set_graph=_async_none,
            tidy_graph=lambda *_a, **_k: None,
        ),
        "rag.nlp": _module(
            "rag.nlp",
            rag_tokenizer=SimpleNamespace(tokenize=lambda value: value, fine_grained_tokenize=lambda value: value),
            search=SimpleNamespace(index_name=lambda tenant_id: tenant_id),
        ),
        "rag.utils.redis_conn": _module("rag.utils.redis_conn", RedisDistributedLock=object),
    }
    with patch.dict(sys.modules, stubs):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


index = _load_index_module()


_GRAPHRAG_ENV_NAMES = [
    "GRAPHRAG_CONFIG_ENV_OVERRIDE",
    "GRAPHRAG_MERGE_TIMEOUT_SECONDS",
    "GRAPHRAG_MERGE_RETRY_ATTEMPTS",
    "GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS",
    "GRAPHRAG_RESOLUTION_TIMEOUT_SECONDS",
    "GRAPHRAG_COMMUNITY_TIMEOUT_SECONDS",
    "GRAPHRAG_RETRY_ATTEMPTS",
    "GRAPHRAG_RETRY_BACKOFF_SECONDS",
    "GRAPHRAG_RETRY_BACKOFF_MAX_SECONDS",
]


@pytest.fixture(autouse=True)
def clear_graphrag_env(monkeypatch):
    for name in _GRAPHRAG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.p1
def test_env_fallback_when_parser_config_key_missing(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_MERGE_TIMEOUT_SECONDS", "1800")

    value = index._bounded_int_config_with_env(
        {},
        "merge_timeout_seconds",
        "GRAPHRAG_MERGE_TIMEOUT_SECONDS",
        index.DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS,
        0,
        86400,
    )

    assert value == 1800


@pytest.mark.p1
def test_parser_config_wins_over_env_by_default(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_MERGE_TIMEOUT_SECONDS", "1800")

    value = index._bounded_int_config_with_env(
        {"merge_timeout_seconds": 240},
        "merge_timeout_seconds",
        "GRAPHRAG_MERGE_TIMEOUT_SECONDS",
        index.DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS,
        0,
        86400,
    )

    assert value == 240


@pytest.mark.p1
def test_env_wins_when_graphrag_config_env_override_enabled(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_CONFIG_ENV_OVERRIDE", "true")
    monkeypatch.setenv("GRAPHRAG_MERGE_TIMEOUT_SECONDS", "1800")

    value = index._bounded_int_config_with_env(
        {"merge_timeout_seconds": 240},
        "merge_timeout_seconds",
        "GRAPHRAG_MERGE_TIMEOUT_SECONDS",
        index.DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS,
        0,
        86400,
    )

    assert value == 1800


@pytest.mark.p1
def test_invalid_env_value_falls_back_and_logs_warning(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("GRAPHRAG_MERGE_TIMEOUT_SECONDS", "not-an-int")

    value = index._bounded_int_config_with_env(
        {},
        "merge_timeout_seconds",
        "GRAPHRAG_MERGE_TIMEOUT_SECONDS",
        index.DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS,
        0,
        86400,
    )

    assert value == index.DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS
    assert "Invalid env GRAPHRAG_MERGE_TIMEOUT_SECONDS='not-an-int'" in caplog.text


@pytest.mark.p1
def test_lock_acquire_timeout_uses_env_fallback(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS", "1800")

    assert index._lock_acquire_timeout_config({}) == 1800


@pytest.mark.p1
@pytest.mark.asyncio
async def test_run_with_retry_zero_timeout_disables_wait_for(monkeypatch):
    async def fail_wait_for(*_args, **_kwargs):
        raise AssertionError("asyncio.wait_for should not be used when timeout_seconds=0")

    async def slow_result():
        await asyncio.sleep(0.01)
        return "ok"

    monkeypatch.setattr(index.asyncio, "wait_for", fail_wait_for)

    result = await index._run_with_retry(
        "merge_subgraph doc:doc-1",
        slow_result,
        attempts=1,
        timeout_seconds=0,
        backoff_seconds=0,
        backoff_max_seconds=0,
    )

    assert result == "ok"


@pytest.mark.p1
def test_merge_subgraph_has_no_fixed_timeout_decorator():
    source = inspect.getsource(index.merge_subgraph)

    assert source.lstrip().startswith("async def merge_subgraph")
    assert "@timeout(60 * 3)" not in source
