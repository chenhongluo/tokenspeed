# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import os
from test.runtime.test_lite_model_loader import lite_config_dict

import torch
from lite_oe_checkpoint_probe import (
    _touch_tables,
    load_oe_checkpoint,
    mapping_metrics,
    run_ledger,
)
from safetensors.torch import load_file, save_file


def _tiny_checkpoint(path):
    config = lite_config_dict()
    tensors = {}
    weight_map = {}
    hidden = config["hidden_size"]
    oe_hidden = hidden // 12
    for table_id in range(12):
        rows = int(config["vocab_size"] * config["ngram_vocab_size_ratio"]) + 1
        rows += table_id * 2
        table_name = f"model.ngram_embeddings.embedders.{table_id}.weight"
        projection_name = f"model.ngram_embeddings.post_projs.{table_id}.weight"
        tensors[table_name] = torch.arange(
            rows * oe_hidden, dtype=torch.bfloat16
        ).reshape(rows, oe_hidden)
        tensors[projection_name] = torch.arange(
            hidden * oe_hidden, dtype=torch.bfloat16
        ).reshape(hidden, oe_hidden)
        weight_map[table_name] = "model.safetensors"
        weight_map[projection_name] = "model.safetensors"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    save_file(tensors, path / "model.safetensors")
    return sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in tensors.items()
        if ".embedders." in name
    )


def test_mapping_metrics_deduplicates_file_backed_storage(tmp_path) -> None:
    checkpoint = tmp_path / "mapped.safetensors"
    save_file(
        {
            "a": torch.arange(4096, dtype=torch.int32),
            "b": torch.arange(4096, dtype=torch.int32),
        },
        checkpoint,
    )
    tensors = load_file(checkpoint, device="cpu")
    metrics = mapping_metrics(
        os.getpid(), [tensors["a"].data_ptr(), tensors["b"].data_ptr()]
    )

    assert metrics["mapping_count"] == 1
    assert metrics["all_file_backed"]
    assert metrics["anonymous_kib"] == 0


def test_tiny_checkpoint_adoption_and_single_worker_ledger(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    payload_bytes = _tiny_checkpoint(checkpoint)
    layer, storage, _ = load_oe_checkpoint(checkpoint, device=None)

    assert storage["payload_bytes"] == payload_bytes
    assert storage["shard_count"] == 1
    assert all(table.weight.device.type == "cpu" for table in layer.embedders)

    output = tmp_path / "ledger.json"
    result = run_ledger(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            output=str(output),
            workers=1,
            devices="",
            touch_mib=1,
            timeout=60,
            require_exact_layout=False,
            exact_numerics=False,
        )
    )
    assert result["passed"]
    assert all(result["checks"].values())
    assert str(checkpoint) not in json.dumps(result)


def test_touch_set_reads_one_element_per_os_page(monkeypatch) -> None:
    class Table:
        def __init__(self):
            self.weight = torch.arange(12, dtype=torch.int32)

    layer = argparse.Namespace(embedders=[Table(), Table()])
    monkeypatch.setattr(os, "sysconf", lambda _name: 16)

    assert _touch_tables(layer, touch_mib=1) == float(sum((0, 4, 8)) * 2)
