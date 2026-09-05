# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.patch.worker.patch_deepseek_mtp import (
    _EXTRA_LAYER_FILTER,
    _filter_extra_checkpoint_layers,
    _make_extra_layer_predicate,
    _patched_should_skip_weight,
)


def test_layer_predicate_skips_only_unconfigured_layers():
    should_skip = _make_extra_layer_predicate(num_hidden_layers=3)

    assert should_skip("model.layers.3.self_attn.q_proj.weight")
    assert should_skip("model.layers.78.shared_head.norm.weight")
    assert should_skip("layers.3.mlp.gate_proj.weight")
    assert not should_skip("model.layers.2.mlp.gate_proj.weight")
    assert not should_skip("model.embed_tokens.weight")
    assert not should_skip("rot.weight")


def test_filter_keeps_weight_order_and_non_layer_weights():
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", 0),
        ("model.layers.3.self_attn.q_proj.weight", 3),
        ("model.embed_tokens.weight", 4),
        ("layers.2.mlp.gate_proj.weight", 2),
        ("model.layers.78.shared_head.norm.weight", 78),
        ("rot.weight", 5),
    ]

    assert list(_filter_extra_checkpoint_layers(weights, num_hidden_layers=3)) == [
        ("model.layers.0.self_attn.q_proj.weight", 0),
        ("model.embed_tokens.weight", 4),
        ("layers.2.mlp.gate_proj.weight", 2),
        ("rot.weight", 5),
    ]


def test_patched_should_skip_weight_respects_layer_slice_context():
    token = _EXTRA_LAYER_FILTER.set(_make_extra_layer_predicate(num_hidden_layers=3))
    try:
        assert _patched_should_skip_weight("model.layers.3.self_attn.q_proj.weight", None)
        assert not _patched_should_skip_weight("model.layers.2.self_attn.q_proj.weight", None)
        assert not _patched_should_skip_weight("model.embed_tokens.weight", None)
    finally:
        _EXTRA_LAYER_FILTER.reset(token)

    assert _EXTRA_LAYER_FILTER.get() is None
