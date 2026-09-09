import torch
from torch import nn
from torch.nn import functional as F

from badit_tf.core import TFAbilityLinear


def _layer() -> TFAbilityLinear:
    torch.manual_seed(3)
    return TFAbilityLinear(
        nn.Linear(24, 32, bias=False),
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=2,
        dense_steps=1,
        svd_seed=11,
        svd_oversample=2,
        svd_niter=2,
    )


def test_all_one_reconstructs_and_assignment_is_balanced():
    layer = _layer()
    assert layer.initial_weight_relative_error.item() < 1e-6
    matrix = layer.assignment_matrix()
    assert matrix.shape == (8, 4)
    assert torch.equal(matrix.sum(1), torch.ones(8, dtype=torch.int64))
    assert torch.equal(matrix.sum(0), torch.full((4,), 2, dtype=torch.int64))
    x = torch.randn(2, 5, 24)
    raw, dense, actual = layer.route(x)
    assert torch.count_nonzero(raw) == 0
    assert torch.equal(dense, torch.ones_like(dense))
    assert torch.equal(actual, dense)
    assert not bool(layer.topk_enabled.item())
    expected = layer.base_layer(x)
    observed = layer(x)
    assert torch.equal(observed, expected)


def test_nonempty_unequal_assignment_routes_primitives_without_new_parameters():
    torch.manual_seed(23)
    layer = TFAbilityLinear(
        nn.Linear(12, 16, bias=False),
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=2,
        dense_steps=0,
        allow_unequal_capacity=True,
        initialization_method="kaiming_zero",
    )
    before = sum(parameter.numel() for parameter in layer.parameters())
    labels = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3])
    layer.apply_assignment(labels)
    matrix = layer.assignment_matrix()
    assert torch.equal(matrix.sum(0), torch.tensor([3, 1, 3, 1]))
    assert torch.equal(matrix.sum(1), torch.ones(8, dtype=torch.int64))
    assert sum(parameter.numel() for parameter in layer.parameters()) == before
    x = torch.randn(2, 3, 12)
    output = layer(x)
    assert output.shape == (2, 3, 16)
    assert torch.isfinite(output).all()


def test_dense_first_update_then_mask_only_topk():
    layer = _layer()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    x = torch.randn(2, 5, 24)
    target = torch.randn(2, 5, 32)
    before = layer.residual_router.weight.detach().clone()
    loss = (layer(x) - target).square().mean()
    loss.backward()
    grad_norm = layer.residual_router.weight.grad.norm().item()
    assert grad_norm > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, layer.residual_router.weight)
    layer.advance_optimizer_step()
    assert bool(layer.topk_enabled.item())
    _, dense, actual = layer.route(x)
    assert torch.all((actual == 0).sum(dim=-1) == 2)
    selected = actual != 0
    assert torch.equal(actual[selected], dense[selected])
    assert not torch.allclose(actual.sum(dim=-1), torch.ones(2))


def test_topk_ties_choose_lowest_expert_ids_deterministically():
    layer = _layer()
    layer.advance_optimizer_step()
    raw, dense, actual = layer.route(torch.randn(3, 5, 24))
    assert torch.count_nonzero(raw) == 0
    assert torch.equal(dense, torch.ones_like(dense))
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0]).expand_as(actual)
    assert torch.equal(actual, expected)


def test_upstream_softmax_diagnostic_uses_normalized_topk_weights():
    torch.manual_seed(3)
    layer = TFAbilityLinear(
        nn.Linear(24, 32, bias=False),
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=2,
        dense_steps=1,
        router_mode="upstream_softmax",
        svd_seed=11,
        svd_oversample=2,
        svd_niter=2,
    )
    x = torch.randn(2, 5, 24)
    raw, dense, actual = layer.route(x)
    assert torch.count_nonzero(raw) > 0
    assert torch.allclose(dense.sum(dim=-1), torch.ones(2))
    assert torch.equal(actual, dense)
    layer.advance_optimizer_step()
    _, dense, actual = layer.route(x)
    assert torch.all((actual == 0).sum(dim=-1) == 2)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2))
    assert not torch.equal(actual, dense)


def test_public_router_is_bias_free_and_topk_from_first_forward():
    layer = TFAbilityLinear(
        nn.Linear(24, 32, bias=False),
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=2,
        dense_steps=0,
        router_mode="upstream_softmax",
        router_bias=False,
        svd_method="exact",
        svd_seed=11,
        svd_oversample=2,
        svd_niter=2,
    )
    assert layer.residual_router.bias is None
    assert bool(layer.topk_enabled.item())
    _, _, actual = layer.route(torch.randn(2, 5, 24))
    assert torch.all((actual == 0).sum(dim=-1) == 2)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2))


def test_public_slot_regroup_keeps_fixed_residual_and_records_duplicates():
    layer = TFAbilityLinear(
        nn.Linear(24, 32, bias=False),
        num_experts=2,
        rank=2,
        top_k=2,
        lora_dropout=0,
        cache_last_gradients=True,
    )
    initial_before = layer.initial_lora_A.detach().clone()
    layer.apply_public_slot_regroup([[0, 1], [1, 2]])
    assert torch.equal(layer.initial_lora_A, initial_before)
    assert layer.primitive_indices.flatten().tolist() == [0, 1, 1, 2]


def test_public_mutated_base_matches_source_pissa_residual():
    base = nn.Linear(24, 32, bias=False)
    original = base.weight.detach().clone()
    layer = TFAbilityLinear(
        base,
        num_experts=2,
        rank=2,
        lora_alpha=8,
        top_k=2,
        dense_steps=0,
        router_mode="upstream_softmax",
        router_bias=False,
        svd_method="exact",
        residual_implementation="mutated_base",
    )
    assert torch.count_nonzero(layer.initial_lora_A) == 0
    expert_sum = layer._expert_weight(layer.lora_B.float(), layer.lora_A.float())
    torch.testing.assert_close(
        layer.base_layer.weight.float() + expert_sum,
        original.float(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert layer.initial_weight_relative_error.item() < 1e-5


def test_kaiming_zero_matches_classic_lora_initial_contract():
    torch.manual_seed(23)
    base = nn.Linear(24, 32, bias=False)
    original = base.weight.detach().clone()
    layer = TFAbilityLinear(
        base,
        num_experts=8,
        rank=4,
        lora_alpha=32,
        lora_dropout=0.05,
        top_k=4,
        dense_steps=0,
        router_mode="upstream_softmax",
        router_bias=False,
        initialization_method="kaiming_zero",
        apply_lora_dropout=False,
        residual_implementation="mutated_base",
    )
    assert layer.lora_A.shape == (8, 4, 24)
    assert layer.lora_B.shape == (8, 32, 4)
    assert torch.count_nonzero(layer.lora_A) > 0
    assert torch.count_nonzero(layer.lora_B) == 0
    assert torch.count_nonzero(layer.initial_lora_A) == 0
    assert torch.count_nonzero(layer.initial_lora_B) == 0
    assert isinstance(layer.dropout, nn.Identity)
    assert torch.equal(layer.base_layer.weight, original)
    assert layer.initial_weight_relative_error.item() == 0
    x = torch.randn(2, 5, 24)
    assert torch.equal(layer(x), layer.base_layer(x))


def test_uploaded_effective_dropout_switch_is_explicit():
    disabled = TFAbilityLinear(
        nn.Linear(12, 16, bias=False),
        num_experts=2,
        rank=2,
        top_k=2,
        lora_dropout=0.5,
        apply_lora_dropout=False,
    )
    enabled = TFAbilityLinear(
        nn.Linear(12, 16, bias=False),
        num_experts=2,
        rank=2,
        top_k=2,
        lora_dropout=0.5,
        apply_lora_dropout=True,
    )
    assert isinstance(disabled.dropout, nn.Identity)
    assert isinstance(enabled.dropout, nn.Dropout)


def test_kaiming_zero_virtual_gate_signal_is_zero_before_warmup():
    layer = TFAbilityLinear(
        nn.Linear(12, 16, bias=False),
        num_experts=8,
        rank=4,
        top_k=4,
        lora_dropout=0,
        initialization_method="kaiming_zero",
    )
    layer.enable_calibration()
    loss = layer(torch.randn(2, 3, 12)).square().mean()
    loss.backward()
    assert torch.count_nonzero(layer.primitive_gates.grad) == 0


def test_pissa_residual_is_not_base_preserving_under_simplex_routing():
    base = nn.Linear(12, 16, bias=False)
    original_weight = base.weight.detach().clone()
    layer = TFAbilityLinear(
        base,
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=4,
        dense_steps=0,
        router_mode="upstream_softmax",
        router_bias=False,
        initialization_method="svd",
        svd_method="exact",
        residual_implementation="mutated_base",
    )
    with torch.no_grad():
        layer.residual_router.weight.zero_()
    x = torch.randn(2, 3, 12)
    expected = F.linear(x, original_weight)
    observed = layer(x)
    assert layer.initial_weight_relative_error.item() < 1e-6
    assert not torch.allclose(observed, expected, rtol=1e-5, atol=1e-6)


def test_uploaded_loramoe_forward_and_backward_parity_without_regroup():
    """Match the uploaded LoraMoELinear path without importing its PEFT fork."""

    torch.manual_seed(19)
    layer = TFAbilityLinear(
        nn.Linear(12, 16, bias=False),
        num_experts=4,
        rank=2,
        lora_alpha=8,
        lora_dropout=0,
        top_k=2,
        dense_steps=0,
        router_mode="upstream_softmax",
        router_bias=False,
        svd_method="exact",
        residual_implementation="mutated_base",
    )
    x = torch.randn(3, 5, 12)
    target = torch.randn(3, 5, 16)

    reference_a = layer.lora_A.detach().clone().requires_grad_(True)
    reference_b = layer.lora_B.detach().clone().requires_grad_(True)
    reference_router = (
        layer.residual_router.weight.detach().clone().requires_grad_(True)
    )
    gate_logits = F.linear(x.mean(dim=-2), reference_router)
    gate_weights = F.softmax(gate_logits, dim=-1)
    top_weights, top_indices = torch.topk(gate_weights, layer.top_k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    mask = torch.zeros_like(gate_weights).scatter_(-1, top_indices, 1)
    selected_weights = top_weights.new_zeros(gate_weights.shape).scatter_(
        -1, top_indices, top_weights
    )
    assert torch.equal(mask != 0, selected_weights != 0)
    reference_intermediate = torch.einsum("bsi,kri->bskr", x, reference_a)
    reference_delta = torch.einsum(
        "bskr,bk,kor->bso", reference_intermediate, selected_weights, reference_b
    )
    reference_output = layer.base_layer(x) + reference_delta * layer.scaling

    observed_output = layer(x)
    torch.testing.assert_close(observed_output, reference_output, rtol=1e-6, atol=1e-6)
    observed_loss = (observed_output - target).square().mean()
    reference_loss = (reference_output - target).square().mean()
    observed_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        layer.lora_A.grad, reference_a.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        layer.lora_B.grad, reference_b.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        layer.residual_router.weight.grad,
        reference_router.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_regrouping_preserves_paired_primitive_sum():
    layer = _layer()
    before = layer.reconstructed_weight().detach().clone()
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    layer.apply_assignment(labels)
    after = layer.reconstructed_weight().detach()
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
    assert torch.equal(layer.assignment_matrix().sum(0), torch.full((4,), 2))


def test_fixed_residual_does_not_drift_when_experts_change():
    layer = _layer()
    initial = layer._expert_weight(
        layer.initial_lora_B.float(), layer.initial_lora_A.float()
    )
    residual = layer.base_layer.weight.float() - initial
    with torch.no_grad():
        layer.lora_A.add_(0.01)
    expected = residual + layer._expert_weight(
        layer.lora_B.float(), layer.lora_A.float()
    )
    assert torch.allclose(layer.reconstructed_weight(), expected)


def test_virtual_gates_have_per_primitive_gradients():
    layer = _layer()
    layer.enable_calibration()
    x = torch.randn(1, 3, 24)
    loss = layer(x).square().mean()
    (gradient,) = torch.autograd.grad(loss, layer.primitive_gates)
    assert gradient.shape == (4, 2)
    assert torch.isfinite(gradient).all()


def test_per_example_virtual_gate_override():
    layer = _layer()
    layer.enable_calibration()
    x = torch.randn(2, 3, 24)
    gates = torch.ones(2, 4, 2)
    gates[1, 0, 0] = 1.5
    layer.set_virtual_gate_override(gates)
    output = layer(x)
    layer.set_virtual_gate_override(None)
    baseline = layer(x)
    assert torch.equal(output[0], baseline[0])
    assert not torch.equal(output[1], baseline[1])
