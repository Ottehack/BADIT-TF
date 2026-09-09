"""Shared primitive bank and residual routing used by TF and matched GG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AbilityInjectionResult:
    layers: Mapping[str, "TFAbilityLinear"]
    reconstruction_errors: Mapping[str, float]


def _child_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    path = qualified_name.split(".")
    parent = root
    for component in path[:-1]:
        parent = getattr(parent, component)
    return parent, path[-1]


class TFAbilityLinear(nn.Module):
    """A frozen residual plus K rank-r experts and one residual router.

    The layer has three explicit modes:

    * initial/training dense mode: code = 1 + R(h), no top-k;
    * normal routed mode: binary top-k mask times the unchanged dense code;
    * calibration mode: one independent virtual gate per primitive.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        num_experts: int = 8,
        rank: int = 4,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        top_k: int = 4,
        dense_steps: int = 1,
        router_mode: str = "residual_mask",
        router_bias: bool = True,
        initialization_method: str = "svd",
        apply_lora_dropout: bool = True,
        svd_method: str = "randomized",
        residual_implementation: str = "paired_subtraction",
        cache_last_gradients: bool = False,
        allow_unequal_capacity: bool = False,
        assignment: torch.Tensor | None = None,
        svd_seed: int = 1,
        svd_oversample: int = 8,
        svd_niter: int = 4,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("TFAbilityLinear supports torch.nn.Linear only")
        if rank <= 0 or num_experts <= 0:
            raise ValueError("rank and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if dense_steps < 0:
            raise ValueError("dense_steps must be nonnegative")
        if router_mode not in {"residual_mask", "upstream_softmax"}:
            raise ValueError(f"unsupported router_mode={router_mode}")
        if initialization_method not in {"svd", "kaiming_zero"}:
            raise ValueError(
                f"unsupported initialization_method={initialization_method}"
            )
        if svd_method not in {"randomized", "exact"}:
            raise ValueError(f"unsupported svd_method={svd_method}")
        if residual_implementation not in {"paired_subtraction", "mutated_base"}:
            raise ValueError(
                f"unsupported residual_implementation={residual_implementation}"
            )

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.total_primitives = self.num_experts * self.rank
        self.allow_unequal_capacity = bool(allow_unequal_capacity)
        self.lora_alpha = float(lora_alpha)
        self.scaling = float(lora_alpha / rank)
        self.top_k = int(top_k)
        self.dense_steps = int(dense_steps)
        self.router_mode = str(router_mode)
        self.initialization_method = str(initialization_method)
        self.apply_lora_dropout = bool(apply_lora_dropout)
        self.svd_method = str(svd_method)
        self.residual_implementation = str(residual_implementation)
        self.dropout = (
            nn.Dropout(float(lora_dropout))
            if lora_dropout > 0 and self.apply_lora_dropout
            else nn.Identity()
        )

        original_dtype = base_layer.weight.dtype
        original_device = base_layer.weight.device
        # ``Tensor.to`` may return the same storage when the source is already
        # float32.  Clone before mutating ``base_layer.weight`` into W_res.
        original_weight = (
            base_layer.weight.detach().to(dtype=torch.float32).clone()
        )
        labels = self._normalized_assignment(assignment, original_device)
        if self.initialization_method == "svd":
            primitive_left, primitive_right, singular_values = self.extract_primitives(
                original_weight,
                self.total_primitives,
                scaling=self.scaling,
                method=self.svd_method,
                seed=svd_seed,
                oversample=svd_oversample,
                niter=svd_niter,
            )
            grouped_left, grouped_right, primitive_indices = self._group_primitives(
                primitive_left, primitive_right, labels
            )
        else:
            # Match PEFT's classic LoRA initialization independently for each
            # expert: A is Kaiming-uniform with a=sqrt(5), B is exactly zero.
            # Create in fp32, then cast with the containing adapter module.
            grouped_right = torch.empty(
                self.num_experts,
                self.rank,
                self.in_features,
                dtype=torch.float32,
                device=original_device,
            )
            for expert in range(self.num_experts):
                nn.init.kaiming_uniform_(grouped_right[expert], a=math.sqrt(5))
            grouped_left = torch.zeros(
                self.num_experts,
                self.out_features,
                self.rank,
                dtype=torch.float32,
                device=original_device,
            )
            primitive_indices = torch.stack(
                [
                    torch.nonzero(labels == expert, as_tuple=False).flatten()
                    for expert in range(self.num_experts)
                ],
                dim=0,
            )
            singular_values = torch.zeros(
                self.total_primitives,
                dtype=torch.float32,
                device=original_device,
            )

        # Quantize factors before freezing the initial primitive bank.  The
        # base layer remains W0.  Forward evaluates
        #
        #   W0 x - sum(E_init x) + sum(c_k E_k x),
        #
        # which is exactly W_res x + sum(c_k E_k x) with a fixed
        # W_res = W0 - sum(E_init).  At all-one initialization the two
        # identical low-rank paths cancel bit-for-bit even in bf16.
        grouped_left = grouped_left.to(dtype=original_dtype, device=original_device)
        grouped_right = grouped_right.to(
            dtype=original_dtype, device=original_device
        )
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.lora_B = nn.Parameter(grouped_left)
        self.lora_A = nn.Parameter(grouped_right)
        if self.initialization_method == "kaiming_zero":
            initial_left = torch.zeros_like(grouped_left)
            initial_right = torch.zeros_like(grouped_right)
        elif self.residual_implementation == "mutated_base":
            initial_left = torch.zeros_like(grouped_left)
            initial_right = torch.zeros_like(grouped_right)
            reconstructed = self._expert_weight(grouped_left, grouped_right)
            with torch.no_grad():
                self.base_layer.weight.copy_(
                    (original_weight - reconstructed.float()).to(original_dtype)
                )
        else:
            initial_left = grouped_left.detach().clone()
            initial_right = grouped_right.detach().clone()
        self.register_buffer("initial_lora_B", initial_left)
        self.register_buffer("initial_lora_A", initial_right)
        self.residual_router = nn.Linear(
            self.in_features,
            self.num_experts,
            bias=bool(router_bias),
            device=original_device,
            dtype=original_dtype,
        )
        if self.router_mode == "residual_mask":
            nn.init.zeros_(self.residual_router.weight)
            if self.residual_router.bias is not None:
                nn.init.zeros_(self.residual_router.bias)

        self._last_gradient_cache: dict[str, torch.Tensor] = {}
        if cache_last_gradients:
            self.lora_A.register_hook(self._cache_gradient("lora_A"))
            self.lora_B.register_hook(self._cache_gradient("lora_B"))

        self.primitive_gates = nn.Parameter(
            torch.ones(
                self.num_experts,
                self.rank,
                device=original_device,
                dtype=original_dtype,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "primitive_indices", primitive_indices.to(device=original_device)
        )
        # Physical factor slots remain K x r even for the Table VI diagnostic.
        # This buffer gives each physical primitive its routed expert label;
        # unequal labels are evaluated per primitive without padding or adding
        # trainable parameters.
        physical_labels = torch.arange(
            self.num_experts, device=original_device, dtype=torch.long
        ).repeat_interleave(self.rank).reshape(self.num_experts, self.rank)
        self.register_buffer("routing_assignment_labels", physical_labels)
        self.register_buffer(
            "singular_values",
            singular_values.to(device=original_device, dtype=torch.float32),
        )
        self.register_buffer(
            "topk_enabled", torch.tensor(dense_steps == 0, device=original_device)
        )
        self.register_buffer(
            "calibration_enabled", torch.tensor(False, device=original_device)
        )
        self.register_buffer(
            "optimizer_updates",
            torch.tensor(0, dtype=torch.long, device=original_device),
        )

        stored_reconstruction = self.reconstructed_weight().to(torch.float32)
        relative_error = (
            torch.linalg.vector_norm(stored_reconstruction - original_weight)
            / torch.linalg.vector_norm(original_weight).clamp_min(1e-30)
        )
        self.register_buffer(
            "initial_weight_relative_error", relative_error.detach()
        )
        self._route_attention_mask: torch.Tensor | None = None
        self._virtual_gate_override: torch.Tensor | None = None
        self._route_code_override: torch.Tensor | None = None
        self._last_route: dict[str, torch.Tensor] = {}

    def _cache_gradient(self, name: str):
        def hook(gradient: torch.Tensor) -> None:
            self._last_gradient_cache[name] = gradient.detach().clone()

        return hook

    @staticmethod
    def extract_primitives(
        weight: torch.Tensor,
        total_rank: int,
        *,
        scaling: float,
        method: str = "randomized",
        seed: int,
        oversample: int,
        niter: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return left/right factors whose scaled products are SVD primitives."""

        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        if total_rank > min(weight.shape):
            raise ValueError(
                f"total rank {total_rank} exceeds matrix limit {min(weight.shape)}"
            )
        if scaling <= 0:
            raise ValueError("scaling must be positive")
        if method == "exact":
            left, singular, right_h = torch.linalg.svd(
                weight, full_matrices=False
            )
            left = left[:, :total_rank]
            singular = singular[:total_rank]
            right_h = right_h[:total_rank]
        elif method == "randomized":
            q = min(min(weight.shape), total_rank + max(0, int(oversample)))
            devices = [weight.device] if weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                if weight.is_cuda:
                    torch.cuda.manual_seed_all(int(seed))
                left, singular, right = torch.svd_lowrank(
                    weight, q=q, niter=int(niter)
                )
            order = torch.argsort(singular, descending=True)[:total_rank]
            left = left[:, order]
            singular = singular[order]
            right_h = right[:, order].mT
        else:
            raise ValueError(f"unsupported SVD method={method}")
        root = torch.sqrt(singular / float(scaling))
        primitive_left = left * root.unsqueeze(0)
        primitive_right = root.unsqueeze(1) * right_h
        return primitive_left, primitive_right, singular

    def _normalized_assignment(
        self, assignment: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if assignment is None:
            labels = torch.arange(self.total_primitives, device=device) // self.rank
        else:
            labels = torch.as_tensor(assignment, dtype=torch.long, device=device)
        if labels.shape != (self.total_primitives,):
            raise ValueError(
                f"assignment must have shape ({self.total_primitives},)"
            )
        counts = torch.bincount(labels, minlength=self.num_experts)
        expected = torch.full_like(counts, self.rank)
        if labels.min() < 0 or labels.max() >= self.num_experts:
            raise ValueError("assignment contains an invalid expert")
        if self.allow_unequal_capacity and torch.any(counts == 0):
            raise ValueError(
                "assignment nonempty constraint mismatch: "
                f"{counts.detach().cpu().tolist()}"
            )
        if not self.allow_unequal_capacity and not torch.equal(counts, expected):
            raise ValueError(
                f"assignment capacity mismatch: {counts.detach().cpu().tolist()}"
            )
        return labels

    def _group_primitives(
        self,
        primitive_left: torch.Tensor,
        primitive_right: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_groups = []
        right_groups = []
        index_groups = []
        for expert in range(self.num_experts):
            indices = torch.nonzero(labels == expert, as_tuple=False).flatten()
            if indices.numel() != self.rank:
                raise AssertionError("assignment is not equal capacity")
            left_groups.append(primitive_left[:, indices])
            right_groups.append(primitive_right[indices, :])
            index_groups.append(indices)
        # B: [K, out, r], A: [K, r, in]
        return (
            torch.stack(left_groups, dim=0),
            torch.stack(right_groups, dim=0),
            torch.stack(index_groups, dim=0),
        )

    def _expert_weight(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return self.scaling * torch.einsum("kor,kri->oi", left, right)

    def reconstructed_weight(self) -> torch.Tensor:
        initial = self._expert_weight(
            self.initial_lora_B.to(torch.float32),
            self.initial_lora_A.to(torch.float32),
        )
        residual = self.base_layer.weight.to(torch.float32) - initial
        current = self._expert_weight(
            self.lora_B.to(torch.float32), self.lora_A.to(torch.float32)
        )
        return residual + current

    def assignment_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(
            self.total_primitives,
            self.num_experts,
            dtype=torch.int8,
            device=self.primitive_indices.device,
        )
        if self.allow_unequal_capacity:
            matrix[
                self.primitive_indices.flatten(),
                self.routing_assignment_labels.flatten(),
            ] = 1
        else:
            for expert in range(self.num_experts):
                matrix[self.primitive_indices[expert], expert] = 1
        return matrix

    def apply_assignment(self, labels: torch.Tensor) -> None:
        """Regroup paired current/initial A rows and B columns.

        The initial factors are the fixed subtraction path representing
        ``W_res``.  Dynamic GG regrouping must move them with the trainable
        factors, otherwise an expert would mix current primitives with the
        wrong initial primitives after regrouping.
        """

        labels = self._normalized_assignment(labels, self.lora_A.device)
        if self.allow_unequal_capacity:
            with torch.no_grad():
                self.routing_assignment_labels.copy_(labels[self.primitive_indices])
            return
        flat_indices = self.primitive_indices.flatten()
        inverse = torch.argsort(flat_indices)

        def regroup_pair(
            left: torch.Tensor, right: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat_left = left.detach().permute(1, 0, 2).reshape(
                self.out_features, self.total_primitives
            )
            flat_right = right.detach().reshape(
                self.total_primitives, self.in_features
            )
            return self._group_primitives(
                flat_left[:, inverse], flat_right[inverse], labels
            )

        new_left, new_right, new_indices = regroup_pair(
            self.lora_B, self.lora_A
        )
        new_initial_left, new_initial_right, initial_indices = regroup_pair(
            self.initial_lora_B, self.initial_lora_A
        )
        if not torch.equal(new_indices, initial_indices):
            raise AssertionError("current/initial regrouping index mismatch")
        with torch.no_grad():
            self.lora_B.copy_(new_left)
            self.lora_A.copy_(new_right)
            self.initial_lora_B.copy_(new_initial_left)
            self.initial_lora_A.copy_(new_initial_right)
            self.primitive_indices.copy_(new_indices)
            self.routing_assignment_labels.copy_(
                torch.arange(
                    self.num_experts,
                    device=self.routing_assignment_labels.device,
                    dtype=self.routing_assignment_labels.dtype,
                )
                .repeat_interleave(self.rank)
                .reshape(self.num_experts, self.rank)
            )

    def cached_public_gradient_vectors(self) -> torch.Tensor:
        """Return the public BADIT per-slot last-backward gradient vectors."""

        if set(self._last_gradient_cache) != {"lora_A", "lora_B"}:
            raise RuntimeError("public regroup gradient cache is incomplete")
        grad_a = self._last_gradient_cache["lora_A"].float()
        grad_b = self._last_gradient_cache["lora_B"].float()
        expected_a = (self.num_experts, self.rank, self.in_features)
        expected_b = (self.num_experts, self.out_features, self.rank)
        if tuple(grad_a.shape) != expected_a or tuple(grad_b.shape) != expected_b:
            raise RuntimeError(
                f"public regroup gradient shape mismatch: "
                f"A={tuple(grad_a.shape)}, B={tuple(grad_b.shape)}"
            )
        vectors = []
        for expert in range(self.num_experts):
            for primitive in range(self.rank):
                vectors.append(
                    torch.cat(
                        [grad_a[expert, primitive], grad_b[expert, :, primitive]]
                    )
                )
        return torch.stack(vectors)

    def apply_public_slot_regroup(self, clusters: list[list[int]]) -> None:
        """Apply the public repository's slot-copy regroup literally.

        Unlike :meth:`apply_assignment`, this intentionally changes only the
        trainable A/B slots.  The fixed PiSSA residual stays unchanged, as it
        does in the public implementation.  Duplicate or omitted slots are
        retained in ``primitive_indices`` so the audit can expose them.
        """

        if len(clusters) != self.num_experts or any(
            len(cluster) != self.rank for cluster in clusters
        ):
            raise ValueError("public regroup must provide rank slots per expert")
        flat_a = self.lora_A.detach().reshape(
            self.total_primitives, self.in_features
        )
        flat_b = self.lora_B.detach().permute(0, 2, 1).reshape(
            self.total_primitives, self.out_features
        )
        flat_ids = self.primitive_indices.detach().reshape(-1)
        indices = torch.tensor(
            [index for cluster in clusters for index in cluster],
            dtype=torch.long,
            device=self.lora_A.device,
        )
        if torch.any(indices < 0) or torch.any(indices >= self.total_primitives):
            raise ValueError("public regroup slot index out of range")
        with torch.no_grad():
            self.lora_A.copy_(
                flat_a[indices].reshape(
                    self.num_experts, self.rank, self.in_features
                )
            )
            self.lora_B.copy_(
                flat_b[indices]
                .reshape(self.num_experts, self.rank, self.out_features)
                .permute(0, 2, 1)
            )
            self.primitive_indices.copy_(
                flat_ids[indices].reshape(self.num_experts, self.rank)
            )
        self._last_gradient_cache.clear()

    def set_attention_mask(self, mask: torch.Tensor | None) -> None:
        self._route_attention_mask = mask

    def set_route_code_override(self, code: torch.Tensor | None) -> None:
        self._route_code_override = None if code is None else code.detach()

    def _sequence_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim != 3:
            raise ValueError(f"expected 2-D or 3-D input, got {x.shape}")
        mask = self._route_attention_mask
        if mask is None or mask.shape[-1] != x.shape[-2]:
            return x.mean(dim=-2)
        mask = mask.to(device=x.device, dtype=x.dtype)
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return (x * mask.unsqueeze(-1)).sum(dim=-2) / denominator

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self._sequence_pool(x)
        raw = self.residual_router(pooled)
        if self.router_mode == "residual_mask":
            dense = torch.ones_like(raw) + raw
        else:
            # Diagnostic parity with the public BADIT implementation: gate
            # logits are converted to simplex weights before top-k, and the
            # retained weights are renormalized.  BADIT-TF formal runs keep
            # the default residual_mask mode; this branch is out-of-method.
            dense = F.softmax(raw, dim=-1)
        if self._route_code_override is not None:
            actual = self._route_code_override.to(
                device=dense.device, dtype=dense.dtype
            )
            if actual.shape != dense.shape:
                raise ValueError(
                    f"route override shape {actual.shape} != {dense.shape}"
                )
        elif bool(self.topk_enabled.item()):
            # ``torch.topk`` does not promise an ordering for equal values.
            # A zero-initialized residual router therefore produced
            # device-dependent expert masks at the first sparse step.  Stable
            # sorting fixes the tie policy to ascending expert ID while
            # preserving the exact top-k set whenever scores differ.
            indices = torch.argsort(
                dense, dim=-1, descending=True, stable=True
            )[..., : self.top_k]
            mask = torch.zeros_like(dense).scatter_(-1, indices, 1)
            actual = dense * mask
            if self.router_mode == "upstream_softmax":
                actual = actual / actual.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(actual.dtype).tiny
                )
        else:
            actual = dense
        self._last_route = {
            "raw": raw.detach(),
            "dense": dense.detach(),
            "actual": actual.detach(),
        }
        return raw, dense, actual

    def enable_calibration(self) -> None:
        self.calibration_enabled.fill_(True)
        self.topk_enabled.fill_(False)
        self.primitive_gates.requires_grad_(True)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        for parameter in self.residual_router.parameters():
            parameter.requires_grad_(False)

    def disable_calibration(self) -> None:
        self.calibration_enabled.fill_(False)
        self.primitive_gates.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        for parameter in self.residual_router.parameters():
            parameter.requires_grad_(True)

    def reset_virtual_gates(self) -> None:
        with torch.no_grad():
            self.primitive_gates.fill_(1)

    def set_virtual_gate_override(self, gates: torch.Tensor | None) -> None:
        if gates is not None:
            expected_tail = (self.num_experts, self.rank)
            if gates.ndim != 3 or tuple(gates.shape[1:]) != expected_tail:
                raise ValueError(
                    "virtual gate override must have shape "
                    f"[batch, {self.num_experts}, {self.rank}]"
                )
        self._virtual_gate_override = gates

    def advance_optimizer_step(self) -> None:
        self.optimizer_updates.add_(1)
        if int(self.optimizer_updates.item()) >= self.dense_steps:
            self.topk_enabled.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_ndim = x.ndim
        if original_ndim == 2:
            x_work = x.unsqueeze(1)
        elif original_ndim == 3:
            x_work = x
        else:
            raise ValueError(f"expected 2-D or 3-D input, got {x.shape}")

        base_output = self.base_layer(x)
        dropped = self.dropout(x_work)
        intermediate = torch.einsum("bsi,kri->bskr", dropped, self.lora_A)
        initial_intermediate = None
        if self.residual_implementation == "paired_subtraction":
            initial_intermediate = torch.einsum(
                "bsi,kri->bskr", dropped, self.initial_lora_A
            )
        if bool(self.calibration_enabled.item()):
            if self._virtual_gate_override is None:
                current_delta = torch.einsum(
                    "bskr,kr,kor->bso",
                    intermediate,
                    self.primitive_gates,
                    self.lora_B,
                )
            else:
                if self._virtual_gate_override.shape[0] != x_work.shape[0]:
                    raise ValueError("virtual gate override batch mismatch")
                current_delta = torch.einsum(
                    "bskr,bkr,kor->bso",
                    intermediate,
                    self._virtual_gate_override.to(
                        device=x_work.device, dtype=x_work.dtype
                    ),
                    self.lora_B,
                )
        else:
            _, _, code = self.route(x_work)
            if self.allow_unequal_capacity:
                primitive_intermediate = intermediate.reshape(
                    x_work.shape[0], x_work.shape[1], self.total_primitives
                )
                primitive_left = self.lora_B.permute(0, 2, 1).reshape(
                    self.total_primitives, self.out_features
                )
                primitive_code = code[:, self.routing_assignment_labels.flatten()]
                current_delta = torch.einsum(
                    "bsm,bm,mo->bso",
                    primitive_intermediate,
                    primitive_code,
                    primitive_left,
                )
            else:
                current_delta = torch.einsum(
                    "bskr,bk,kor->bso", intermediate, code, self.lora_B
                )
        if initial_intermediate is None:
            initial_delta = torch.zeros_like(current_delta)
        else:
            initial_delta = torch.einsum(
                "bskr,kor->bso", initial_intermediate, self.initial_lora_B
            )
        delta = (current_delta - initial_delta) * self.scaling
        if original_ndim == 2:
            delta = delta.squeeze(1)
        return base_output + delta.to(base_output.dtype)


def iter_ability_layers(model: nn.Module) -> Iterable[tuple[str, TFAbilityLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, TFAbilityLinear):
            yield name, module


def inject_ability_layers(
    model: nn.Module,
    *,
    target_suffixes: tuple[str, ...] = ("gate_proj",),
    assignments: Mapping[str, torch.Tensor] | None = None,
    **layer_kwargs,
) -> AbilityInjectionResult:
    assignments = assignments or {}
    base_svd_seed = int(layer_kwargs.pop("svd_seed", 1))
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    if not candidates:
        raise ValueError(f"no Linear target matched suffixes {target_suffixes}")
    injected: dict[str, TFAbilityLinear] = {}
    errors: dict[str, float] = {}
    for index, (name, module) in enumerate(candidates):
        parent, attribute = _child_module(model, name)
        layer = TFAbilityLinear(
            module,
            assignment=assignments.get(name),
            svd_seed=base_svd_seed + index,
            **layer_kwargs,
        )
        setattr(parent, attribute, layer)
        injected[name] = layer
        errors[name] = float(layer.initial_weight_relative_error.item())
    return AbilityInjectionResult(layers=injected, reconstruction_errors=errors)


def advance_router_step(model: nn.Module) -> None:
    for _, layer in iter_ability_layers(model):
        layer.advance_optimizer_step()
