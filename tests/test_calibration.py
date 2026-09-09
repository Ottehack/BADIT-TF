import numpy as np
import torch

from badit_tf.calibration import (
    per_sequence_nll,
    supervised_route_attention_mask,
    tied_gate_vector,
)


def test_per_sequence_nll_masks_prompt():
    logits = torch.tensor(
        [
            [[0.0, 8.0], [8.0, 0.0], [8.0, 0.0]],
            [[8.0, 0.0], [0.0, 8.0], [0.0, 8.0]],
        ]
    )
    labels = torch.tensor([[-100, 1, 0], [-100, 0, 1]])
    losses = per_sequence_nll(logits, labels)
    assert losses.shape == (2,)
    assert torch.all(losses < 0.001)


def test_per_sequence_nll_matches_flatten_sum_over_count():
    torch.manual_seed(19)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor(
        [
            [-100, -100, 2, 3, 4],
            [-100, 5, 6, -100, 7],
        ]
    )
    observed = per_sequence_nll(logits, labels)
    expected = []
    for index in range(2):
        shifted_logits = logits[index, :-1].float()
        shifted_labels = labels[index, 1:]
        expected.append(
            torch.nn.functional.cross_entropy(
                shifted_logits,
                shifted_labels,
                ignore_index=-100,
                reduction="sum",
            )
            / (shifted_labels != -100).sum()
        )
    assert torch.equal(observed, torch.stack(expected))


def test_tied_gate_is_fisher_weighted():
    q = np.asarray([2.0, 6.0, 3.0, 5.0])
    fisher = np.asarray([1.0, 3.0, 1.0, 1.0])
    labels = np.asarray([0, 0, 1, 1])
    gates, regret = tied_gate_vector(
        q, fisher, labels, eta=0.1, num_experts=2
    )
    assert np.allclose(gates[:2], 1.2)
    assert np.allclose(gates[2:], 1.4)
    assert regret >= 0


def test_prompt_only_route_mask_excludes_teacher_forced_target():
    batch = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, -100, 7, 8]]),
    }
    observed = supervised_route_attention_mask(batch, scope="prompt_only")
    assert torch.equal(observed, torch.tensor([[1, 1, 1, 0, 0]]))
    legacy = supervised_route_attention_mask(
        batch, scope="full_supervised_sequence"
    )
    assert torch.equal(legacy, batch["attention_mask"])
