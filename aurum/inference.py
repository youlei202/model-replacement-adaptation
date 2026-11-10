"""Inference-only AURUM replay; no teacher logits, fitting labels or model loading."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from .models import FrozenAnchor


def predict_decoded(state: dict, peer_logits: Tensor, features: Tensor | None = None,
                    basis: Tensor | None = None) -> Tensor:
    family = state["family"]
    if (peer_logits.ndim != 3 or peer_logits.shape[1] != len(state["peers"])
            or peer_logits.shape[-1] != state["vocab"]):
        raise ValueError("Peer count/vocabulary does not match encoded state")
    a = [torch.as_tensor(x, device=peer_logits.device, dtype=peer_logits.dtype)
         for x in state["arrays"]]
    if not a or a[0].shape != (len(state["peers"]),):
        raise ValueError("Invalid anchor coefficients")
    kind = "probability" if family.endswith("probability") else "logit"
    anchor = FrozenAnchor(a[0], kind)
    if family.startswith("anchor_"):
        if len(a) != 1:
            raise ValueError("Unexpected anchor payload")
        return anchor(peer_logits)
    if features is None or features.shape != (len(peer_logits), state["feature_dim"]):
        raise ValueError("Feature transform does not match encoded metadata")
    if family.startswith("gate_"):
        if len(a) != 4 or a[1].shape != (1, state["feature_dim"]) or a[2].shape != (1,) or a[3].shape != (1,):
            raise ValueError("Invalid gate arrays")
        if len(state["peers"]) != 2 or a[3].item() <= 0:
            raise ValueError("Two peers and positive encoded gate strength required")
        gate = a[3][0] * torch.tanh(features @ a[1].T + a[2])
        correction = gate * (peer_logits[:, 0] - peer_logits[:, 1])
    else:
        rank = state["basis_rank"]
        if basis is None or basis.shape != (state["vocab"], rank):
            raise ValueError("Public output basis does not match state metadata")
        if family.startswith("linear_"):
            if len(a) != 3 or a[1].shape != (rank,) or a[2].shape != (state["feature_dim"], rank):
                raise ValueError("Invalid linear residual arrays")
            low = features @ a[2] + a[1]
        elif family.startswith("factored_"):
            if (len(a) != 4 or a[1].shape != (rank,) or a[2].ndim != 2
                    or a[2].shape[0] != state["feature_dim"] or a[3].shape != (a[2].shape[1], rank)):
                raise ValueError("Invalid factored residual arrays")
            low = (features @ a[2]) @ a[3] + a[1]
        elif family.startswith("mlp_"):
            if (len(a) != 5 or a[1].ndim != 2 or a[1].shape[1] != state["feature_dim"]
                    or a[2].shape != (a[1].shape[0],) or a[3].shape != (rank, a[1].shape[0])
                    or a[4].shape != (rank,)):
                raise ValueError("Invalid MLP arrays")
            low = torch.tanh(features @ a[1].T + a[2]) @ a[3].T + a[4]
        else:
            raise ValueError("Unknown encoded predictor family")
        correction = low @ basis.T
    return torch.log_softmax(anchor.logits(peer_logits) + correction, -1)


class DecodedPredictor(nn.Module):
    """Evaluation adapter retaining only the decoded state and public basis."""

    def __init__(self, state: dict, basis: Tensor | None = None, *, device=None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.state = state
        self.register_buffer("basis", None if basis is None else basis.detach())
        reference = basis if basis is not None else torch.empty(0, device=device, dtype=dtype)
        self.register_buffer("device_reference", reference.new_empty(0))

    def forward(self, peers: Tensor, features: Tensor | None = None) -> Tensor:
        return predict_decoded(self.state, peers, features, self.basis)
