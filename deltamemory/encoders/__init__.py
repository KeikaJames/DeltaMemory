"""Stage 9 address-encoder variants.

The Stage 8 v3 baseline mean-pools static input embeddings of the address
tokens. That representation is structurally limited: at N=4096 retrieval
recall@1 saturates around 0.83 regardless of KeyProjector tuning. This
module provides richer encoders intended to break that ceiling.

Encoders all expose:
    encode(model, tokenizer, addresses, prompts) -> Tensor (B, H_addr)

where prompts is the read prompt for each fact (used by encoders that
need query-side context from the frozen base model).
"""

from .address_encoders import (
    AddressEncoder,
    MeanPoolEncoder,
    AttnPoolEncoder,
    MultiLayerEncoder,
    PromptHiddenEncoder,
    ResidualMLPEncoder,
    build_encoder,
)

__all__ = [
    "AddressEncoder",
    "MeanPoolEncoder",
    "AttnPoolEncoder",
    "MultiLayerEncoder",
    "PromptHiddenEncoder",
    "ResidualMLPEncoder",
    "build_encoder",
]
