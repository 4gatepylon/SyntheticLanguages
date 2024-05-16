from typing import Optional, Union

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig  # type: ignore

from synthetic_languages.training.configs.base_config import Config


class RawModelConfig(Config):
    d_vocab: int
    d_model: int
    n_ctx: int
    d_head: int
    n_head: int
    d_mlp: int
    n_layers: int

    def to_hooked_transformer(
        self, device: Optional[Union[str, torch.device]], seed: Optional[int] = None
    ) -> HookedTransformer:
        device = str(device) if isinstance(device, torch.device) else device
        assert device is None or isinstance(device, str)
        config = HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            n_ctx=self.n_ctx,
            n_heads=self.n_head,
            d_mlp=self.d_mlp,
            d_vocab=self.d_vocab,
            seed=seed,
            device=device,
            act_fn="relu",
        )
        return HookedTransformer(config)
