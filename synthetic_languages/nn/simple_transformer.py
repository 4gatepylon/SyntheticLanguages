import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO(Adriano) dumb this in favor of existing hooked transformers and the like!
class Head(nn.Module):
    def __init__(self, input_size: int, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(input_size, input_size)))

    def forward(self, x: torch.Tensor):
        # x is of size (batch_size, input_size, d_model)
        # get the query, key, and value
        Q = self.W_Q(x)  # (batch_size, input_size, d_head)
        K = self.W_K(x)  # (batch_size, input_size, d_head)
        V = self.W_V(x)  # (batch_size, input_size, d_head)
        # get the attention weights
        A = torch.einsum("bid,bjd->bij", Q, K) / (self.d_head**0.5)
        A = A.masked_fill(self.mask == 0, float("-inf"))
        A = F.softmax(A, dim=-1)  # the rows of A sum to 1
        # apply the attention weights

        # this is the output of the attention head, we weight the values by
        # the attention weights
        return torch.einsum("bij,bjd->bid", A, V)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor):
        # x is of size (batch_size, input_size, d_model)
        x = self.W_in(x)  # (batch_size, input_size, d_mlp)
        x = F.relu(x)
        x = self.W_out(x)  # (batch_size, input_size, d_model)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        input_size: int,
        d_head: int,
        n_head: int,
        d_mlp: int,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.heads = nn.ModuleList(
            [Head(input_size, d_model, d_head) for _ in range(n_head)]
        )
        self.mlp = MLP(d_model, d_mlp)
        self.W_O = nn.Linear(n_head * d_head, d_model, bias=False)

        # Add Layer Normalization layers
        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # apply the attention heads, stack them
        head_output = torch.cat([head(x) for head in self.heads], dim=-1)

        # Apply normalization and residual connection
        if self.use_layernorm:
            x = x + self.norm1(self.W_O(head_output))
        else:
            x = x + self.W_O(head_output)

        # apply the MLP
        if self.use_layernorm:
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.mlp(x)

        return x


class MultilayerTransformer(nn.Module):
    def __init__(
        self,
        d_vocab: int = 2,
        d_model: int = 16,
        input_size: int = 3,
        d_head: int = 4,
        n_head: int = 4,
        d_mlp: int = 4 * 16,
        n_layers: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model, input_size, d_head, n_head, d_mlp, use_layernorm
                )
                for _ in range(n_layers)
            ]
        )
        self.unembedding = nn.Linear(d_model, d_vocab)
        self.hooks = {}
        self.current_batch = 0

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        # x is of size (batch_size, input_size)
        # embed the input
        x = self.embedding(x) + self.pos_embedding(
            torch.arange(self.input_size, device=x.device)
        )
        activations = []
        # pass through each transformer layer
        for layer in self.layers:
            x = layer(x)
            if return_activations:
                activations.append(x.detach())
        # unembed the output
        x = self.unembedding(x)
        if return_activations:
            return x, activations
        else:
            return x

    def predict_probs(self, x: torch.Tensor):
        # pass input through the model
        logits = self.forward(x)
        assert isinstance(logits, torch.Tensor)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs


def initialize_weights(module: nn.Module):
    """Initialize the weights of the Transformer as per the original paper."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
