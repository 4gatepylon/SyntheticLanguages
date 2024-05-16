from collections import Counter
from typing import List

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore


def compute_block_entropy(seq: List[int], mx_blk_ln: int) -> np.ndarray:
    """Compute the block entropy for blocks of varying lengths."""
    block_entropies = []

    for L in range(1, mx_blk_ln + 1):
        # Create blocks of length L
        blocks = [tuple(seq[i : i + L]) for i in range(len(seq) - L + 1)]

        # Compute the probability distribution of the blocks
        block_counts = Counter(blocks)
        total_blocks = len(blocks)
        block_probs = {
            block: count / total_blocks for block, count in block_counts.items()
        }

        # Compute the block entropy H(L)
        entropy_L = -sum(p * np.log(p) for p in block_probs.values())
        block_entropies.append(entropy_L)

    return np.array(block_entropies)


def compute_cond_entropy(seq: List[int], mx_blk_ln: int) -> np.ndarray:
    """
    Compute the conditional entropy H(next symbol | previous L symbols)
    for varying L.
    """
    conditional_entropies = []

    # First, compute the block entropies for all required lengths (up to L+1)
    all_block_entropies = compute_block_entropy(seq, mx_blk_ln + 1)

    for L in range(1, mx_blk_ln + 1):
        joint_entropy = all_block_entropies[L]  # Joint entropy for L+1 symbols
        block_entropy = all_block_entropies[L - 1]  # Block entropy for L symbols

        # Compute conditional entropy
        conditional_entropy = joint_entropy - block_entropy
        conditional_entropies.append(conditional_entropy)

    return np.array(conditional_entropies)


def compute_empirical_cond_entropy(seq: List[int], mx_blk_ln: int) -> List[float]:
    """
    Compute the empirical conditional entropy H(next symbol | previous L symbols)
    for varying L.
    """
    NUM_SYMBOLS = 2
    conditional_entropies = []

    for L in range(1, mx_blk_ln + 1):
        # Dictionary to store counts of observed blocks of length L followed by a symbol
        block_followed_by_symbol_counts = Counter(
            [(tuple(seq[i : i + L]), seq[i + L]) for i in range(len(seq) - L)]
        )

        # Dictionary to store counts of observed blocks of length L
        block_counts = Counter([tuple(seq[i : i + L]) for i in range(len(seq) - L + 1)])

        # Conditional entropy computation
        entropy = 0.0
        for block, block_count in block_counts.items():
            for symbol in range(NUM_SYMBOLS):
                # Empirical conditional probability p(symbol | block)
                conditional_prob = (
                    block_followed_by_symbol_counts.get((block, symbol), 0)
                    / block_count
                )
                if conditional_prob > 0:
                    entropy -= (
                        (block_count / len(seq))
                        * conditional_prob
                        * np.log(conditional_prob)
                    )

        conditional_entropies.append(entropy)

    return conditional_entropies


def binary_entropy(p: float) -> float:
    """Compute the binary entropy for a given probability p."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def inverse_binary_entropy(target_entropy: float) -> float:
    """Find the probability p corresponding to a given binary entropy value."""

    # Objective function: the difference between target entropy and binary entropy of p
    def objective(p: float):
        return (binary_entropy(p) - target_entropy) ** 2

    # Minimize the objective function to find p
    result = minimize_scalar(objective, bounds=(0, 0.5), method="bounded")

    return result.x
