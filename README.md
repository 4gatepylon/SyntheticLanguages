# WARNING: unfinished; not yet launched
# synthetic_languages
## Usage
Currently `synthetic_languages` is a `pip` installable. You can get it using `pip install synthetic_languages`.

For a tutorial on how to use this library check out our documentation: .

## What is this?
This is a library and set of tools to easily form synthetic languages of varying difficulties for the purpose of understanding the capabilities of the current generation of large language models.

Having a ground truth of some form can be very helpful when trying to do (mechanistic) interpretability (MI) on (large) language models. We do not know how to model the english language very deeply (at least AFAIK), so it is hard to get a good sense for what language models _might_ be learning to do: how they are actually _processing_ the input they receive to predict tokens.

Some research (https://transformer-circuits.pub/2023/monosemantic-features) has shown that it is possible to find features that correspond to known concepts, and others (i.e. https://openreview.net/forum?id=NpsVSN6o4ul) have found _circuits_ with mild success. Because MI is hard, people have also tried to do circuit analysis on algorithmic tasks (i.e. https://arxiv.org/pdf/2306.17844) and some made up sequences (i.e. https://www.youtube.com/watch?v=yo4QvDn-vsU&ab_channel=NeelNanda for the purposes of understanding the capabilities of transformers). This latter approach appears to be easier than doing MI directly on models of real language (for obvious reasons) but it's not as useful for understanding real world behavior.

What if we could bridge that gap? In computer vision we have datasets ranging from the most basic (MNIST, CIFAR) to more standard ones such as ImageNet and CoCo, as well as specific and realistic ones (though these may be priority in some cases). Generally, you can see that there is a ramp from simple problems to more complicated ones. Do we have such a situation in language? Sort of: algorithmic problems currently post as the "simple" ones, but it's not clear if there's such a clear ramp up into real language going through intermediate difficulty problems.

And thus our proposed solution: to create synthetic languages that scale as continuously as possible from very easy to understand (for an LLM) to very hard. What can be made here is open and collaboration is welcome, but there are two main goals:
1. That the languages enable us to learn with certainty an expediency what the LLM learned
2. That the languages induce LLM algorithms as close as possible to those learned from real world datasets
3. It should be as easy and automatable as possible to go from "can the LLM learn X pattern/algorithm?" to training a small-medium model on a custom dataset that isolates the question of "learning X".

Synthetic languages can be determinisic or non-determinisic and do not in fact even need to be synthetic. Some ideas to consider may include:
- Memoryless and/or time-invariant distributions over tokens
- Finite State Machines (FSMs) (Markov Chain) with markov probability distributions
- Trees of FSMs (basically, more complicated FSMs that help us model things like context or underlying traits like the emotions or goals/desires of the writer which may affect the distribution of tokens).
- Context-free grammars
- Subsets of english or human language using less words or somehow less concepts
- Animal communication

## What Can you Do Right Now?
The `synthetic_languages` package is very new and a work in progress. Currently you can:
1. Create time-invariant memoryless distributions with `numpy` arrays you define or by specifying the column entropy
2. Create markov chain distributions `numpy` arrays you define or by specifying the average entropy

## Future Directions
- Markov chain trees
- Context-free grammars
- More intermediate-levels of control over your languages
- High-To-Low-Level Compilation Pipeline support (where you create a simple, high-level language and then map a low-level language's phrases to those of the high-level language in a many-to-one, non-injective, surjective fashion)
- Automatic interpretability tooling
    - Integrate https://github.com/neelnanda-io/TransformerLens
    - NLP equivalent of https://github.com/ttumiel/interpret (generate inputs that maximize)
    - SAE and SAE Viz support: integrate https://github.com/callummcdougall/sae_vis
    - Algorithms to pattern match and search the LLM for weights or activation patterns that look like the generating algorithm or have some relation to it (open-ended, WIP)
- `word2vec` embedding strategy (and generally better embedding/tokenization)
- Languages that are subsets of existing languages (such as english), maybe even animal sounds or something else

Please feel free to collaborate and add new ideas! PRs are welcome! Help from linguists and others with deep language would also be appreciated.

# Documentation
We are using `mkdocstrings` and `mkdocs`. Deploying is as easy as `mkdocs gh-deploy`. Configuration and docs are in `docs/` and `mkdocs.yml`. Please read the tutorials (https://www.mkdocs.org/getting-started/ then https://mkdocstrings.github.io/python/usage/) to understand how this works.

# Setting up Environment
1. Install poetry
2. `poetry install`
3. Do your thing.

TODO(Adriano) find a solution to not working on Mac... (maybe it was that we needed cuda?)
