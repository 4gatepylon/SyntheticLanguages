# synthetic_languages
## Usage
Currently `synthetic_languages` is a `pip` installable. You can get it using `pip install synthetic_languages`.

Usage is pretty simple:
```python
# Can a 1-layer transformer always get optimal accuracy on a Markov Chain input? Let's find out by training on
# 1M tokens from a made up (arbitrary) language.

from synthetic_languages import MarkovLanguage, SmallTransformer, TrainingRun

my_lang = MarkovLangauge(
    # Low entropy means this is close to deterministic.
    num_tokens=1024,
    # Use a gaussian distribution to sample entropy for each of 1024 columns. Each entropy value is first sampled
    # from a gaussian of average 4 bits and with >= 95% chance of sampling in [0, 8] bits. If it's negative it gets set
    # to zero. Every column will be a different probability distribution with that value of entropy.
    entropy_distribution='gaussian',
    negative_entropy_mapping='zero',
    average_entropy=4,
    entropy_stdev=2,
    entropy_units='bits',
    # Your langauge name can be used to identify it for experiments and recognize its definition file
    language_name='my_experiment',
    language_author='my_name'
    # Markov languages have markov-chain (FSM) - specific parameters
    connected=True,
    ergodic=True
)
# You should save your language to a file so you can reuse it for comparisons (and reload it later)
# The language definition has metadata, and in this case, the state machine transition probabilities in matrix
# Notation
my_lang.save('my_lang.lang')

# You can create datasets from your language. Each dataset is one or more files. Each different language type class
# has a different language generation process. The MarkovLanguage class, specifically, just follows a random path along
# the markov chain states. You can start from a random "letter" and re-sample (restart) from a random letter (for example 
# if your FSM were not ergodic, though that doesn't apply here).
my_lang.create_dataset(
    'my_lang_dataset.lang',
    initial_sample_strategy='uniform',
    num_samples=1,
    num_tokens_per_sample=1_000_000
)

# You can get a dataloader and we have a simple class of small transformers for ease of testing.
# Here we try with a 1-layer transformer that just down-projects into an attention layer, up--projects, and then
# up-projects again into logits. Because transformers can supposedly learn bigram statistics this might be
# something you'd want to confirm with.
dataloader_train, dataloader_test = my_lang.load_train_test(
    'my_lang_dataset.lang',
    train_amount=0.8,
    embedding_strategy='one-hot',
    batch_size=512,
    shuffle=True
)
model = SmallTransformer(num_layers=1, use_mlp=False)
# We provide a light wrapper around common training code so you don't have to write it yourself. If you want
# something more involved that is also possible, since the model is an torch.nn.Module (and the data is in a file
# regardless, at the end of the day)
training_run = TrainingRun(
    num_epochs=100,
    log_every=5
    save_every=5
    output_folder='.my_experiment'
    loss='nll',
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test
)
training_run.run()
```

## What is this?
This is a library and set of tools to easily form synthetic languages of varying difficulties for the purpose of understanding the capabilities of the current generation of large language models.

Having a ground truth of some form can be very helpful when trying to do (mechanistic) interpretability (MI) on (large) language models. We do not know how to model the english language very deeply (at least AFAIK), so it is hard to get a good sense for what language models _might_ be learning to do: how they are actually _processing_ the input they recieve to predict tokens. 

Some research (https://transformer-circuits.pub/2023/monosemantic-features) has shown that it is possible to find features that correspond to known concepts, and others (i.e. https://openreview.net/forum?id=NpsVSN6o4ul) have found _circuits_ with mild success. Because MI is hard, people have also tried to do circuit analysis on algorithmic tasks (i.e. https://arxiv.org/pdf/2306.17844) and some made up sequences (i.e. https://www.youtube.com/watch?v=yo4QvDn-vsU&ab_channel=NeelNanda for the purposes of understanding the capabilities of transformers). This latter approach appears to be easier than doing MI directly on models of real language (for obvious reasons) but it's not as useful for understanding real world behavior.

What if we could bridge that gap? In computer vision we have datasets ranging from the most basic (MNIST, CIFAR) to more standard ones such as ImageNet and CoCo, as well as specific and realistic ones (though these may be prioritary in some cases). Generally, you can see that there is a ramp from simple problems to more complicated ones. Do we have such a situation in language? Sort of: algorithmic problems currently post as the "simple" ones, but it's not clear if there's such a clear ramp up into real language going through intermediate difficulty problems.

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
    - SAE and SAE Viz support: intgrate https://github.com/callummcdougall/sae_vis
    - Algorithms to pattern match and search the LLM for weights or activation patterns that look like the generating algorithm or have some relation to it (open-ended, WIP)
- `word2vec` embedding strategy (and generally better embedding/tokenization)
- Languages that are subsets of existing languages (such as english), maybe even animal sounds or something else

Please feel free to collaborate and add new ideas! PRs are welcome! Help from linguists and others with deep language would also be appreciated.