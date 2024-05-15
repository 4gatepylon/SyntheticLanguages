# Getting Started
In this quick guide we illustrate how to train a small transformer on an existing (preset) language.

<!-- Here is a good pattern to get code from existing code-files! -->
<!-- ## Easy and Efficient Edge Patching
tic tic tic python
--8<-- "experiments/demos/zero_ablate_an_edge.py:20:27"
tic tic tic -->

Synthetic languages exists to make it easy to create known patterns and train transformers on them. Look below for an example.
`
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
