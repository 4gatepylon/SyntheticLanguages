# Setup jupyter: https://code.visualstudio.com/docs/datascience/jupyter-notebooks
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import LoggingConfig, OptimizerConfig, PersistanceConfig, ProcessDatasetConfig, TrainConfig
from epsilon_transformers.training.train import train_model
from pathlib import Path
import itertools

if __name__ == "__main__":
    print("Training model on mock config!")
    Path("./z1r-test-500B").mkdir(exist_ok=False) # NOTE THIS IS COPIED FROM experiments/
    
    # 500T
    num_tokens = 500_000_000_000_000
    test_split = 0.00_000_000_015
    weight_decay = 0.00_000_000_1 # 1/billion
    processes = ['z1r', 'rrxor', 'mess3']
    parameters = [
        [{'prob_of_zero_from_r_state': 0.1}, {'prob_of_zero_from_r_state': 0.3}, {'prob_of_zero_from_r_state': 0.5}, {'prob_of_zero_from_r_state': 0.7}, {'prob_of_zero_from_r_state': 0.9}],
        [{'pR1':0.2, 'pR2':0.5}, {'pR1':0.5, 'pR2':0.5}, {'pR1':0.8, 'pR2':0.5}, {'pR1':0.5, 'pR2':0.5}, {'pR1':0.5, 'pR2':0.5}, {'pR1':0.5, 'pR2':0.5}],
        [{'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}, {'x': 0.5, 'a': 0.85}]
    ]

    model_config = RawModelConfig(
        d_vocab=2,
        d_model=64,
        n_ctx=10,
        d_head=8,
        n_head=8,
        # Normally MLP should be HIGHER dimensional... without thinking too much we set it to 4x
        d_mlp=256,
        n_layers=4,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type='adam',
        learning_rate=1e-2,
        weight_decay=weight_decay
    )

    dataset_config = ProcessDatasetConfig(
        process='z1r',
        # process_params={'x': 0.5, 'a': 0.85},
        # process_params={'pR1':0.5, 'pR2':0.5},
        process_params={'prob_of_zero_from_r_state': 0.5},
        batch_size=64,
        num_tokens=10_000_000,
        test_split=0.00_000_015
    )

    persistance_config = PersistanceConfig(
        location='local',
        collection_location=Path("./z1r-test-10m"),
        checkpoint_every_n_tokens=100_000
    )

    mock_config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        persistance=persistance_config,
        logging=LoggingConfig(project_name="z1r-test", wandb=True),
        verbose=True,
        seed=42
    )
    train_model(element_config)
