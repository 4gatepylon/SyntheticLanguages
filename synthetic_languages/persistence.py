import json
import os
import pathlib
import re
from abc import ABC
from io import BytesIO
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import boto3  # type: ignore
import dotenv
import pandas as pd
import torch
import torch.nn as nn
from botocore.exceptions import ClientError  # type: ignore

from synthetic_languages.training.configs.model_configs import RawModelConfig

# TODO: LocalPersister.load_model
# TODO: Add create_bucket option in S3 persister

# TODO: Add "check for versioning" in S3
# TODO: Persist list of buckets on init
# TODO: When inferring correct config, log a warning to the user about n_ctx

# TODO: Create set collection_location
# TODO: Make all save_model functions async
# TODO: Create save config & implement it in the training loop
# TODO: Create query commit hash and add it to the save_config method

# TODO: Change _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to nested dict
# TODO: Clean up _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to have smaller num of keys
# TODO: Add a check to _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to make sure every
#    key is visited (??)
# TODO: Generalize _state_dict_to_model_config into it's own base class so that it
#    can handle different kinds of model classes


class Persister(ABC):
    # TODO(Adriano) consider bringing this back but it just spews type-errors and sucks
    # @abstractmethod
    # def save_model(self, model: nn.Module, num_tokens_trained: int):
    #     ...

    # @abstractmethod
    # def load_model(
    #    self, model_class: nn.Module, object_name: pathlib.Path | str
    # ) -> nn.Module:
    #     ...

    # @abstractmethod
    # def _save_overwrite_protection(self, object_name: pathlib.Path | str):
    #     ...
    pass


class LocalPersister(Persister):
    def __init__(self, collection_location: pathlib.Path):
        assert collection_location.is_dir()
        assert collection_location.exists()
        self.collection_location: pathlib.Path = collection_location

    def _save_overwrite_protection(
        self, object_name: pathlib.Path | str
    ):  # type: ignore[override]
        if isinstance(object_name, str):
            object_name = pathlib.Path(object_name)
        assert isinstance(object_name, pathlib.Path)
        if object_name.exists():
            raise ValueError(f"Overwrite Protection: {object_name} already exists.")

    def save_model(self, model: nn.Module, num_tokens_trained: int):
        save_path = self.collection_location / f"{num_tokens_trained}.pt"
        self._save_overwrite_protection(object_name=save_path)

        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)

    def load_model(
        self, model: nn.Module, object_name: pathlib.Path | str
    ) -> nn.Module:
        # state_dict = torch.load(self.collection_location / object_name)
        # model.load_state_dict(state_dict=state_dict)
        # return model
        raise NotImplementedError


class S3Persister(Persister):
    def __init__(self, collection_location: str):
        dotenv.load_dotenv()
        assert os.environ.get("AWS_ACCESS_KEY_ID") is not None
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") is not None

        self.s3 = boto3.client("s3")
        buckets = [x["Name"] for x in self.s3.list_buckets()["Buckets"]]
        if collection_location not in buckets:
            raise ValueError(
                f"{collection_location} is not an existing bucket. "
                + "Either use one of the existing buckets or create a new bucket"
            )
        self.collection_location: str = collection_location

    def _save_overwrite_protection(self, object_name: str):  # type: ignore[override]
        try:
            self.s3.head_object(Bucket=self.collection_location, Key=object_name)
            raise ValueError(
                "Overwrite Protection: "
                + f"{self.collection_location}/{object_name} already exists"
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass
            else:
                raise ValueError(f"Expected 404 from empty object, received {e}")

    def save_model(self, model: nn.Module, num_tokens_trained: int):
        object_name = f"{num_tokens_trained}.pt"
        self._save_overwrite_protection(object_name=object_name)

        print(f"Saving model as {object_name} in bucket {self.collection_location}")

        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.collection_location, object_name)

    def load_csv(self, object_name: str) -> pd.DataFrame:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        return pd.read_csv(download_buffer)

    def load_model(self, object_name: str, device: torch.device) -> nn.Module:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        state_dict = torch.load(download_buffer)

        train_config = self.load_json("train_config.json")
        if train_config is not None:
            # TODO: refactor this
            required_fields = [
                "d_vocab",
                "d_model",
                "n_ctx",
                "d_head",
                "n_heads",
                "n_layers",
            ]
            config_dict = {
                k: v for k, v in train_config.items() if k in required_fields
            }
            config_dict["d_mlp"] = 4 * config_dict["d_model"]
            # change key n_heads to n_head
            config_dict["n_head"] = config_dict.pop("n_heads")
            config = RawModelConfig(**config_dict)
        else:
            print("No train_config.json found, inferring from state_dict")
            config = _state_dict_to_model_config(state_dict=state_dict)

        model = config.to_hooked_transformer(device=device)
        model.load_state_dict(state_dict=state_dict)
        assert isinstance(model, nn.Module)  # typing
        return model

    def list_objects(self) -> List[str]:
        objects = []
        continuation_token = None

        while True:
            kwargs = {"Bucket": self.collection_location}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = self.s3.list_objects_v2(**kwargs)
            contents = response.get("Contents", [])
            objects.extend([obj["Key"] for obj in contents])

            if "NextContinuationToken" in response:
                continuation_token = response["NextContinuationToken"]
            else:
                break

        return objects

    def load_json(self, object_name: str) -> Optional[Dict[str, Any]]:
        try:
            json_str = self.load_object(object_name)
            return json.loads(json_str)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise e

    def load_object(self, object_name: str) -> str:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        return download_buffer.read().decode("utf-8")


def _state_dict_to_model_config(
    state_dict: OrderedDict[str, Any], n_ctx: int = 10
) -> RawModelConfig:
    # Naming => Hack for line length
    FGOENDSFGS: Dict[str, List[Tuple[str, int]]] = {
        r"embed\.W_E": [("d_vocab", 0), ("d_model", 1)],
        r"pos_embed\.W_pos": [],
        r"blocks\.\d+\.ln\d+\.(w|b)": [],
        r"blocks\.\d+\.attn\.W_Q": [("n_head", 0), ("d_head", 2)],
        r"blocks\.\d+\.attn\.b_Q": [],
        r"blocks\.\d+\.attn\.W_K": [],
        r"blocks\.\d+\.attn\.b_K": [],
        r"blocks\.\d+\.attn\.W_O": [],
        r"blocks\.\d+\.attn\.b_O": [],
        r"blocks\.\d+\.attn\.W_V": [],
        r"blocks\.\d+\.attn\.b_V": [],
        r"blocks\.\d+\.attn\.mask": [],
        r"blocks\.\d+\.attn\.IGNORE": [],
        r"blocks\.\d+\.mlp\.W_in": [("d_mlp", 1)],
        r"blocks\.\d+\.mlp\.b_in": [],
        r"blocks\.\d+\.mlp\.W_out": [],
        r"blocks\.\d+\.mlp\.b_out": [],
        r"ln_final\.(w|b)": [],
        r"unembed\.(W_U|b_U)": [],
    }
    _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY = FGOENDSFGS

    def _extract_true_key(dictionary: Dict[str, bool]) -> str:
        out = []
        for key, value in dictionary.items():
            if value:
                out.append(key)
        assert len(out) == 1, (
            f"{out} does not fit one of the expected module regexs: "
            + f"{_HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY}"
        )
        return out[0]

    def _extract_n_layers(state_dict: OrderedDict[str, Any]) -> int:
        highest_block_idx = None
        for key in state_dict.keys():
            if not bool(re.match(r"blocks\.\d+\.", key)):
                continue
            _search = re.search(r"\d+", key)
            assert _search is not None
            local_block_idx = int(_search.group())
            if highest_block_idx is None:
                highest_block_idx = local_block_idx
            elif local_block_idx > highest_block_idx:
                highest_block_idx = local_block_idx
        assert highest_block_idx is not None
        return highest_block_idx + 1

    param_dict = dict(
        d_vocab=None,
        d_model=None,
        n_ctx=n_ctx,
        d_head=None,
        n_head=None,
        d_mlp=None,
        n_layers=_extract_n_layers(state_dict=state_dict),
    )
    for module_name, module in state_dict.items():
        regex_dict = {
            pattern: bool(re.match(pattern, module_name))
            for pattern in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY.keys()
        }
        pattern = _extract_true_key(regex_dict)
        for key, dim in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY[pattern]:
            if param_dict[key] is None:
                param_dict[key] = module.size()[dim]
    assert all([value is not None for value in param_dict.values()])
    return RawModelConfig(**param_dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    from transformer_lens import HookedTransformer  # type: ignore

    persister = S3Persister(collection_location="mess3-param-change")
    model = persister.load_model(device=torch.device("cpu"), object_name="4800000.pt")
    assert isinstance(model, nn.Module)
    assert isinstance(model, HookedTransformer)
