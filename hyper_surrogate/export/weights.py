from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hyper_surrogate.data.dataset import Normalizer
from hyper_surrogate.models.base import LayerInfo


@dataclass
class ExportedModel:
    layers: list[LayerInfo]
    weights: dict[str, np.ndarray]
    input_normalizer: dict[str, np.ndarray] | None = None
    output_normalizer: dict[str, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        save_dict: dict[str, Any] = {}
        # Weights
        for k, v in self.weights.items():
            save_dict[f"w_{k}"] = v
        # Normalizers
        if self.input_normalizer:
            save_dict["in_norm_mean"] = self.input_normalizer["mean"]
            save_dict["in_norm_std"] = self.input_normalizer["std"]
        if self.output_normalizer:
            save_dict["out_norm_mean"] = self.output_normalizer["mean"]
            save_dict["out_norm_std"] = self.output_normalizer["std"]
        # Metadata and layers as JSON strings
        save_dict["_metadata"] = np.array([json.dumps(self.metadata)])
        layers_data = [
            {"weights": layer.weights, "bias": layer.bias, "activation": layer.activation} for layer in self.layers
        ]
        save_dict["_layers"] = np.array([json.dumps(layers_data)])
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> ExportedModel:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["_metadata"][0]))
        layers_data = json.loads(str(data["_layers"][0]))
        layers = [LayerInfo(**d) for d in layers_data]
        weights = {k[2:]: data[k] for k in data if k.startswith("w_")}
        in_norm = None
        if "in_norm_mean" in data:
            in_norm = {"mean": data["in_norm_mean"], "std": data["in_norm_std"]}
        out_norm = None
        if "out_norm_mean" in data:
            out_norm = {"mean": data["out_norm_mean"], "std": data["out_norm_std"]}
        return cls(
            layers=layers, weights=weights, input_normalizer=in_norm, output_normalizer=out_norm, metadata=metadata
        )


def extract_weights(
    model: Any,
    input_normalizer: Normalizer | None = None,
    output_normalizer: Normalizer | None = None,
) -> ExportedModel:
    from hyper_surrogate.models.base import SurrogateModel

    if not isinstance(model, SurrogateModel):
        msg = f"Expected SurrogateModel, got {type(model)}"
        raise TypeError(msg)
    return ExportedModel(
        layers=model.layer_sequence(),
        weights=model.export_weights(),
        input_normalizer=input_normalizer.params if input_normalizer else None,
        output_normalizer=output_normalizer.params if output_normalizer else None,
        metadata={
            "architecture": model.__class__.__name__.lower(),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
        },
    )
