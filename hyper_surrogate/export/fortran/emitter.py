from __future__ import annotations

from typing import ClassVar

import numpy as np

from hyper_surrogate.export.weights import ExportedModel


class FortranEmitter:
    """Emits Fortran 90 code for neural network inference."""

    ACTIVATIONS: ClassVar[dict[str, str]] = {
        "relu": "max(0.0d0, {x})",
        "tanh": "tanh({x})",
        "sigmoid": "1.0d0 / (1.0d0 + exp(-({x})))",
        "softplus": "log(1.0d0 + exp({x}))",
        "identity": "{x}",
    }

    def __init__(self, exported: ExportedModel) -> None:
        self.exported = exported

    def _format_array_1d(self, arr: np.ndarray, name: str) -> str:
        n = arr.shape[0]
        values = ", ".join(f"{v:.15e}d0" for v in arr.flat)
        return f"  DOUBLE PRECISION, PARAMETER :: {name}({n}) = (/ {values} /)"

    def _format_array_2d(self, arr: np.ndarray, name: str) -> str:
        rows, cols = arr.shape
        values = ", ".join(f"{v:.15e}d0" for v in arr.T.flat)  # Fortran is column-major
        return f"  DOUBLE PRECISION, PARAMETER :: {name}({rows},{cols}) = RESHAPE((/ {values} /), (/ {rows}, {cols} /))"

    def _emit_activation(self, var: str, activation: str, size: int) -> list[str]:
        if activation == "identity":
            return []
        lines = []
        template = self.ACTIVATIONS[activation]
        lines.append(f"  DO i = 1, {size}")
        lines.append(f"    {var}(i) = {template.format(x=f"{var}(i)")}")
        lines.append("  END DO")
        return lines

    def emit_mlp(self) -> str:
        layers = self.exported.layers
        weights = self.exported.weights
        meta = self.exported.metadata
        in_dim = meta["input_dim"]
        out_dim = meta["output_dim"]

        lines = ["MODULE nn_surrogate", "  IMPLICIT NONE", ""]

        # Declare weight/bias arrays as parameters
        for i, layer in enumerate(layers):
            w = weights[layer.weights]
            b = weights[layer.bias]
            lines.append(self._format_array_2d(w, f"w{i}"))
            lines.append(self._format_array_1d(b, f"b{i}"))

        # Normalization parameters
        if self.exported.input_normalizer:
            lines.append(self._format_array_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._format_array_1d(self.exported.input_normalizer["std"], "in_std"))
        if self.exported.output_normalizer:
            lines.append(self._format_array_1d(self.exported.output_normalizer["mean"], "out_mean"))
            lines.append(self._format_array_1d(self.exported.output_normalizer["std"], "out_std"))

        lines.extend(["", "CONTAINS", ""])

        # Subroutine
        lines.append("  SUBROUTINE nn_forward(input, output)")
        lines.append(f"    DOUBLE PRECISION, INTENT(IN) :: input({in_dim})")
        lines.append(f"    DOUBLE PRECISION, INTENT(OUT) :: output({out_dim})")

        # Local variables
        hidden_dims = [weights[layer.weights].shape[0] for layer in layers]
        for i in range(len(layers)):
            lines.append(f"    DOUBLE PRECISION :: z{i}({hidden_dims[i]})")
        lines.append(f"    DOUBLE PRECISION :: x_norm({in_dim})")
        lines.append("    INTEGER :: i")
        lines.append("")

        # Normalize input
        if self.exported.input_normalizer:
            lines.append("    ! Normalize input")
            lines.append("    x_norm = (input - in_mean) / in_std")
        else:
            lines.append("    x_norm = input")
        lines.append("")

        # Forward pass
        for i, layer in enumerate(layers):
            input_var = "x_norm" if i == 0 else f"z{i - 1}"
            lines.append(f"    ! Layer {i}")
            lines.append(f"    z{i} = MATMUL(w{i}, {input_var}) + b{i}")
            lines.extend(["  " + line for line in self._emit_activation(f"z{i}", layer.activation, hidden_dims[i])])
            lines.append("")

        # Copy output and denormalize
        last = f"z{len(layers) - 1}"
        if self.exported.output_normalizer:
            lines.append("    ! Denormalize output")
            lines.append(f"    output = {last} * out_std + out_mean")
        else:
            lines.append(f"    output = {last}")

        lines.extend(["", "  END SUBROUTINE nn_forward", "", "END MODULE nn_surrogate"])

        return "\n".join(lines)

    def emit_icnn(self) -> str:
        layers = self.exported.layers
        weights = self.exported.weights
        meta = self.exported.metadata
        in_dim = meta["input_dim"]

        lines = ["MODULE nn_surrogate", "  IMPLICIT NONE", ""]

        # Declare all weight arrays - ICNN wz weights need softplus pre-applied
        for i, layer in enumerate(layers):
            w_key = layer.weights
            w = weights[w_key]
            if "wz" in w_key:
                w = np.log(1.0 + np.exp(w))  # Pre-apply softplus
            lines.append(self._format_array_2d(w, f"w{i}"))
            b_key = layer.bias
            b = weights[b_key]
            lines.append(self._format_array_1d(b, f"b{i}"))

        # wx skip-connection weights
        wx_keys = sorted([k for k in weights if k.startswith("wx_layers") and "weight" in k])
        for i, wk in enumerate(wx_keys):
            if f"w{i}" not in "\n".join(lines):
                lines.append(self._format_array_2d(weights[wk], f"wx{i}"))

        if self.exported.input_normalizer:
            lines.append(self._format_array_1d(self.exported.input_normalizer["mean"], "in_mean"))
            lines.append(self._format_array_1d(self.exported.input_normalizer["std"], "in_std"))

        lines.extend(["", "CONTAINS", ""])

        # Forward + backward pass for energy and stress
        lines.append("  SUBROUTINE nn_forward(input, energy, stress)")
        lines.append(f"    DOUBLE PRECISION, INTENT(IN) :: input({in_dim})")
        lines.append("    DOUBLE PRECISION, INTENT(OUT) :: energy")
        lines.append(f"    DOUBLE PRECISION, INTENT(OUT) :: stress({in_dim})")
        lines.append(f"    DOUBLE PRECISION :: x_norm({in_dim})")

        hidden_dims = []
        for layer in layers[:-1]:
            w = weights[layer.weights]
            hidden_dims.append(w.shape[0])

        for i, hd in enumerate(hidden_dims):
            lines.append(f"    DOUBLE PRECISION :: z{i}({hd})")
            lines.append(f"    DOUBLE PRECISION :: dz{i}({hd})")

        lines.append(f"    DOUBLE PRECISION :: denergy({in_dim})")
        lines.append("    INTEGER :: i, j")
        lines.append("")

        if self.exported.input_normalizer:
            lines.append("    x_norm = (input - in_mean) / in_std")
        else:
            lines.append("    x_norm = input")
        lines.append("")

        # Forward pass
        lines.append("    ! Forward pass")
        lines.append("    ! Layer 0 (x-path only)")
        lines.append("    z0 = MATMUL(w0, x_norm) + b0")
        lines.extend(["  " + line for line in self._emit_activation("z0", layers[0].activation, hidden_dims[0])])
        lines.append("")

        for i in range(1, len(hidden_dims)):
            lines.append(f"    ! Layer {i} (wz + wx skip)")
            lines.append(f"    z{i} = MATMUL(w{i}, z{i - 1}) + MATMUL(wx{i}, x_norm) + b{i}")
            lines.extend(["  " + line for line in self._emit_activation(f"z{i}", layers[i].activation, hidden_dims[i])])
            lines.append("")

        # Output
        last_hidden = len(hidden_dims) - 1
        last_layer_idx = len(layers) - 1
        lines.append("    ! Output layer")
        lines.append(f"    energy = DOT_PRODUCT(w{last_layer_idx}(1,:), z{last_hidden}) + b{last_layer_idx}(1)")
        lines.append("")

        # Backward pass placeholder
        lines.append("    ! Backward pass (chain rule for stress = d_energy/d_input)")
        lines.append("    energy = 0.0d0")
        lines.append("    stress = 0.0d0")

        lines.extend(["", "  END SUBROUTINE nn_forward", "", "END MODULE nn_surrogate"])

        return "\n".join(lines)

    def emit(self) -> str:
        arch = self.exported.metadata.get("architecture", "mlp")
        if arch == "mlp":
            return self.emit_mlp()
        elif arch == "icnn":
            return self.emit_icnn()
        else:
            msg = f"Unknown architecture: {arch}"
            raise ValueError(msg)

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.emit())
