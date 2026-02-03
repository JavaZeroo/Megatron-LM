# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import os
import shutil

import torch


_COMPUTE_GRAPH_CAPTURED = False
_ACTIVE_TRACER: "ComputeGraphTracer | None" = None


def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _set_tensor_name(tensor: torch.Tensor, name: str) -> None:
    setattr(tensor, "_megatron_graph_name", name)


def _get_tensor_name(tensor: torch.Tensor) -> str | None:
    return getattr(tensor, "_megatron_graph_name", None)


def maybe_tag_compute_graph_inputs(args, **named_tensors: Any) -> None:
    if not getattr(args, "compute_graph", False):
        return
    for name, tensor in named_tensors.items():
        for item in _iter_tensors(tensor):
            _set_tensor_name(item, name)


def maybe_record_compute_graph_outputs(args, output: Any) -> None:
    if not getattr(args, "compute_graph", False):
        return
    if _ACTIVE_TRACER is None:
        return
    _ACTIVE_TRACER.record_outputs(output)


def should_capture_compute_graph(args, iteration: int | None) -> bool:
    if _COMPUTE_GRAPH_CAPTURED:
        return False
    if not getattr(args, "compute_graph", False):
        return False
    capture_ranks = getattr(args, "compute_graph_ranks", [0])
    rank = getattr(args, "rank", 0)
    if rank not in capture_ranks:
        return False
    if iteration is None:
        return True
    target_iteration = getattr(args, "compute_graph_iteration", 0)
    return iteration == target_iteration


def mark_compute_graph_captured() -> None:
    global _COMPUTE_GRAPH_CAPTURED
    _COMPUTE_GRAPH_CAPTURED = True


def set_active_compute_graph_tracer(tracer: "ComputeGraphTracer | None") -> None:
    global _ACTIVE_TRACER
    _ACTIVE_TRACER = tracer


@dataclass
class ComputeGraphSettings:
    output_dir: str
    file_prefix: str
    file_format: str
    mode: str


class ComputeGraphTracer:
    def __init__(self, settings: ComputeGraphSettings) -> None:
        normalized_mode = settings.mode.lower()
        if normalized_mode not in ("module", "autograd", "both"):
            raise ValueError(f"Unsupported compute graph mode: {settings.mode}")
        self.settings = ComputeGraphSettings(
            output_dir=settings.output_dir,
            file_prefix=settings.file_prefix,
            file_format=settings.file_format,
            mode=normalized_mode,
        )
        self.forward_edges: set[tuple[str, str]] = set()
        self.autograd_edges: set[tuple[str, str]] = set()
        self._tensor_producer: dict[int, str] = {}
        self._input_counter = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._module_names: dict[torch.nn.Module, str] = {}
        self._output_tensors: list[torch.Tensor] = []

    def register_model(self, model: torch.nn.Module, prefix: str = "") -> None:
        for name, param in model.named_parameters():
            setattr(param, "_megatron_param_name", f"{prefix}{name}" if prefix else name)
        for name, module in model.named_modules():
            if module is model:
                continue
            module_name = f"{prefix}{name}" if prefix else name
            self._module_names[module] = module_name
            self._handles.append(module.register_forward_hook(self._forward_hook))

    def clear(self) -> None:
        self.forward_edges.clear()
        self.autograd_edges.clear()
        self._tensor_producer.clear()
        self._input_counter = 0
        self._output_tensors.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _next_input_name(self) -> str:
        name = f"input_{self._input_counter}"
        self._input_counter += 1
        return name

    def _resolve_input_name(self, tensor: torch.Tensor) -> str:
        named = _get_tensor_name(tensor)
        if named is not None:
            return named
        existing = self._tensor_producer.get(id(tensor))
        if existing is not None:
            return existing
        name = self._next_input_name()
        self._tensor_producer[id(tensor)] = name
        return name

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        module_name = self._module_names.get(module, module.__class__.__name__)
        for tensor in _iter_tensors(inputs):
            source = self._resolve_input_name(tensor)
            self.forward_edges.add((source, module_name))
        for tensor in _iter_tensors(output):
            self._tensor_producer[id(tensor)] = module_name

    def record_outputs(self, output: Any) -> None:
        for tensor in _iter_tensors(output):
            if torch.is_tensor(tensor) and tensor.grad_fn is not None:
                self._output_tensors.append(tensor)

    def _autograd_node_name(self, fn: torch.autograd.Function | None) -> str | None:
        if fn is None:
            return None
        name = type(fn).__name__
        if name == "AccumulateGrad" and hasattr(fn, "variable"):
            variable = fn.variable
            param_name = getattr(variable, "_megatron_param_name", None)
            if param_name is not None:
                return f"param::{param_name}"
        return name

    def _build_autograd_edges(self) -> None:
        visited: set[int] = set()
        stack: list[torch.autograd.Function] = []
        for tensor in self._output_tensors:
            if tensor.grad_fn is not None:
                stack.append(tensor.grad_fn)
        while stack:
            fn = stack.pop()
            fn_id = id(fn)
            if fn_id in visited:
                continue
            visited.add(fn_id)
            dst = self._autograd_node_name(fn)
            for next_fn, _ in fn.next_functions:
                src = self._autograd_node_name(next_fn)
                if src is not None and dst is not None:
                    self.autograd_edges.add((src, dst))
                if next_fn is not None:
                    stack.append(next_fn)

    def _write_dot(self, edges: set[tuple[str, str]], path: Path) -> None:
        lines = ["digraph compute_graph {", '  rankdir="LR";']
        for src, dst in sorted(edges):
            safe_src = src.replace('"', "'")
            safe_dst = dst.replace('"', "'")
            lines.append(f'  "{safe_src}" -> "{safe_dst}";')
        lines.append("}")
        path.write_text("\n".join(lines))

    def _render_dot(self, dot_path: Path, output_path: Path) -> None:
        dot_bin = shutil.which("dot")
        if dot_bin is None:
            return
        os.makedirs(output_path.parent, exist_ok=True)
        command = [dot_bin, f"-T{self.settings.file_format}", str(dot_path), "-o", str(output_path)]
        os.spawnv(os.P_WAIT, dot_bin, command)

    def write_graphs(self) -> dict[str, Path]:
        output_dir = Path(self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        if self.settings.mode in ("module", "both"):
            forward_dot = output_dir / f"{self.settings.file_prefix}_forward.dot"
            self._write_dot(self.forward_edges, forward_dot)
            artifacts["forward_dot"] = forward_dot
            if self.settings.file_format != "dot":
                forward_render = output_dir / f"{self.settings.file_prefix}_forward.{self.settings.file_format}"
                self._render_dot(forward_dot, forward_render)
                artifacts["forward_render"] = forward_render

        if self.settings.mode in ("autograd", "both"):
            self._build_autograd_edges()
            autograd_dot = output_dir / f"{self.settings.file_prefix}_autograd.dot"
            self._write_dot(self.autograd_edges, autograd_dot)
            artifacts["autograd_dot"] = autograd_dot
            if self.settings.file_format != "dot":
                autograd_render = output_dir / f"{self.settings.file_prefix}_autograd.{self.settings.file_format}"
                self._render_dot(autograd_dot, autograd_render)
                artifacts["autograd_render"] = autograd_render

        return artifacts
