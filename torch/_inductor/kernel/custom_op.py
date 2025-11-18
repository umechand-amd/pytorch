# Owner(s): ["module: inductor"]

import functools
import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
from torch._inductor import config
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class CustomOpConfig:
    """Config for custom op autotuning.

    Specifies optional decomposition function with parameter values.
    Each config creates exactly one variant to benchmark.

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(head_dim=128)  # Use default impl with params
    """

    def __init__(
        self,
        decomposition: Optional[Callable[..., Any]] = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )
        self.decomposition = decomposition
        self.params = params

    def get_decomposition(
        self, default_impl: Optional[Callable[..., Any]] = None
    ) -> Callable[..., Any]:
        """Return the decomposition function for this config."""
        if self.decomposition is not None:
            return self.decomposition
        if default_impl is not None and callable(default_impl):
            return default_impl
        raise TypeError(
            "No decomposition specified in config and no default implementation provided."
        )

    def __repr__(self) -> str:
        decomp_name = self.decomposition.__name__ if self.decomposition else "default"
        parts = [decomp_name]
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            parts.append(params_str)
        return f"CustomOpConfig({', '.join(parts)})"


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
    "CustomOpConfig",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from args/kwargs, separating from non-tensor parameters."""
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Process args and kwargs: separate tensor inputs and non tensor args
    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer)):
            tensor_inputs.append(arg)
        else:
            # Add non-tensor positional args to kwargs with generated names
            non_tensor_kwargs[f"arg_{i}"] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer)):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _merge_config_and_runtime_kwargs(
    config_params: dict[str, Any],
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge config parameters with runtime kwargs. Runtime kwargs take precedence.
       If there are conflicts, log a warning and use runtime value.

    Args:
        config_params: Parameters from CustomOpConfig
        runtime_kwargs: Runtime non-tensor kwargs from _extract_tensor_inputs

    Returns:
        Merged kwargs dictionary with runtime values taking precedence
    """
    merged_kwargs = config_params.copy()

    # Check for conflicts and let runtime kwargs dominate
    conflicts = OrderedSet(config_params.keys()).intersection(runtime_kwargs.keys())

    for key in conflicts:
        log.warning(
            "Parameter '%s' specified both in CustomOpConfig (%s) "
            "and at runtime (%s). Using runtime value.",
            key,
            config_params[key],
            runtime_kwargs[key],
        )

    # Runtime kwargs override config params
    merged_kwargs.update(runtime_kwargs)

    return merged_kwargs


def _adapt_user_input_gen_fns(
    inputs: list[Any],
    arg_names: list[str],
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
       Inductor autotune's input_gen_fns expects index of arg_names as key.

    Uses V.graph.sizevars.size_hints() to guess best for dynamic shapes.
    """

    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            log.warning(
                "Unknown argument name '%s' in input_gen_fns. "
                "Available argument names: %s",
                name,
                list(name_to_index.keys()),
            )

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            raw_shape = ir_buffer.get_size()
            concrete_shape = V.graph.sizevars.size_hints(
                raw_shape, fallback=config.unbacked_symint_fallback
            )

            fake_tensor = torch.empty(
                concrete_shape, dtype=ir_buffer.get_dtype(), device="meta"
            )
            return user_function(fake_tensor)

        return internal_input_gen_fn

    return {
        i: create_internal_input_gen_fn(
            user_gen_fn, arg_names[i] if i < len(arg_names) else f"arg_{i}"
        )
        for i, user_gen_fn in index_based_fns.items()
        if i < len(inputs)
    }


def _generate_dispatch_function(
    name: str,
    range_to_best_impl: dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]],
    tensor_name: str,
    dim_index: int,
    op_overload: torch._ops.OpOverload,
) -> str:
    """Generate Python code for torch.cond dispatch function.

    Args:
        name: Name of the operation
        range_to_best_impl: Mapping from (range_start, range_end) to (impl_func, kwargs, impl_name)
        tensor_name: Name of tensor parameter to dispatch on
        dim_index: Dimension index to check
        op_overload: The original custom op

    Returns:
        Python code as string
    """
    import inspect

    # Sort ranges
    sorted_items = sorted(range_to_best_impl.items())

    # Build the function code
    lines = []
    lines.append('"""Auto-generated dispatch function for range-based autotuning."""')
    lines.append("")
    lines.append("import torch")
    lines.append("")

    # Import the implementations
    impl_names_set = set()
    for _, (impl_func, _, impl_name) in sorted_items:
        if impl_name not in impl_names_set:
            impl_names_set.add(impl_name)
            # Get module and qualname
            module = inspect.getmodule(impl_func)
            if module and module.__name__ != "__main__":
                lines.append(f"from {module.__name__} import {impl_name}")
            else:
                lines.append(
                    f"# Note: {impl_name} is defined in __main__, you need to import it manually"
                )

    lines.append("")
    lines.append("")

    sig = inspect.signature(op_overload)
    params = list(sig.parameters.keys())
    params_str = ", ".join(params)

    lines.append(f"def {name}_dispatch({params_str}):")
    lines.append(
        f'    """Dispatch function with torch.cond based on {tensor_name}.shape[{dim_index}]."""'
    )
    lines.append("    ")
    lines.append("    # Get dimension value for dispatch")
    lines.append(f"    dim_size = {tensor_name}.shape[{dim_index}]")
    lines.append("    ")

    lines.append("    # Range-based dispatch using torch.cond")

    def build_cond_code(idx: int, indent_level: int) -> list[str]:
        """Recursively build torch.cond code."""
        result_lines = []
        indent = "    " * indent_level

        (range_start, range_end), (impl_func, impl_kwargs, impl_name) = sorted_items[
            idx
        ]

        if idx == len(sorted_items) - 1:
            result_lines.append(
                f"{indent}# Range [{range_start}, {range_end if range_end != float('inf') else 'inf'})"
            )
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in impl_kwargs.items())
            if kwargs_str:
                result_lines.append(f"{indent}{impl_name}({params_str}, {kwargs_str})")
            else:
                result_lines.append(f"{indent}{impl_name}({params_str})")
        else:
            # Create torch.cond
            result_lines.append(
                f"{indent}# Range [{range_start}, {range_end if range_end != float('inf') else 'inf'})"
            )

            end_str = "float('inf')" if range_end == float("inf") else str(range_end)
            result_lines.append(f"{indent}torch.cond(")
            result_lines.append(f"{indent}    dim_size <= {end_str},")

            # True branch
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in impl_kwargs.items())
            if kwargs_str:
                result_lines.append(
                    f"{indent}    lambda: {impl_name}({params_str}, {kwargs_str}),"
                )
            else:
                result_lines.append(f"{indent}    lambda: {impl_name}({params_str}),")

            # False branch - recursively build next
            result_lines.append(f"{indent}    lambda: (")
            result_lines.extend(build_cond_code(idx + 1, indent_level + 2))
            result_lines.append(f"{indent}    )")
            result_lines.append(f"{indent})")

        return result_lines

    # Start building from the first range
    lines.append("    return (")
    lines.extend(build_cond_code(0, 2))
    lines.append("    )")

    lines.append("")
    lines.append("")

    # Add a main section for testing
    lines.append('if __name__ == "__main__":')
    lines.append('    print("Range-based dispatch function generated successfully!")')
    lines.append(f'    print("Function name: {name}_dispatch")')
    lines.append(f'    print("Dispatch parameter: {tensor_name}.shape[{dim_index}]")')
    lines.append(f'    print("Number of ranges: {len(sorted_items)}")')
    lines.append('    print("Ranges:")')

    for (range_start, range_end), (_, _, impl_name) in sorted_items:
        end_str = "inf" if range_end == float("inf") else str(range_end)
        lines.append(f'    print("  [{range_start}, {end_str}): {impl_name}")')

    return "\n".join(lines)


def _merge_identical_implementations(
    range_to_best_impl: dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]],
) -> dict[tuple[int, Union[int, float]], tuple[Callable, dict, str]]:
    """Merge consecutive ranges using the same implementation."""
    if not range_to_best_impl:
        return {}

    sorted_ranges = sorted(range_to_best_impl.items(), key=lambda x: x[0][0])
    merged = {}
    current_range_start, current_range_end = sorted_ranges[0][0]
    current_impl, current_kwargs, current_name = sorted_ranges[0][1]

    for i in range(1, len(sorted_ranges)):
        (next_start, next_end), (next_impl, next_kwargs, next_name) = sorted_ranges[i]

        if (
            current_impl == next_impl
            and current_kwargs == next_kwargs
            and current_name == next_name
            and next_start == current_range_end + 1
        ):
            current_range_end = next_end
        else:
            merged[(current_range_start, current_range_end)] = (
                current_impl,
                current_kwargs,
                current_name,
            )
            current_range_start, current_range_end = next_start, next_end
            current_impl, current_kwargs, current_name = (
                next_impl,
                next_kwargs,
                next_name,
            )

    merged[(current_range_start, current_range_end)] = (
        current_impl,
        current_kwargs,
        current_name,
    )

    if len(merged) < len(range_to_best_impl):
        log.info(
            f"Range merging: reduced from {len(range_to_best_impl)} to {len(merged)} ranges"
        )

    return merged


def _split_points_to_ranges(
    split_points: list[int],
) -> list[tuple[int, Union[int, float]]]:
    """Convert split points to inclusive-inclusive ranges.

    Example: split_points=[512, 2048] ->
             [(1, 512), (513, 2048), (2049, float('inf'))]
    """
    ranges = []
    start = 1

    for split_point in split_points:
        ranges.append((start, split_point))
        start = split_point + 1

    ranges.append((start, float("inf")))

    return ranges


def _create_range_input_gen_fn(
    base_gen_fn: Callable[[torch.Tensor], torch.Tensor],
    dim_index: int,
    range_start: int,
    range_end: Union[int, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create input generator that produces tensor with dimension in range."""

    def constrained_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
        result = base_gen_fn(fake_tensor)
        shape = list(result.shape)

        # Pick middle of range
        if range_end == float("inf"):
            target_dim = int(range_start + 100)
        else:
            target_dim = (int(range_start) + int(range_end)) // 2

        target_dim = max(
            int(range_start),
            min(
                target_dim,
                int(range_end) - 1 if range_end != float("inf") else target_dim,
            ),
        )

        shape[dim_index] = target_dim
        return torch.randn(*shape, dtype=result.dtype, device=result.device)

    return constrained_gen_fn


def _extract_winning_decomposition_index(
    choice_name: str,
    decompositions: list[Callable],
) -> int:
    """Extract the decomposition index from winning SubgraphChoiceCaller's name.

    The choice name format is: "{op_name}_range_{start}_{end}_{decomp_name}_{counter}"
    We parse it to find which decomposition won by matching decomp_name.

    Args:
        choice_name: Name of the winning SubgraphChoiceCaller
        decompositions: List of decomposition functions

    Returns:
        Index into decompositions list (0-based)
    """
    if not choice_name:
        log.warning("Empty choice name, defaulting to first decomposition")
        return 0

    # Try to match decomposition by name
    for i, decomp in enumerate(decompositions):
        decomp_name = decomp.__name__
        # Check if decomposition name appears in choice name
        if decomp_name in choice_name:
            log.debug(
                f"Matched choice '{choice_name}' to decomposition[{i}] '{decomp_name}'"
            )
            return i

    # Fallback: could not determine, use first
    log.warning(
        f"Could not determine winning decomposition from choice name '{choice_name}', "
        f"defaulting to first decomposition"
    )
    return 0


def _extract_tensor_by_name(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tensor_name: str,
    op_overload: torch._ops.OpOverload,
) -> Optional[Any]:
    """Extract a tensor from args/kwargs by parameter name.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        tensor_name: Name of the parameter to extract
        op_overload: OpOverload to get parameter names

    Returns:
        The tensor (TensorBox/Buffer) if found, None otherwise
    """
    import inspect

    # Get parameter names from the op's signature
    try:
        sig = inspect.signature(op_overload)
        param_names = list(sig.parameters.keys())
    except Exception:
        log.warning("Could not get signature for %s, using fallback", op_overload)
        # Fallback: assume tensor_name matches position or kwargs
        if tensor_name in kwargs:
            return kwargs[tensor_name]
        return None

    # Check if tensor_name is in kwargs
    if tensor_name in kwargs:
        return kwargs[tensor_name]

    # Check if tensor_name is in positional args
    if tensor_name in param_names:
        param_index = param_names.index(tensor_name)
        if param_index < len(args):
            return args[param_index]

    return None


def _get_dimension_value(tensor: Any, dim_index: int) -> Any:
    """Get the dimension value from a tensor IR node.

    Args:
        tensor: TensorBox or Buffer IR node
        dim_index: Dimension index to extract

    Returns:
        Dimension value (may be symbolic or concrete)
    """
    if hasattr(tensor, "get_size"):
        # Buffer has get_size()
        shape = tensor.get_size()
    elif hasattr(tensor, "data") and hasattr(tensor.data, "get_size"):
        # TensorBox wraps data
        shape = tensor.data.get_size()
    else:
        raise RuntimeError(f"Cannot extract shape from {type(tensor)}")

    if dim_index >= len(shape):
        raise IndexError(
            f"dim_index {dim_index} out of range for tensor with {len(shape)} dimensions"
        )

    return shape[dim_index]


def _create_fallback_choice(
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Create fallback choice for default implementation."""

    def fallback_wrapper(*args: Any) -> Any:
        return default_impl(*args, **kwargs)

    return ExternKernelChoice(
        kernel=fallback_wrapper,
        name=f"{name}_fallback_default",
        has_out_variant=False,
        op_overload=default_impl,
        use_fallback_kernel=True,
    )


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[torch.fx.Node],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: Optional[
        dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    return_choice: bool = False,
) -> Union[TensorBox, Any, tuple[Any, Any]]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    Currently supports SINGLE OUTPUT custom ops only.
    TODO: Add support for multiple output custom ops (tuple/list returns).

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.
    After selecting the best choice, applies inline fusion if the winning choice has a graph.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        non_tensor_args: List of kwargs dicts, paired with corresponding decompositions arg
        op_overload: OpOverload of the custom op, used as fallback implementation
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.

    Returns:
        IR node representing the optimized operation result

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    if len(decompositions) != len(non_tensor_args):
        raise ValueError(
            f"decompositions and non_tensor_args must have same length, "
            f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
        )

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        # pyrefly: ignore [bad-argument-type]
        decompositions=decompositions,
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
    )

    # Add default implementation as fallback
    if op_overload and hasattr(op_overload, "_op"):
        fallback_name = f"{name}_fallback_default"
        from torch._inductor.select_algorithm import extern_kernels

        # Skip if extern_kernel already registered to avoid duplicate registration error
        if not hasattr(extern_kernels, fallback_name):
            with V.fake_mode:
                fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
                fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}
                fake_output = op_overload(*fake_inputs, **fallback_kwargs)

            fallback_choice = _create_fallback_choice(
                name, op_overload, fake_output, fallback_kwargs
            )
            fallback_choice.maybe_append_choice(
                choices=choices,
                input_nodes=list(inputs),
                layout=FixedLayout(
                    device=fake_output.device,
                    dtype=fake_output.dtype,
                    size=fake_output.shape,
                    stride=fake_output.stride(),
                ),
            )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    # Convert user input generation functions to internal format
    input_gen_fns = {}
    if user_input_gen_fns:
        import inspect

        arg_names = (
            list(inspect.signature(decompositions[0]).parameters.keys())
            if decompositions
            else []
        )
        input_gen_fns = _adapt_user_input_gen_fns(inputs, arg_names, user_input_gen_fns)

    # Run autotuning and get both result and winning choice
    selected_result, winning_choice = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        return_choice=True,
    )

    # Apply inlining for fusion if winning_choice has graph; otherwise return result as-is(default fallback impl)
    if winning_choice.gm is not None:
        log.debug(
            "Inlining winning choice: %s (name=%s)",
            getattr(winning_choice, "name", type(winning_choice).__name__),
            name,
        )
        from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
        if return_choice:
            return result, winning_choice
        return result

    log.debug(
        "Winning choice does not support inlining: %s (name=%s)",
        getattr(winning_choice, "name", type(winning_choice).__name__),
        name,
    )
    if return_choice:
        return selected_result, winning_choice
    return selected_result


def _create_autotuning_lowering(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]],
    is_range_based: bool = False,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning."""
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_fn(*args: Any, **kwargs: Any) -> Any:
            tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

            decompositions = []
            non_tensor_args = []

            for cfg in processed_configs:
                decomp = cfg.get_decomposition(default_impl=default_impl)
                decompositions.append(decomp)
                merged_kwargs = _merge_config_and_runtime_kwargs(
                    cfg.params, runtime_kwargs
                )
                non_tensor_args.append(merged_kwargs)

            result = autotune_custom_op(
                name=name,
                decompositions=decompositions,
                inputs=tensor_inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=input_gen_fns,
            )

            validate_ir(result)
            return result

        return standard_lowering_fn

    # Range-based autotuning path
    tensor_name, dim_index = dispatch_on
    ranges = _split_points_to_ranges(split_points)

    @functools.wraps(op_overload)
    def range_based_lowering_fn(*args: Any, **kwargs: Any) -> Any:
        log.info("=== Range-based Autotuning for %s ===", name)
        log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

        # Benchmark each range and store the winning choices
        range_to_winning_choice: dict[tuple[int, Union[int, float]], Any] = {}

        for range_start, range_end in ranges:
            # Create range-specific input generator
            range_input_gen_fns = None
            if input_gen_fns and tensor_name in input_gen_fns:
                base_gen_fn = input_gen_fns[tensor_name]
                range_gen_fn = _create_range_input_gen_fn(
                    base_gen_fn, dim_index, range_start, range_end
                )
                range_input_gen_fns = {**input_gen_fns, tensor_name: range_gen_fn}

            # Build decompositions and kwargs for this range
            decompositions = []
            non_tensor_args = []

            for cfg in processed_configs:
                decomp = cfg.get_decomposition(default_impl=default_impl)
                decompositions.append(decomp)
                merged_kwargs = _merge_config_and_runtime_kwargs(
                    cfg.params, runtime_kwargs
                )
                non_tensor_args.append(merged_kwargs)

            range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

            # Run autotuning for this range and get winning choice
            autotuned_result, winning_choice = autotune_custom_op(
                name=range_name,
                decompositions=decompositions,
                inputs=tensor_inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=range_input_gen_fns,
                return_choice=True,
            )

            range_to_winning_choice[(range_start, range_end)] = winning_choice

            log.info(
                "Range [%s, %s]: Selected %s",
                range_start,
                range_end if range_end != float("inf") else "inf",
                getattr(winning_choice, "name", "unknown"),
            )

        # Build range_to_best_impl from range_to_winning_choice
        range_to_best_impl = {}
        for (range_start, range_end), choice in range_to_winning_choice.items():
            choice_name = getattr(choice, "name", "")

            # Extract winning decomposition index
            winning_idx = _extract_winning_decomposition_index(
                choice_name, decompositions
            )

            impl = decompositions[winning_idx]
            impl_kwargs = non_tensor_args[winning_idx]
            impl_name = impl.__name__

            range_to_best_impl[(range_start, range_end)] = (
                impl,
                impl_kwargs,
                impl_name,
            )

        log.info("Completed autotuning for %d ranges", len(range_to_best_impl))

        # Merge consecutive ranges that use the same implementation
        merged_range_to_best_impl = _merge_identical_implementations(range_to_best_impl)

        log.info(
            "After merging: %d unique implementations across %d ranges",
            len(set((name for _, _, name in merged_range_to_best_impl.values()))),
            len(merged_range_to_best_impl),
        )

        # Check if all ranges use the same implementation (no dispatch needed)
        unique_impls = set(
            (impl, tuple(sorted(kwargs.items())))
            for impl, kwargs, _ in merged_range_to_best_impl.values()
        )
        if len(unique_impls) == 1:
            log.info(
                "All ranges selected the same implementation - skipping dispatch, using direct inline"
            )
            # Use the single implementation directly - inline it without dispatch
            single_impl, single_kwargs, single_name = list(
                merged_range_to_best_impl.values()
            )[0]

            # Trace and inline the single implementation
            from torch.fx.experimental.proxy_tensor import make_fx
            from ..decomposition import select_decomp_table

            def single_impl_wrapper(*tensors):
                return single_impl(*tensors, **{**runtime_kwargs, **single_kwargs})

            with V.fake_mode:
                fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
                decomposition_table = select_decomp_table()
                impl_gm = make_fx(
                    single_impl_wrapper,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)

            log.info(f"Inlining single implementation: {single_name}")
            from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

            result = inline_subgraph_to_ir_nodes(impl_gm, tensor_inputs, name)
            validate_ir(result)
            return result

        # Generate dispatch function for user review (using merged ranges)
        dispatch_func_code = _generate_dispatch_function(
            name=name,
            range_to_best_impl=merged_range_to_best_impl,
            tensor_name=tensor_name,
            dim_index=dim_index,
            op_overload=op_overload,
        )

        # Save to file for debugging/review
        import os

        output_dir = "/tmp/torch_inductor_range_dispatch"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{name}_dispatch.py")

        with open(output_file, "w") as f:
            f.write(dispatch_func_code)

        log.info("Generated dispatch function saved to: %s", output_file)

        # ========================================
        # Option B: Use make_fx + inline_subgraph_to_ir_nodes
        # ========================================
        log.info("Creating runtime dispatch using make_fx tracing")

        # Use merged ranges for compilation
        sorted_ranges = sorted(merged_range_to_best_impl.items())

        # Build dispatch function for tracing
        def build_dispatch_fn_for_tracing():
            def dispatch_fn(*fake_tensors):
                dispatch_tensor = fake_tensors[0]
                dim_value = dispatch_tensor.size(dim_index)

                # Build nested torch.cond
                def build_cond_recursive(ranges_list, idx=0):
                    if idx >= len(ranges_list):
                        raise RuntimeError("No ranges available")

                    (r_start, r_end), (impl_fn, impl_kwargs, impl_name) = ranges_list[
                        idx
                    ]

                    # Last range - no condition
                    if idx == len(ranges_list) - 1:
                        return impl_fn(
                            *fake_tensors, **{**runtime_kwargs, **impl_kwargs}
                        )

                    # Recursive case with torch.cond
                    return torch.cond(
                        pred=dim_value <= r_end,
                        true_fn=lambda: impl_fn(
                            *fake_tensors, **{**runtime_kwargs, **impl_kwargs}
                        ),
                        false_fn=lambda: build_cond_recursive(ranges_list, idx + 1),
                        operands=[],
                    )

                return build_cond_recursive(sorted_ranges, 0)

            return dispatch_fn

        dispatch_fn = build_dispatch_fn_for_tracing()

        # Trace with make_fx to create GraphModule
        from torch.fx.experimental.proxy_tensor import make_fx
        from ..decomposition import select_decomp_table

        log.debug("Tracing dispatch function with make_fx...")

        with V.fake_mode:
            fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)

            decomposition_table = select_decomp_table()
            dispatch_gm = make_fx(
                dispatch_fn,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
            )(*fake_inputs)

        log.debug(
            f"GraphModule created with {len(list(dispatch_gm.graph.nodes))} nodes"
        )

        log.info("Creating SubgraphBuffer with multi-range dispatch capability...")

        from ..ir import FixedLayout, SubgraphBuffer, TensorBox

        range_gms = []

        for (range_start, range_end), (
            impl_fn,
            impl_kwargs,
            perf_time,
        ) in sorted_ranges:
            log.debug(
                f"  Compiling range [{range_start}, {range_end}]: {impl_fn.__name__}"
            )

            # Create wrapper for this specific implementation
            def create_impl_wrapper(fn, kwargs):
                def wrapper(*tensors):
                    return fn(*tensors, **{**runtime_kwargs, **kwargs})

                return wrapper

            impl_wrapper = create_impl_wrapper(impl_fn, impl_kwargs)

            # Trace this implementation independently (no torch.cond!)
            with V.fake_mode:
                impl_gm = make_fx(
                    impl_wrapper,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)

                log.debug(
                    f"    → Generated GraphModule with {len(list(impl_gm.graph.nodes))} nodes"
                )

                # Store (range, GraphModule) tuple
                range_gms.append(((range_start, range_end), impl_gm))

        log.info(f"Compiled {len(range_gms)} range implementations")

        # Step 2: Create unified SubgraphBuffer with multi-range dispatch
        # Passing a list of (range, gm) tuples triggers multi-range mode
        with V.fake_mode:
            fake_output = dispatch_gm(*fake_inputs)
            output_layout = FixedLayout(
                device=fake_output.device,
                dtype=fake_output.dtype,
                size=fake_output.shape,
                stride=fake_output.stride(),
            )

        result = TensorBox.create(
            SubgraphBuffer(
                layout=output_layout,
                input_nodes=tensor_inputs,
                gm=range_gms,  # List of (range, gm) tuples triggers multi-range mode
                example_inputs=list(fake_inputs),
                subgraph_name=f"{name}_autotuned",
                dispatch_dim_index=dim_index,
            )
        )

        log.info(
            f"Created SubgraphBuffer with multi-range dispatch ({len(range_gms)} ranges)"
        )

        validate_ir(result)
        return result

    return range_based_lowering_fn


def register_custom_op_autotuning(
    custom_op: torch._library.custom_ops.CustomOpDef,
    configs: Union[list[CustomOpConfig], list[Callable[..., Any]]],
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dispatch_on: Optional[tuple[str, int]] = None,
    split_points: Optional[list[int]] = None,
) -> None:
    """Register custom op for autotuning.

    Two modes:
    1. Standard autotuning: Benchmark all configs and select the best globally
    2. Range-based autotuning: Benchmark per range and generate runtime dispatch

    Standard Example:
        register_custom_op_autotuning(
            my_attention,
            configs=[
                CustomOpConfig(impl1, head_dim=32),
                CustomOpConfig(impl2, head_dim=64),
            ],
        )

    Range-based Example:
        register_custom_op_autotuning(
            my_op,
            configs=[CustomOpConfig(impl1), CustomOpConfig(impl2), CustomOpConfig(impl3)],
            dispatch_on=("x", 1),  # Dispatch on x[1]
            split_points=[512, 2048],  # Creates ranges: [1,512], [513,2048], [2049,inf]
        )
    """
    from torch._library.custom_ops import CustomOpDef

    if not isinstance(custom_op, CustomOpDef):
        raise TypeError(f"custom_op must be a CustomOpDef, got {type(custom_op)}")

    op_overload = custom_op._opoverload
    default_impl = custom_op._init_fn

    if not isinstance(configs, (list, tuple)):
        raise TypeError(f"configs must be a list or tuple, got {type(configs)}")

    processed_configs = []
    for cfg in configs:
        if isinstance(cfg, CustomOpConfig):
            processed_configs.append(cfg)
        else:
            raise TypeError(
                f"Each config must be a CustomOpConfig object, got {type(cfg)}"
            )

    if not processed_configs:
        raise ValueError("At least one config must be provided")

    if name is None:
        name = f"{op_overload._name}_autotuned"

    # Validate range-based parameters
    is_range_based = dispatch_on is not None or split_points is not None
    if is_range_based:
        if dispatch_on is None or split_points is None:
            raise ValueError(
                "Both dispatch_on and split_points must be specified for range-based autotuning"
            )
        if not isinstance(dispatch_on, tuple) or len(dispatch_on) != 2:
            raise ValueError("dispatch_on must be a tuple of (tensor_name, dim_index)")
        if not isinstance(split_points, list) or len(split_points) == 0:
            raise ValueError("split_points must be a non-empty list of integers")
        if sorted(split_points) != split_points:
            raise ValueError("split_points must be sorted in ascending order")

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        processed_configs=processed_configs,
        default_impl=default_impl,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        dispatch_on=dispatch_on,
        split_points=split_points,
    )

    lowerings[op_overload] = lowering_fn
