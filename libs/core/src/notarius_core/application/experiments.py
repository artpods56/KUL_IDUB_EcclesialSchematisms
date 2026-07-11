from copy import deepcopy
from dataclasses import replace
from itertools import product

from notarius_core.domain.models import (
    ExperimentParameter,
    JsonObject,
    JsonValue,
    WorkflowNode,
    WorkflowVersion,
)


def expand_parameter_grid(parameters: list[ExperimentParameter]) -> list[JsonObject]:
    if not parameters:
        return [{}]

    names = [parameter.name for parameter in parameters]
    if len(names) != len(set(names)):
        raise ValueError("Experiment parameters must have unique names")

    variants: list[JsonObject] = []
    value_sets = [parameter.values for parameter in parameters]
    for values in product(*value_sets):
        variants.append(dict(zip(names, values, strict=True)))
    return variants


def apply_experiment_parameters(
    version: WorkflowVersion,
    parameters: list[ExperimentParameter],
    parameter_values: JsonObject,
) -> WorkflowVersion:
    nodes = [
        _node_with_parameter_values(node, parameters, parameter_values)
        for node in version.definition_snapshot.nodes
    ]
    definition = replace(
        version.definition_snapshot,
        nodes=nodes,
        metadata={
            **version.definition_snapshot.metadata,
            "experiment_parameter_values": parameter_values,
        },
    )
    return replace(version, definition_snapshot=definition)


def _node_with_parameter_values(
    node: WorkflowNode,
    parameters: list[ExperimentParameter],
    parameter_values: JsonObject,
) -> WorkflowNode:
    config = deepcopy(node.config)
    for parameter in parameters:
        if parameter.node_id != node.id:
            continue
        if parameter.name not in parameter_values:
            raise ValueError(
                f"Experiment parameter {parameter.name!r} has no selected value"
            )
        config = _config_with_path_value(
            config,
            parameter.config_path,
            parameter_values[parameter.name],
            parameter_name=parameter.name,
            node_id=node.id,
        )
    return replace(node, config=config)


def _config_with_path_value(
    config: JsonObject,
    config_path: tuple[str, ...],
    value: JsonValue,
    *,
    parameter_name: str,
    node_id: str,
) -> JsonObject:
    updated = deepcopy(config)
    current = updated
    for path_part in config_path[:-1]:
        existing = current.get(path_part)
        if existing is None:
            nested: JsonObject = {}
            current[path_part] = nested
            current = nested
        elif isinstance(existing, dict):
            nested = dict(existing)
            current[path_part] = nested
            current = nested
        else:
            path = ".".join(config_path)
            raise ValueError(
                f"Experiment parameter {parameter_name!r} cannot set {path!r} "
                f"on node {node_id!r}: {path_part!r} is not an object"
            )
    current[config_path[-1]] = value
    return updated
