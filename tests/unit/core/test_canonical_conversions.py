import ast
from hashlib import sha256
from inspect import getsource
import json
from textwrap import dedent
from typing import cast

import pytest

from grafy_core.artifact_contracts import INTEGER_VALUE, TEXT_VALUE
from grafy_core.canonical_conversions import (
    CANONICAL_ARTIFACT_CONVERSIONS,
    CANONICAL_ARTIFACT_CONVERSIONS_BY_KEY,
    INTEGER_TO_TEXT,
)


def test_integer_to_text_is_an_exact_deployment_owned_conversion() -> None:
    assert CANONICAL_ARTIFACT_CONVERSIONS == (INTEGER_TO_TEXT,)
    assert CANONICAL_ARTIFACT_CONVERSIONS_BY_KEY == {
        INTEGER_TO_TEXT.key: INTEGER_TO_TEXT
    }
    assert INTEGER_TO_TEXT.key.id == "builtin.scalar.integer_to_text"
    assert INTEGER_TO_TEXT.key.version == 1
    assert INTEGER_TO_TEXT.source == INTEGER_VALUE.key
    assert INTEGER_TO_TEXT.target == TEXT_VALUE.key
    assert INTEGER_TO_TEXT.source_type is int
    assert INTEGER_TO_TEXT.target_type is str
    assert INTEGER_TO_TEXT.title == "As text"
    assert INTEGER_TO_TEXT.convert(42) == "42"


def test_integer_to_text_contract_and_implementation_require_a_version_bump() -> None:
    implementation = ast.dump(
        ast.parse(dedent(getsource(INTEGER_TO_TEXT.convert))),
        include_attributes=False,
    )
    snapshot = {
        "key": [INTEGER_TO_TEXT.key.id, INTEGER_TO_TEXT.key.version],
        "source": [INTEGER_TO_TEXT.source.id, INTEGER_TO_TEXT.source.schema_version],
        "target": [INTEGER_TO_TEXT.target.id, INTEGER_TO_TEXT.target.schema_version],
        "source_type": INTEGER_TO_TEXT.source_type.__qualname__,
        "target_type": INTEGER_TO_TEXT.target_type.__qualname__,
        "title": INTEGER_TO_TEXT.title,
        "implementation": implementation,
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()

    assert sha256(encoded).hexdigest() == (
        "0e8d88b5be422ce9fb74e4f080ad368934df534cb1efe78aae0d59ad6fdee17e"
    )


def test_canonical_conversion_registry_is_immutable() -> None:
    mutable_view = cast(dict[object, object], CANONICAL_ARTIFACT_CONVERSIONS_BY_KEY)

    with pytest.raises(TypeError):
        mutable_view[INTEGER_TO_TEXT.key] = INTEGER_TO_TEXT
