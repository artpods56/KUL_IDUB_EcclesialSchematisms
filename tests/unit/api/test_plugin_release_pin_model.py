import pytest
from pydantic import ValidationError

from grafy_core.domain.plugin_releases import PluginReleaseScope

from grafy_api.v1.models import PluginReleasePinModel


def test_plugin_release_pin_requires_explicit_scope() -> None:
    with pytest.raises(ValidationError, match="scope"):
        PluginReleasePinModel.model_validate(
            {"slug": "notes", "revision": 4}
        )


def test_plugin_release_pin_round_trips_exact_scoped_identity() -> None:
    pin = PluginReleasePinModel(
        scope=PluginReleaseScope.SYSTEM,
        slug="builtin.text",
        revision=2,
    )

    assert pin.model_dump(mode="json") == {
        "scope": "system",
        "slug": "builtin.text",
        "revision": 2,
    }
