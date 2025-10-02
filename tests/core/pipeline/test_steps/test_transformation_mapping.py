import pandas as pd
import pytest

from core.pipeline.steps.transformation import DataFrameSchemaMappingStep


def test_rename_only_preserve_unmapped() -> None:
    df = pd.DataFrame({
        "parish": ["A", "B"],
        "deanery": ["D1", "D2"],
        "other": [1, 2],
    })

    step = DataFrameSchemaMappingStep(mapping={"parish": "parafia", "deanery": "dekanat"}, target_columns=None, preserve_unmapped=True)
    out = step.process(df)

    assert list(out.columns) == ["parafia", "dekanat", "other"]
    assert out.loc[0, "parafia"] == "A"
    assert out.loc[1, "dekanat"] == "D2"


def test_rename_and_drop_unmapped() -> None:
    df = pd.DataFrame({
        "parish": ["A"],
        "deanery": ["D1"],
        "other": [1],
    })
    step = DataFrameSchemaMappingStep(mapping={"parish": "parafia", "deanery": "dekanat"}, target_columns=None, preserve_unmapped=False)
    out = step.process(df)

    assert set(out.columns) == {"parafia", "dekanat"}
    assert "other" not in out.columns


def test_target_schema_enforcement() -> None:
    df = pd.DataFrame({
        "parish": ["A"],
        "page_number": ["12"],
    })
    mapping = {"parish": "parafia", "page_number": "strona_p"}
    target = [
        "id", "dekanat", "parafia", "miejsce", "typ_obiektu", "wezwanie", "material_typ",
        "dekanat_kom", "parafia_kom", "miejsce_kom", "typ_obiektu_kom", "wezwanie_kom",
        "material_kom", "varia", "the_geom", "tworca_rekordu", "user_name", "last_date",
        "diecezja", "strona_p", "strona_k", "zakon", "zakon_kom", "valid", "valid_kom",
        "skany", "id_obiekt", "data_rekordu", "material", "numer_skanu", "wezwanie_par",
        "ahp_id", "typ_beneficjum", "typ_beneficjum_kom", "archidiakonat", "patronat_typ",
        "patronat_typ_kom", "rok", "rok_powstania_ob", "rok_kom", "rok_powstania_par",
        "przedzial", "faksymile", "id_church_places",
    ]

    step = DataFrameSchemaMappingStep(mapping=mapping, target_columns=target)
    out = step.process(df)

    # All target columns present in order
    assert list(out.columns) == target
    # Values propagated for mapped columns
    assert out.loc[0, "parafia"] == "A"
    assert out.loc[0, "strona_p"] == "12"
    # Unmapped target columns are NaN
    assert pd.isna(out.loc[0, "id"]) and pd.isna(out.loc[0, "miejsce"])


def test_strict_mode_missing_source_raises() -> None:
    df = pd.DataFrame({"parish": ["A"]})
    step = DataFrameSchemaMappingStep(mapping={"parish": "parafia", "deanery": "dekanat"}, strict=True)
    with pytest.raises(ValueError):
        step.process(df)



