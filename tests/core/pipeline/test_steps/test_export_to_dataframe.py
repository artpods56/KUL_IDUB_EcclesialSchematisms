import pandas as pd
import pytest

from core.pipeline.steps.export import SaveDataFrameStep
from core.pipeline.steps.transformation import ToPandasDataFrameStep
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage, SchematismEntry


def _entry(
    parish: str,
    deanery: str | None = None,
    dedication: str | None = None,
    building_material: str | None = None,
) -> SchematismEntry:
    return SchematismEntry(
        parish=parish,
        deanery=deanery,
        dedication=dedication,
        building_material=building_material,
    )


def test_to_dataframe_auto_mode_mixes_sources() -> None:
    item0 = PipelineData(
        image=None,
        ground_truth=None,
        parsed_prediction=SchematismPage(
            page_number="1",
            entries=[_entry("p1", "d1"), _entry("p2", None, "ded2", "wood")],
        ),
        metadata={"file_name": "a.jpg", "region": "mazowieckie"},
    )

    item1 = PipelineData(
        image=None,
        ground_truth=None,
        llm_prediction=SchematismPage(page_number="2", entries=[_entry("p3", "d3")]),
        metadata={"file_name": "b.jpg"},
    )

    step = ToPandasDataFrameStep(source="auto", include_metadata=True)
    df = step.process([item0, item1])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    # check sources and sample_index propagation
    assert set(df["source"]) == {"parsed", "llm"}
    assert set(df["sample_index"]) == {0, 1}

    # check required columns
    for col in [
        "page_number",
        "parish",
        "deanery",
        "dedication",
        "building_material",
        "file_name",
    ]:
        assert col in df.columns

    # verify specific row content
    first_row = df.iloc[0]
    assert first_row["page_number"] == "1"
    assert first_row["parish"] in {"p1", "p2"}
    assert first_row["file_name"] == "a.jpg"


def test_metadata_keys_filtering() -> None:
    item = PipelineData(
        image=None,
        ground_truth=SchematismPage(page_number="10", entries=[_entry("pX", "dX")]),
        metadata={"file_name": "x.jpg", "region": "małopolskie"},
    )

    step = ToPandasDataFrameStep(
        source="ground_truth", include_metadata=True, metadata_keys=["file_name"]
    )  # only include file_name
    df = step.process([item])

    assert "file_name" in df.columns
    assert "region" not in df.columns
    assert df.loc[0, "file_name"] == "x.jpg"


def test_empty_dataset_returns_empty_dataframe_with_expected_columns() -> None:
    # No entries available
    item = PipelineData(
        image=None,
        ground_truth=SchematismPage(page_number="1", entries=[]),
        metadata={},
    )

    step = ToPandasDataFrameStep(
        source="ground_truth",
        include_metadata=True,
        metadata_keys=["file_name", "region"],
    )  # expect these columns present
    df = step.process([item])

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    for col in [
        "sample_index",
        "source",
        "page_number",
        "parish",
        "deanery",
        "dedication",
        "building_material",
        "file_name",
        "region",
    ]:
        assert col in df.columns


def test_explicit_source_selection() -> None:
    # Provide multiple sources; enforce ground_truth
    item = PipelineData(
        image=None,
        ground_truth=SchematismPage(page_number="GT", entries=[_entry("pg")]),
        llm_prediction=SchematismPage(page_number="LLM", entries=[_entry("pl")]),
        metadata={},
    )

    step = ToPandasDataFrameStep(source="ground_truth")
    df = step.process([item])

    assert len(df) == 1
    assert df.loc[0, "source"] == "ground_truth"
    assert df.loc[0, "page_number"] == "GT"


def test_save_dataframe_to_csv(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "out.csv"

    step = SaveDataFrameStep(file_path=out, file_format="csv", overwrite=False)
    result = step.process(df)

    assert out.exists()
    assert result is df

    loaded = pd.read_csv(out)
    # Index is not included by default
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2


def test_save_dataframe_to_excel(tmp_path) -> None:
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    out = tmp_path / "out.xlsx"

    step = SaveDataFrameStep(
        file_path=out, file_format="excel", overwrite=True, excel_sheet_name="Data"
    )
    _ = step.process(df)

    assert out.exists()
    # Spot check readback
    loaded = pd.read_excel(out, sheet_name="Data")
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 1


def test_save_dataframe_disallow_overwrite(tmp_path) -> None:
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "exists.csv"
    df.to_csv(out, index=False)

    step = SaveDataFrameStep(file_path=out, file_format="csv", overwrite=False)
    with pytest.raises(FileExistsError):
        step.process(df)
