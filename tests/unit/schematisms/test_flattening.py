from notarius_schematisms.data.flattening import FlatteningService
from notarius_schematisms.domain.dataset import BaseMetaData, PredictionDataItem
from notarius_schematisms.domain.models import SchematismEntry, SchematismPage


def test_flatten_prediction_data_item() -> None:
    item = PredictionDataItem(
        image_path=None,
        metadata=BaseMetaData(sample_id=1, schematism_name="s", filename="f"),
        predictions=SchematismPage(entries=[SchematismEntry(parish="A")]),
    )

    flat = FlatteningService.flatten_prediction_data_item(item)

    assert flat[0].parish == "A"
    assert flat[0].schematism_name == "s"

