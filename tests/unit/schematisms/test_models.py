from notarius_schematisms.domain.models import PageContext, SchematismEntry, SchematismPage


def test_schematism_page_model() -> None:
    page = SchematismPage(
        page_number="1",
        entries=[SchematismEntry(parish="Krakow")],
        context=PageContext(active_deanery="Krakowski"),
    )

    assert page.entries[0].parish == "Krakow"
    assert page.context.active_deanery == "Krakowski"

