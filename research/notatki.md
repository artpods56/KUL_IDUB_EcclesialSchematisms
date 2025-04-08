# Notatki

## Etapy 
OCR - wykonywany przy użyciu modelu SmolDocling-256M-preview
Przetwarzanie tekstu z formatu Docling do czegoś wygodniejszego
Ekstrakcja informacji z tekstu (poszukać artykułu / mały model llm):
- NuExtract (rodzina modeli, można wykorzystać do zadań takich jak rozpoznawanie encji nazwanych (NER), klasyfikacja tekstu czy ekstrakcja relacji) https://huggingface.co/numind/NuExtract-1.5
- GLiNER
Mapowanie wartości dla dostepnych pól (fuzzy search / rule based / regex)


## Baza danych
### Tabele:
- dane_hasła:
    - id
    - dekanat
    - diecezja
    - parafia
    - miejsce
    - typ_obiektu
    - wezwanie
    - wezwanie_parafii (jesli nie jest takie samo jak wezwanie kościoła)
    - material_typ
    - the_geom
    - strona_p
    - strona_k
    - skany
    - faksymile
### SQLite 
``` sql
CREATE TABLE dane_hasla (
    id INTEGER PRIMARY KEY,
    dekanat TEXT,
    diecezja TEXT,
    parafia TEXT,
    miejsce TEXT,
    typ_obiektu TEXT,
    wezwanie TEXT,
    wezwanie_par TEXT,
    material_typ TEXT,
    the_geom TEXT,
    strona_p TEXT,
    strona_k TEXT,
    skany TEXT,
    faksymile TEXT
);
```
## Dataset
- Posiadane dane:
• Zdjecia schematyzmów (wszystkie strony)
• CSV z adnotaciami
• GeoJSON z lokalizacja informacji na stronach
• Mapped possible values dla pól
## Struktura danych
- SQLite zamiast CSV:
• Lepsze query capabilities
• Indeksowanie
• Catwiejszy backup
• Zero setup overhead
• A TODO: zaprojektowaé schemat bazy
## Srodowisko pracy
- Jupyter Notebook:
• Szybsze iteracje
• Wizualizacja danych
• Dokumentacja procesu
• Rozwaz Deepnote (wiecej RAM)
## Wyzwania
1. Ekstrakcja informacji:
• Dekanat
• Plebania
• Wezwania (1000+ wariantow)
2. Mapowanie wezwan:
• Standaryzacja formatów
• Grupowanie hierarchiczne
• Fuzzy matching
• Mapping wariantow histor


