# Kwantyzacja Qwen3.8-27B i Gemma 4 31B

Data sprawdzenia: 2026-08-26

## Decyzja

Najbezpieczniejszy pierwszy wariant to zachować `google/gemma-4-31B-it` w BF16 dla skanów i structured output, a model do kodowania zmienić na oficjalny `Qwen/Qwen3.8-27B-FP8`.

To nie wynika z dowodu, że kod zawsze lepiej znosi kwantyzację niż OCR. Takiego porównania dla tych dwóch modeli i tych dwóch zadań nie ma. Za tym wariantem przemawia jakość dostępnych artefaktów:

- Qwen publikuje własny checkpoint FP8 dla 27B. Model card opisuje blokową kwantyzację FP8 o rozmiarze bloku 128 i deklaruje wyniki niemal identyczne z modelem wyjściowym. Checkpoint jest wskazany jako zgodny z vLLM.
- Qwen nie publikuje oficjalnego AWQ ani GPTQ 4-bit dla Qwen3.8-27B. Oficjalna kolekcja zawiera wariant BF16 i FP8.
- Google publikuje oficjalny Gemma 4 31B QAT W4A16 dla vLLM. To rozsądny drugi etap oszczędzania pamięci, ale Google nie podaje osobnych wyników QAT dla OmniDocBench, skanów historycznych ani waszego schematu danych.
- Gemma wykonuje tu zadanie, w którym ma dwa miejsca utraty jakości: odczyt obrazu i przypisanie odczytanych informacji do właściwych pól. Guided decoding może wymusić poprawny JSON, ale nie może wymusić poprawnej treści pól.

Wniosek jest operacyjny, nie uniwersalny: najpierw kwantyzować Qwena oficjalnym FP8, ponieważ ta zmiana ma najlepsze wsparcie producenta i nie dotyka obecnego toru produkcyjnego dla skanów.

## Co wiadomo z oficjalnych źródeł

| Model i format | Oficjalny artefakt | Rozmiar lub pamięć | Dowód jakości | Ocena dla tego wdrożenia |
|---|---|---:|---|---|
| Qwen3.8-27B BF16 | Tak | Repozytorium 55,6 GB | Bazowe wyniki kodowania: Terminal-Bench 2.1 73,0, SWE-bench Pro 61,7, LiveCodeBench v6 90,3 | Punkt odniesienia |
| Qwen3.8-27B FP8 | Tak | Repozytorium 30,9 GB | Qwen deklaruje metryki niemal identyczne z oryginałem, lecz nie publikuje w karcie osobnych różnic dla każdego benchmarku | Rekomendowany pierwszy ruch |
| Qwen3.8-27B AWQ lub GPTQ 4-bit | Nie w oficjalnej kolekcji Qwen3.8 | Brak oficjalnej wartości | Brak oficjalnego checkpointu i wyników | Nie wdrażać bez własnego testu kodowania |
| Gemma 4 31B BF16 | Tak | Google szacuje 69,9 GB z 20% narzutem na załadowanie wag | OmniDocBench 1.5: 0,131 średniej odległości edycyjnej, mniej znaczy lepiej | Zachować dla skanów na pierwszym etapie |
| Gemma 4 31B SFP8 | Google podaje klasę pamięci, ale w przejrzanych oficjalnych materiałach nie ma osobnego wyniku dla skanów | 34,9 GB według tabeli Google | Brak wyniku quant vs BF16 dla OmniDocBench i schema extraction | Nie jest lepiej udokumentowana od Qwen FP8 |
| Gemma 4 31B QAT W4A16 | Tak, format compressed-tensors dla vLLM | Repozytorium 23,3 GB. Tabela Google podaje 17,5 GB dla Q4_0, ale to inny format i nie jest to gwarancja zużycia procesu vLLM | Google deklaruje jakość podobną do BF16, bez osobnego wyniku dla skanów | Dobry drugi etap, po teście na waszych skanach |

Rozmiary repozytoriów nie są pełnym zużyciem VRAM. Google zaznacza, że jego tabela nie obejmuje kontekstu, KV cache ani całego oprogramowania serwującego. Przejście Qwena z oficjalnego BF16 na oficjalny FP8 zmniejsza jednak same pliki checkpointu o 24,7 GB, czyli około 44%. To daje realną przestrzeń na drugi proces vLLM, pod warunkiem osobnego ograniczenia KV cache i współbieżności obu serwerów.

## BF16, FP8 i 4-bit oznaczają różne poziomy ryzyka

### BF16

BF16 pozostaje punktem odniesienia. Nie usuwa ryzyka błędów OCR ani złego przypisania danych, ale nie dodaje błędu kwantyzacji. Dlatego ma sens dla Gemmy, dopóki wynik skanów nie ma automatycznego testu regresji.

### FP8 i SFP8

FP8 zmniejsza pamięć wag mniej więcej o połowę. H200 należy do rodziny Hopper, a vLLM oficjalnie obsługuje na Hopperze FP8 W8A8. W przypadku Qwen3.8-27B istnieje gotowy checkpoint producenta. Nie trzeba samodzielnie wybierać danych kalibracyjnych ani konwertera.

To jest główny argument za kwantyzacją Qwena. Nie ma jednak liczbowego, apples-to-apples wyniku BF16 vs FP8 dla waszych zadań kodowych. Zdanie o niemal identycznych metrykach pochodzi z model card Qwen, nie z naszego pomiaru.

### QAT W4A16

Google przygotował Gemma 4 31B QAT W4A16 specjalnie dla vLLM. QAT symuluje kwantyzację podczas treningu, więc model może nauczyć się kompensować część utraty precyzji. Google opisuje te modele jako zachowujące jakość podobną do BF16.

Ten artefakt jest lepszym kandydatem niż przypadkowy PTQ AWQ lub GPTQ wykonany po treningu. Nadal brakuje wyników dla dokumentów historycznych. Bazowy model osiąga 0,131 na OmniDocBench 1.5, lecz karta QAT pokazuje tabelę rodziny modeli, a nie osobne porównanie BF16 i QAT na tym benchmarku.

Konfiguracja oficjalnego checkpointu wyłącza z 4-bitowej kwantyzacji wieżę vision, projekcję obrazu i `lm_head`. To ogranicza bezpośrednie ryzyko utraty cech obrazu. Dekoder językowy nadal jest kwantyzowany i to on wybiera treść pól, więc sam wyjątek dla vision tower nie zastępuje testu ekstrakcji.

### AWQ i GPTQ 4-bit

Qwen nie udostępnia oficjalnego wariantu AWQ ani GPTQ dla Qwen3.8-27B. Dostępne checkpointy społecznościowe różnią się kalibracją, zakresem kwantyzowanych modułów i runtime. Nie ma podstaw, by przypisać im deklarację jakości z oficjalnego checkpointu FP8.

Oficjalne materiały Qwen dla wcześniejszej rodziny wyjaśniają, że przy niskiej liczbie bitów dokładność może spaść, a dane kalibracyjne powinny przypominać zadanie docelowe. Dokumentacja wymienia też przypadki, w których GPTQ Int4 powodował niekończącą się lub zdegenerowaną generację. To nie dowodzi błędu Qwen3.8-27B, ale uzasadnia osobny test przed wdrożeniem 4-bit.

## Ryzyko według zadania

### Gemma na skanach i structured output

Baseline Gemmy ma bezpośredni benchmark dokumentowy, OmniDocBench 1.5. Nie ma jednak oficjalnego wyniku kwantyzacji na tym benchmarku. Historyczne skany mogą zawierać nietypowe kroje pisma, zabrudzenia, łamanie kolumn i dawną ortografię, których zbiorczy benchmark nie odtwarza.

Structured output w vLLM wymusza zgodność tokenów z JSON Schema lub gramatyką. Zapewnia to składnię, nie prawdziwość danych. Kwantyzowany model może zwrócić poprawny JSON z błędną miejscowością, datą albo pustym polem. Dla tej usługi miernikiem nie może być wyłącznie odsetek poprawnie parsowanych odpowiedzi.

### Qwen do kodowania

Kodowanie też jest czułe na małe błędy. Jedna zła nazwa lub warunek może zepsuć program. Różnica polega na dostępności oficjalnego FP8 i łatwiejszym teście końcowym. Kompilator, testy i benchmark z prawdziwych zadań potrafią jednoznacznie wykryć wiele regresji.

Dlatego nie zakładam, że coding jest z natury odporniejszy. Twierdzę jedynie, że oficjalny Qwen FP8 ma mniejsze ryzyko wdrożeniowe i łatwiejszą bramkę jakości niż zmiana precyzji modelu odczytującego skany.

## Zalecany test przed zmianą produkcyjną

1. Uruchomić `Qwen/Qwen3.8-27B-FP8` obok zapisanych odpowiedzi BF16 na stałym zestawie rzeczywistych zadań kodowych.
2. Mierzyć odsetek zadań zakończonych przejściem testów, liczbę potrzebnych poprawek, czas odpowiedzi i błędy runtime. Samo podobieństwo tekstu odpowiedzi nie wystarczy.
3. Zachować Gemmę BF16 w pierwszym wariancie dwóch serwerów.
4. Jeżeli nadal brakuje VRAM, porównać Gemma BF16 z oficjalnym `google/gemma-4-31B-it-qat-w4a16-ct` na tych samych skanach. Mierzyć dokładność każdego pola, CER lub odległość edycyjną transkrypcji, brakujące pola i poprawność JSON.
5. Nie łączyć tego testu ze speculative decoding. Zmiana wag i zmiana sposobu dekodowania w jednym eksperymencie uniemożliwią wskazanie przyczyny regresji.

## Granice wniosków

Nie znalazłem oficjalnego, wspólnego eksperymentu, który porównuje:

- Qwen3.8-27B BF16 i FP8 na tym samym zestawie zadań kodowych z pełnymi wynikami obu wariantów;
- Gemma 4 31B BF16, SFP8 i QAT W4A16 na OmniDocBench 1.5;
- którykolwiek wariant na polskich skanach historycznych i waszym JSON Schema.

Nie da się więc uczciwie policzyć, który model straci więcej procent jakości. Można wybrać zmianę z lepszym oficjalnym artefaktem i łatwiejszym testem regresji. Dziś jest nią Qwen3.8-27B FP8.

Orientacyjny rachunek dla dwóch modeli to 69,9 GB Gemmy BF16 według szacunku Google, 30,9 GB plików Qwena FP8 i około 4,1 GB obecnych embeddingów. Daje to około 104,9 GB przed KV cache i różnicami runtime. Na H200 141 GB pozostaje około 36 GB. To wygląda wykonalnie po ograniczeniu kontekstu i współbieżności, ale wymaga pomiaru po uruchomieniu, ponieważ składniki rachunku pochodzą z dwóch różnych metod raportowania pamięci.

## Źródła pierwotne

- [Oficjalny model card Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Pliki oficjalnego Qwen3.8-27B-FP8, 30,9 GB](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/main)
- [Pliki oficjalnego Qwen3.8-27B BF16, 55,6 GB](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main)
- [Oficjalna kolekcja Qwen3.8](https://huggingface.co/collections/Qwen/qwen38)
- [Oficjalny model card Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it)
- [Dokumentacja Google Gemma 4 z tabelą pamięci i opisem QAT](https://ai.google.dev/gemma/docs/core)
- [Oficjalny Gemma 4 31B QAT W4A16 dla vLLM](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)
- [Konfiguracja Gemma 4 31B QAT W4A16 z zakresem wyłączeń z kwantyzacji](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/main/config.json)
- [Oficjalna kolekcja Gemma 4 QAT](https://huggingface.co/collections/google/gemma-4-qat-q4-0)
- [Dokumentacja vLLM o kwantyzacji i wsparciu Hopper](https://docs.vllm.ai/en/stable/features/quantization/)
- [Dokumentacja vLLM o structured outputs](https://docs.vllm.ai/en/v0.15.0/features/structured_outputs/)
- [Oficjalna dokumentacja Qwen o kwantyzacji i danych kalibracyjnych](https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/llama.cpp.md)
- [Oficjalna dokumentacja Qwen o znanych problemach GPTQ Int4](https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/gptq.md)
