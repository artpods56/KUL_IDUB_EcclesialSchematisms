# Gemma 4: błędy structured output w vLLM

Stan na 1 września 2026. Ten dokument oddziela wyniki z `ai-test-ihpan` od
informacji z repozytoriów vLLM, Gemma i XGrammar. Testowany model to
`google/gemma-4-31B-it` w rewizji
`842da3794eaa0b77d5f08bae87a17459d91ff475`.

## Wniosek

Na serwerze występują dwa różne błędy structured output.

1. Outlines w vLLM 0.25.1 odrzuca `<eos>` o identyfikatorze `1`. Ten błąd
   dotyczy zakończenia gramatyki.
2. vLLM 0.28.0 z XGrammar i włączonym chunked prefill przekazuje na GPU starą,
   pełną maskę zamiast aktualnej maski JSON. Model może wtedy wybrać
   niedozwolony pierwszy token. FSM poprawnie odrzuca go po fakcie i API zwraca
   HTTP 500.

Drugi błąd został zawężony do ponownego użycia bloku z cache alokatora pamięci
hosta typu pinned w działającym procesie vLLM. Maska na CPU jest poprawna, lecz
dwie blokujące metody kopiowania tego bloku dają na GPU 8192 niezgodności na
8192 porównanych elementów. Zwolnienie nieużywanych bloków z cache przed nową
alokacją zmienia adres źródła i przywraca poprawne kopie oraz wynik 12/12.
Migawka w zwykłej pamięci hosta również kopiuje się bez niezgodności.

To nie dowodzi ogólnego błędu PyTorch, CUDA, sterownika NVIDIA ani vGPU.
Izolowany test pinned H2D przechodzi w obu obrazach. Problem wymaga stanu
działającego vLLM i ujawnia się przy chunked prefill. Bez dokładnego
publicznego reproduktora nie da się przypisać odpowiedzialności między
PyTorch 2.13, CUDA, sterownikiem 580.173.02 i warstwą time-sliced vGPU.

## Macierz zachowania

Deterministyczna macierz używa skryptu
`docs/research/diagnostics/gemma_structured_output_matrix.py`. Każdy z czterech
przypadków został wykonany trzy razy.

| Wariant | Wynik | Znaczenie |
| --- | --- | --- |
| vLLM 0.25.1, XGrammar | 12/12 HTTP 200 | Punkt odniesienia działa |
| vLLM 0.28.0 MRV1, 16k, chunked prefill wyłączony | 12/12 HTTP 200 | Kontrola zachowania; poprawności maski GPU nie zmierzono |
| vLLM 0.28.0 MRV1, 16k, chunked prefill włączony | 3/12 HTTP 200 | Chunked prefill ujawnia błąd |
| Ten sam wariant, `pin_memory=False` dla maski | 12/12 HTTP 200 | Ominięcie pamięci pinned usuwa błąd w macierzy |
| Ten sam wariant, cache pinned wyczyszczony przed alokacją | 12/12 HTTP 200 | Nowy blok po opróżnieniu cache usuwa błąd |

W błędnym wariancie 0.28.0 wyniki czterech przypadków były następujące:

- `integer_ambiguous`: 0/3, odrzucony pierwszy token `2717`, czyli potrójny
  backtick;
- `integer_no_markdown`: 3/3 po instrukcji, aby zacząć od `{`;
- `classification_ambiguous`: 0/3, odrzucony pierwszy token `228874`, czyli
  `Aby`;
- `classification_no_markdown`: 0/3, odrzucony pierwszy token `1489`, czyli
  `Pro`.

Kontrole bez `response_format` potwierdzają, że są to naturalne pierwsze tokeny
modelu. Instrukcja `zacznij od {` tylko skłania model do samodzielnego wyboru
tokena zgodnego z gramatyką. Nie naprawia constrained decoding.

## Drabina dowodowa

### FSM i maska CPU są poprawne

- Test 0.28.0 nie używa `--reasoning-parser gemma4` ani dodatkowego szablonu.
  Thinking jest wyłączony per request. Log serwera potwierdza
  `reasoning_parser=''`.
- Scheduler wybiera właściwe żądanie i wiersz `0`.
- Aktualna maska oglądana przez NumPy i tensor CPU ma bit `2717=0`. Bit tokenu
  `{` ma wartość `1`. NumPy i tensor CPU wskazują ten sam adres pamięci.
- Oba obrazy mają `xgrammar==0.2.3` i `tokenizers==0.22.2`.
- Izolowany `GrammarMatcher` oraz `apply_grammar_bitmask` na CPU poprawnie
  blokują `2717` i dopuszczają `{` w obu obrazach.
- Funkcja `apply_grammar_bitmask` ma ten sam kod w tagach v0.25.1 i v0.28.0.

Te wyniki wykluczają błędny JSON Schema, kompilację gramatyki, parser thinkingu
i wybór niewłaściwego wiersza schedulera dla tego reproduktora.

### Maska zmienia się między hostem i GPU

Instrumentacja działającego wariantu z chunked prefill porównała trzy kopie
tej samej poprawnej maski CPU:

| Źródło i metoda | Niezgodności na GPU |
| --- | ---: |
| Pinned tensor, blokujące `.to(device)` | 8192/8192 |
| Pinned tensor, `empty_like(...).copy_(..., non_blocking=False)` | 8192/8192 |
| Migawka pageable, blokujące `.to(device)` | 0/8192 |

Obie kopie pinned pokazują na GPU wcześniejszą wartość `-1`, czyli pełną maskę.
Kopia pageable pokazuje aktualną maskę JSON. Globalne `cuda.synchronize()` nie
zmienia wyniku. Bieżący stream i stream domyślny mają identyfikator `0`, więc
nie znaleziono rozjazdu między streamami.

Logity mają typ BF16, są ciągłe i mają stride `(262144, 1)`. Po wywołaniu
XGrammar logit niedozwolonego tokenu pozostaje skończony. Ten wynik jest zgodny
z pełną maską widzianą na GPU. Sam kernel nakłada więc dane, które otrzymał.

### Izolowana kopia nie odtwarza błędu

Mały test pinned H2D przechodzi zarówno w obrazie 0.25.1, jak i 0.28.0. Nie
można na tej podstawie przypisać winy całej implementacji `.to()`, PyTorch,
CUDA, sterownikowi ani vGPU. Różnica pojawia się dopiero w żywym procesie vLLM
po warmupie, gdy host caching allocator ponownie używa bloku dla kolejnych
masek.

### Kontrola cache-clear

W działającym vLLM 0.28.0 przed każdą alokacją pinned wywołano prywatne
`torch._C._host_emptyCache()`. Publicznym odpowiednikiem w PyTorch 2.13 jest
[`torch.accelerator.empty_host_cache()`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/accelerator/memory.py#L795-L806),
które zwalnia nieużywaną pamięć pinned przechowywaną przez host caching
allocator.

- Adres CPU zmienił się z `1169036541952`, czyli wadliwego bloku z cache, na
  `1099687789056`, czyli nowy blok uzyskany po opróżnieniu cache.
- Blokujące pinned `.to(device)` zmieniło wynik z 8192/8192 do 0/8192
  niezgodności.
- Blokujące pinned `empty_like(...).copy_(..., non_blocking=False)` również
  zmieniło wynik z 8192/8192 do 0/8192 niezgodności.
- Kopia pageable pozostała poprawna: 0/8192 niezgodności.
- Najpierw przeszła kontrola 4/4, a następnie pełna macierz 12/12 HTTP 200.
- Kontener nie miał ustawionych `PYTORCH_CUDA_ALLOC_CONF` ani
  `PYTORCH_ALLOC_CONF`, więc test nie korzystał z niestandardowej konfiguracji
  cache pinned.

Wynik silnie wskazuje, że bezpośrednim warunkiem awarii jest ponowne użycie
starej lub niepoprawnej rejestracji pinned host po rozgrzaniu procesu. Nie
rozstrzyga, która warstwa stosu utworzyła ten stan.

## Co potwierdza upstream

### Poprawka stop-tokenów nie obejmuje tego błędu

Gemma 4 ma trzy tokeny końca: `1`, `50` i `106`. Przed poprawką XGrammar znał
tylko `1`, więc `50` lub `106` mogły przerwać JSON przed zakończeniem gramatyki.
vLLM naprawił to w [PR #49227](https://github.com/vllm-project/vllm/pull/49227),
a [informacja o wydaniu 0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
wymienia tę poprawkę wprost.

Token `2717` jest zwykłym tokenem potrójnego backticku. Nie należy do zbioru
tokenów końca, więc #49227 nie naprawia lokalnego reproduktora.

### Backend odrzuca token już wygenerowany przez sampler

W [backendzie XGrammar z vLLM 0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/structured_output/backend_xgrammar.py)
`accept_tokens()` zwraca błąd, gdy `GrammarMatcher.accept_token()` odrzuci już
wygenerowany token. JSON Schema nie dopuszcza backticku ani polskiego słowa
przed otwierającym `{`.

Sampler nie powinien móc wybrać tych tokenów po nałożeniu poprawnej maski.
Lokalna instrumentacja pokazuje teraz konkretny powód: GPU otrzymuje pełną
maskę, więc logit nie zostaje wyzerowany przed losowaniem.

### Pinned staging jest znanym gorącym miejscem, ale nie ma zgłoszenia tej korupcji

[PR vLLM #45424](https://github.com/vllm-project/vllm/pull/45424) zastąpił
czysto NumPy'owy staging maski tensorem hosta w pamięci pinned i asynchroniczną
kopią H2D. [Kod taga v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/structured_output/utils.py#L116-L153)
nadal używa tej sekwencji: alokacja pinned, zapis przez widok NumPy i kopia na
GPU.

Otwarte [zgłoszenie #49013](https://github.com/vllm-project/vllm/issues/49013)
bisekuje około dwukrotną regresję przepustowości structured output dla Gemmy 4
do tego samego fragmentu. Nie opisuje jednak starej zawartości po blokującej
kopii. Zamknięty bez mergowania [PR #49150](https://github.com/vllm-project/vllm/pull/49150)
proponował powrót do czystego NumPy stagingu i jest zbieżny z lokalnym
obejściem pageable. Żaden z tych wątków nie jest dokładnym duplikatem obecnego
błędu poprawności.

### Znane błędy parsera są problemami sąsiednimi

[Recepta Gemma 4 w repozytorium vLLM](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)
zaleca `--reasoning-parser gemma4` i
`--chat-template examples/tool_chat_template_gemma4.jinja`. Otwarty
[problem #50938](https://github.com/vllm-project/vllm/issues/50938) opisuje
prefiks Markdown, który omija gramatykę przed rozpoczęciem kanału thinking.

Lokalny reproduktor nie używa parsera ani dodatkowego szablonu. Problem #50938
nie wyjaśnia więc obecnego A/B. Tak samo [PR #45553](https://github.com/vllm-project/vllm/pull/45553),
który naprawił wyłączenie gramatyki przy `enable_thinking=false`, jest obecny w
obu testowanych tagach.

### Outlines ma osobny błąd zakończenia

W [kodzie Outlines z vLLM 0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/structured_output/backend_outlines.py)
schema JSON jest zamieniana na wyrażenie regularne. Backend opóźnia stan
zakończenia o jeden krok, aby model mógł wygenerować EOS. Lokalnie właśnie ten
EOS został później odrzucony przez FSM. XGrammar pozostaje właściwym backendem
produkcyjnym.

Nie znaleziono publicznego zgłoszenia, które dokładnie łączy vLLM 0.28.0,
chunked prefill, ponowne użycie cached pinned host block i starą zawartość maski
po blokującej kopii H2D. Obecne ustalenia są wynikiem lokalnej instrumentacji,
nie potwierdzoną diagnozą upstream.

## Aktualne hipotezy

Hipotezy dotyczą ostatecznego właściciela błędu. Bezpośredni mechanizm jest już
dobrze zawężony: ponownie użyty blok pinned zawiera na CPU aktualną maskę, ale
GPU otrzymuje zawartość odpowiadającą starej, pełnej masce. Wyczyszczenie cache
i alokacja nowego bloku usuwają rozjazd.

1. Chunked prefill zmienia cykl życia alokacji w żywym pipeline vLLM i prowadzi
   do ponownego użycia wadliwego wpisu lub mapowania z host caching allocator.
2. Ścieżka pageable działa, ponieważ omija tę konkretną rejestrację pinned.
3. Stan może powstawać na styku PyTorch 2.13, CUDA, sterownika 580.173.02 i
   time-sliced vGPU. Obraz 0.25.1 ma PyTorch 2.11.0, a 0.28.0 ma PyTorch 2.13.0.
   Izolowane testy i brak publicznego exact match nie pozwalają przypisać winy
   żadnej z tych warstw.

Blokujące kopie, globalna synchronizacja i wspólny stream osłabiają hipotezę
zwykłego wyścigu asynchronicznego. Poprawny adres CPU i zgodność NumPy z
tensorem osłabiają hipotezę, że instrumentacja czyta inny bufor.

## Następne testy

1. Uruchomić dłuższy test obciążeniowy 0.28.0 z pageable grammar staging,
   blokującą kopią i chunked prefill. Zmierzyć poprawność, przepustowość i CPU.
2. Zebrać statystyki host caching allocator przed rozgrzaniem, po pierwszym
   błędzie i po `empty_host_cache()`. Powiązać je z adresami storage.
3. Przygotować mały live reproduktor, który zachowuje cykl życia i ponowne
   użycie bufora jak vLLM, ale nie ładuje modelu. Izolowany jednorazowy H2D jest
   zbyt słabą kontrolą.
4. Powtórzyć ten reproduktor poza vGPU, o ile dostępny jest fizyczny H200 lub
   porównywalne środowisko. To rozdzieli warstwę wirtualizacji od reszty stosu.
5. Jeżeli reproduktor nadal wskaże cached pinned H2D, zgłosić upstream komplet:
   wersje, konfigurację vGPU, adresy buforów, statystyki alokatora, wynik
   cache-clear, trzy ścieżki kopiowania i macierz chunked prefill.

## Zalecenie na teraz

Do ekstrakcji produkcyjnej należy nadal używać sprawdzonego obrazu vLLM 0.25.1
z XGrammar, thinkingiem wyłączonym per request, walidacją odpowiedzi i retry.
Outlines odpada z powodu odrzucania EOS.

Jeżeli potrzebny jest vLLM 0.28.0, zalecanym patchem jest pageable grammar
staging, czyli `pin_memory=False` dla maski i jawna blokująca kopia. Wariant
z `pin_memory=False` przeszedł macierz 12/12, a osobna blokująca kopia pageable
dała 0/8192 niezgodności. Połączenie obu zabezpieczeń jest rekomendowanym
canary, ale wymaga jeszcze dłuższego testu obciążeniowego i pomiaru kosztu
wydajnościowego.

Wyłączenie chunked prefill także przeszło macierz 12/12, ale w tym wariancie nie
zmierzono maski na GPU. Mogło jedynie zmienić naturalny pierwszy token modelu,
dlatego nie jest potwierdzonym obejściem poprawnościowym. Czyszczenie całego
cache pinned przed każdą maską również nie jest rozwiązaniem produkcyjnym:
służyło do lokalizacji błędu i narzuca koszt świeżej alokacji oraz rejestracji w
gorącej ścieżce.

Instrukcja `zacznij od {` zmniejsza częstość błędu tylko dla promptów, które jej
posłuchają. Nie zastępuje poprawnej maski structured output.
