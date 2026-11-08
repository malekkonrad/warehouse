# Plan realizacji projektu: własne środowisko Gymnasium

## 1. Tytuł projektu
**WarehouseRobotEnv – środowisko robota magazynowego z ciągłą przestrzenią akcji**

---

## 2. Cel projektu
Celem projektu jest stworzenie własnego środowiska zgodnego z biblioteką **Gymnasium**, w którym agent steruje robotem poruszającym się po magazynie i realizuje zadanie transportu paczek do bazy.

Środowisko ma spełniać wymagania projektu, w szczególności:
- implementację własnego środowiska,
- rozwiązanie problemu przez agenta,
- widoczne działanie według strategii,
- wykorzystanie przestrzeni ciągłej,
- możliwość rozszerzenia o tryb graficzny.

---

## 3. Opis wymyślonego środowiska
Środowisko przedstawia magazyn 2D, w którym:
- robot startuje w losowej lub ustalonej pozycji,
- na mapie znajdują się przeszkody reprezentujące regały lub ściany,
- rozmieszczonych jest kilka paczek do zebrania,
- istnieje strefa bazy, do której należy dostarczać paczki.

Robot może przenosić tylko jedną paczkę naraz. Podniesienie paczki następuje automatycznie, gdy robot spełnia **jednocześnie dwa warunki**: znajduje się w zasięgu `r_pickup` od paczki **oraz** jego prędkość (norma wektora `v`) jest mniejsza niż próg `v_pickup_max`. Analogiczny warunek dotyczy dostarczenia do bazy – robot musi mieć `carrying=True`, znajdować się w zasięgu `r_delivery` i poruszać się wolniej niż `v_pickup_max`. Taka zasada wymusza na agencie zwolnienie przy paczce/bazie, co daje bardziej realistyczne i „strategiczne" zachowanie niż samo przelatywanie przez punkty.

Celem agenta jest dostarczenie wszystkich paczek do bazy w możliwie krótkim czasie, unikając kolizji z przeszkodami i ścianami.

---

## 4. Dlaczego ten pomysł jest dobry
Wybrane środowisko:
- jest oryginalne i nie stanowi prostej kopii klasycznych środowisk Gymnasium,
- dobrze nadaje się do ciągłej przestrzeni akcji,
- pozwala pokazać sensowne, strategiczne zachowanie agenta,
- daje się łatwo rozbudować o wizualizację,
- umożliwia przeprowadzenie kilku ciekawych eksperymentów.

---

## 5. Model środowiska

> **Uwaga dotycząca pierwszeństwa specyfikacji**: sekcje 5-10 opisują koncepcję środowiska na wysokim poziomie. **Wiążąca specyfikacja techniczna jest w sekcjach 15-18**. W razie rozbieżności między opisem koncepcyjnym a specyfikacją techniczną obowiązuje treść z sekcji 15-18 (obsługa kolizji, konfiguracja stałych, API środowiska).

### 5.1. Reprezentacja mapy
Mapa będzie prostokątnym obszarem 2D o zadanym rozmiarze, np. `10 x 10` lub `20 x 20` jednostek.

Na mapie będą znajdować się:
- granice środowiska,
- przeszkody prostokątne,
- paczki,
- baza,
- robot.

### 5.2. Stan robota
Robot będzie opisany przez:
- pozycję `(x, y)`,
- prędkość `(vx, vy)`,
- informację, czy niesie paczkę.

### 5.3. Dynamika ruchu
Akcja agenta będzie interpretowana jako sterowanie przyspieszeniem w osiach X i Y.

Przykładowy model ruchu:
- `a = [ax, ay]`, gdzie `ax, ay ∈ [-1, 1]`,
- aktualizacja prędkości na podstawie akcji,
- ograniczenie maksymalnej prędkości,
- aktualizacja pozycji na podstawie prędkości i kroku czasowego `dt`.

Dzięki temu ruch robota będzie płynny i ciągły.

---

## 6. Przestrzeń akcji
Przestrzeń akcji będzie ciągła:

```python
spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

Interpretacja akcji:
- pierwszy wymiar – przyspieszenie w osi X,
- drugi wymiar – przyspieszenie w osi Y.

Takie sterowanie dobrze pasuje do algorytmu **PPO**.

---

## 7. Przestrzeń obserwacji
Obserwacja będzie wektorem cech ciągłych opisujących bieżącą sytuację robota.

Proponowane elementy obserwacji:
- pozycja robota: `x, y`,
- prędkość robota: `vx, vy`,
- flaga `carrying` – czy robot niesie paczkę,
- wektor do aktualnego celu: `dx_target, dy_target`,
- odległość do aktualnego celu,
- odległości z 8 czujników do najbliższej przeszkody lub ściany,
- liczba pozostałych paczek.

Aktualny cel:
- jeśli robot nie niesie paczki – celem jest najbliższa paczka,
- jeśli robot niesie paczkę – celem jest baza.

Wszystkie wartości będą znormalizowane, aby ułatwić trening agenta.

---

## 8. Nagroda (reward function)
Funkcja nagrody powinna promować skuteczne i strategiczne działanie.

Proponowana funkcja nagrody:
- dodatnia nagroda za podniesienie paczki,
- duża dodatnia nagroda za dostarczenie paczki do bazy,
- mała kara za każdy krok, aby zachęcać do szybkiego działania,
- kara za kolizję z przeszkodą lub ścianą,
- nagroda za zbliżanie się do aktualnego celu,
- kara za oddalanie się od celu,
- dodatkowa nagroda za ukończenie całego zadania.

Przykładowe wartości:
- `+20` za podniesienie paczki,
- `+100` za dostarczenie paczki,
- `-0.1` za krok,
- `-5` lub `-10` za kolizję,
- `β * (d_poprzednie - d_aktualne)` jako składnik postępu,
- `+200` za dostarczenie wszystkich paczek.

---

## 9. Warunki zakończenia epizodu
Epizod kończy się, gdy:
- wszystkie paczki zostaną dostarczone (`terminated=True`, sukces),
- zostanie osiągnięty limit kroków (`truncated=True`).

Kolizja **nie kończy** epizodu – skutkuje tylko karą i zatrzymaniem robota (patrz sekcja 15.3). Szczegóły flag `terminated`/`truncated` opisane są w sekcji 18.5.

---

## 10. Algorytm rozwiązujący problem
Do rozwiązania problemu zostanie użyty algorytm **PPO (Proximal Policy Optimization)**.

Powody wyboru:
- dobrze współpracuje z ciągłą przestrzenią akcji,
- jest stabilny i popularny w zadaniach sterowania,
- jest łatwo dostępny w bibliotece `stable-baselines3`,
- pozwala uzyskać płynne zachowanie agenta.

---

## 11. Plan implementacji

### Etap 1 – przygotowanie środowiska
1. Utworzenie klasy `WarehouseRobotEnv` dziedziczącej po `gym.Env`.
2. Zdefiniowanie `action_space` oraz `observation_space`.
3. Zaimplementowanie metod:
   - `__init__()`,
   - `reset()`,
   - `step(action)`,
   - `render()`,
   - `close()`.
4. Utworzenie prostego modelu mapy z przeszkodami, bazą i paczkami.
5. Dodanie logiki kolizji i transportu paczek.

### Etap 2 – trening agenta PPO
1. Podłączenie środowiska do `stable-baselines3` przez `make_vec_env`.
2. Uruchomienie treningu PPO z hiperparametrami z sekcji 17.7.
3. Monitorowanie rewardu i skuteczności agenta przez TensorBoard.
4. Zapis wytrenowanego modelu do `models/best_model.zip`.

### Etap 3 – eksperymenty
1. Porównanie działania dla różnej liczby paczek.
2. Porównanie działania dla różnych układów przeszkód.
3. Sprawdzenie wpływu shaping rewardu.
4. Analiza wpływu limitu kroków i rozmiaru mapy.
5. Porównanie zachowania agenta przed i po treningu.

### Etap 4 – wizualizacja
1. Dodanie prostego renderowania 2D, np. w `pygame`.
2. Oznaczenie robota, przeszkód, paczek i bazy różnymi kolorami.
3. Możliwość oglądania działania wytrenowanego agenta.

### Etap 5 – sprawozdanie w formie notebooka Jupytera
Sprawozdanie ma formę **wykonywalnego notebooka `sprawozdanie.ipynb`**, a nie statycznego PDF-a. Notebook po uruchomieniu od góry do dołu powinien:
- zaimportować środowisko i wszystkie potrzebne biblioteki,
- pokazać działanie środowiska na kilku przykładowych krokach (walidacja przez `check_env`),
- wczytać wytrenowany model z `models/best_model.zip` i uruchomić kilka epizodów ewaluacyjnych,
- wygenerować wykresy krzywych uczenia z logów TensorBoard lub zapisanych metryk,
- wyświetlić statystyki (średni reward, liczba dostarczonych paczek, liczba kolizji),
- opcjonalnie pokazać wizualizację `rgb_array` wybranych epizodów jako obrazki w notebooku.

Struktura komórek notebooka:
1. **Markdown**: opis problemu i celu projektu.
2. **Kod + Markdown**: opis środowiska, przestrzeni akcji i obserwacji – z fragmentami kodu pokazującymi `action_space`, `observation_space` i przykładową obserwację.
3. **Kod**: uruchomienie `check_env()` i kilku losowych epizodów jako dowód poprawności.
4. **Markdown**: opis funkcji nagrody z tabelą składników.
5. **Markdown**: opis algorytmu PPO i uzasadnienie wyboru.
6. **Kod**: wczytanie wytrenowanego modelu, ewaluacja na N epizodach, tabela wyników.
7. **Kod**: wykresy krzywych uczenia (reward, długość epizodu, liczba kolizji w czasie treningu).
8. **Kod**: porównanie eksperymentów (np. różna liczba paczek, różne układy mapy).
9. **Markdown**: wnioski końcowe.

Dzięki takiej formie osoba oceniająca może **samodzielnie zweryfikować działanie** bez przeklejania kodu – wystarczy uruchomić notebook w środowisku `uv` przez `uv run jupyter lab` (lub `uv run jupyter notebook`).

Wymagania techniczne:
- Notebook musi działać **od góry do dołu bez błędów** po wczytaniu wytrenowanego modelu.
- Ścieżki do modelu i logów powinny być względne (`models/best_model.zip`, `logs/`).
- Wszystkie wykresy generowane są w locie z danych – nie ma osadzania statycznych obrazków.
- Długie treningi NIE są uruchamiane w notebooku (tego robi `train/train_ppo.py`); notebook tylko wczytuje i analizuje wyniki.

---

## 12. Proponowana struktura plików projektu

```text
project/
├── env/
│   └── warehouse_robot_env.py
├── train/
│   └── train_ppo.py
├── models/
│   └── best_model.zip
├── render/
│   └── play_agent.py
├── logs/
│   └── ppo_warehouse/           # logi TensorBoard z treningu
├── report/
│   └── sprawozdanie.ipynb       # wykonywalny notebook Jupytera
├── pyproject.toml               # zależności zarządzane przez uv
├── uv.lock                      # zablokowane wersje zależności
└── README.md
```

Notebook `sprawozdanie.ipynb` jest głównym dokumentem końcowym – po uruchomieniu wczytuje model z `models/` i logi z `logs/`, wykonuje ewaluację, generuje wykresy i prezentuje wyniki. Osoba oceniająca uruchamia go przez `uv run jupyter lab report/sprawozdanie.ipynb` i może samodzielnie zweryfikować działanie.

---

## 13. Możliwe rozszerzenia
Po wykonaniu podstawowej wersji można rozważyć:
- większą liczbę paczek,
- bardziej złożone mapy,
- dynamiczne przeszkody,
- ograniczenie energii robota,
- kilka rodzajów paczek,
- losowanie układu magazynu przy `reset()`.

Rozszerzenia nie są konieczne na początku i powinny być dodawane dopiero po uzyskaniu działającej wersji bazowej.

---

## 14. Ryzyka i jak je ograniczyć

### Ryzyko 1: agent nie uczy się skutecznie
Rozwiązanie:
- uprościć mapę,
- zmniejszyć liczbę paczek,
- poprawić funkcję nagrody,
- lepiej znormalizować obserwacje.

### Ryzyko 2: środowisko jest zbyt trudne
Rozwiązanie:
- zacząć od jednej paczki i małej liczby przeszkód,
- stopniowo zwiększać trudność.

### Ryzyko 3: problemy z kolizjami i fizyką ruchu
Rozwiązanie:
- zastosować prosty model ruchu (przyspieszenie → prędkość → pozycja),
- robot jest reprezentowany jako okrąg (szczegóły w sekcji 15.1),
- unikać nadmiernie skomplikowanej fizyki (bez odbić, bez tarcia).

---

## 15. Doprecyzowanie obsługi kolizji

### 15.1. Model robota i przeszkód
Robot jest reprezentowany jako **okrąg** o promieniu `r_robot` (np. `0.3` jednostki). Taka reprezentacja znacznie upraszcza detekcję kolizji w porównaniu do prostokąta z rotacją, a wizualnie wygląda naturalnie.

Przeszkody to **prostokąty osiowo wyrównane** (AABB – axis-aligned bounding boxes), opisane przez `(x_min, y_min, x_max, y_max)`. Granice mapy traktowane są analogicznie do przeszkód.

Paczki i baza to **okręgi** z promieniem odbioru (`r_pickup`, `r_delivery`) i **nie biorą udziału w kolizjach fizycznych** – robot przechodzi przez nie swobodnie, a zbliżenie się do paczki powoduje jej podniesienie.

### 15.2. Detekcja kolizji okrąg–prostokąt
Standardowy algorytm: znajdź najbliższy punkt prostokąta do środka okręgu i sprawdź, czy odległość jest mniejsza niż promień.

```python
def circle_rect_collision(cx, cy, r, x_min, y_min, x_max, y_max):
    closest_x = np.clip(cx, x_min, x_max)
    closest_y = np.clip(cy, y_min, y_max)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) < r*r
```

Algorytm jest szybki i wektoryzowalny przez NumPy, co pozwala sprawdzać kolizje z wieloma przeszkodami jednocześnie.

### 15.3. Strategia reakcji na kolizję – „stop + kara"
Wybrane podejście:
1. Obliczana jest pozycja kandydata po zastosowaniu akcji:
   - `v_new = clip(v + a * dt, -v_max, v_max)`
   - `pos_new = pos + v_new * dt`
2. Sprawdzana jest kolizja `pos_new` z przeszkodami oraz wyjście poza mapę.
3. **Jeśli kolizja**:
   - pozycja NIE jest aktualizowana (robot zostaje w `pos`),
   - prędkość jest zerowana: `v = 0`,
   - nakładana jest kara: `reward -= 5`,
   - flaga `info["collision"] = True` dla statystyk.
4. **Jeśli brak kolizji**:
   - `pos = pos_new`, `v = v_new`.

Uzasadnienie wyboru:
- Odbijanie się komplikuje fizykę (wyznaczanie normalnej do ściany, rozkład prędkości) i daje dziwne zachowania blisko rogów.
- Kończenie epizodu przy kolizji sprawia, że na początku treningu agent ciągle „umiera" po kilku krokach, co ogranicza eksplorację i uniemożliwia działanie reward shapingu.
- Zerowanie prędkości jest kluczowe – bez tego agent mógłby „przyklejać się" do ściany, naciskając w nią cały czas (prędkość akumulowałaby się w kierunku ściany, a pozycja by się nie zmieniała, co dawałoby dziwny gradient).

### 15.4. Granice mapy
Granice mapy traktowane są identycznie jak przeszkody: jeśli `pos_new` wychodzi poza `[0, map_size]` w którejkolwiek osi, to jest to kolizja (stop + kara). Dzięki temu jest jedna spójna logika, a robot nie może opuścić mapy.

Alternatywa w postaci `clip` pozycji do granic została odrzucona, ponieważ traci sygnał kary za próbę wyjścia – a to jest ważny sygnał dla agenta, szczególnie na początku treningu.

### 15.5. Czujniki odległości w obserwacji
Zamiast pełnego raycastingu zastosowana zostanie **uproszczona wersja przez sampling** – dla każdego z 8 kierunków (co 45°) zwracana jest odległość do kolizji wzdłuż promienia, ograniczona do `d_max` (np. 3 jednostki zasięgu).

```python
def sensor_reading(pos, angle, obstacles, d_max, step=0.1):
    direction = np.array([np.cos(angle), np.sin(angle)])
    for d in np.arange(step, d_max, step):
        probe = pos + direction * d
        if any_collision(probe, obstacles):
            return d / d_max  # znormalizowane [0, 1]
    return 1.0  # brak przeszkody w zasięgu
```

Parametry:
- 8 czujników × zasięg `d_max=3` × krok `step=0.1` = 240 sprawdzeń kolizji na krok,
- wynik znormalizowany do `[0, 1]` (0 = kolizja tuż obok, 1 = wolna droga),
- w razie spowolnienia można zwiększyć `step` do `0.2` lub zmniejszyć zasięg.

### 15.6. Kalibracja kary za kolizję
Wartość startowa `-5` jest dopasowana do skali pozostałych nagród:
- reward shaping postępu: `β * Δd` z `β ≈ 1`, czyli typowo ~`±0.1` na krok,
- kara za krok: `-0.1`,
- kolizja: `-5`, co odpowiada ~50 krokom ruchu.

Dostrajanie:
- jeśli agent stanie się zbyt „bojaźliwy" (stoi w miejscu, unika ścian) – zmniejszyć do `-2` lub `-1`,
- jeśli agent ciągle uderza – zwiększyć do `-10`.

Istotne: **epizod NIE kończy się na kolizji**, nawet przy wielu kolizjach pod rząd. Liczba kolizji jest tylko logowana w `info` dla statystyk.

### 15.7. Kolejność implementacji środowiska
Aby łatwiej debugować, środowisko budowane będzie przyrostowo:
1. Pusty magazyn – robot, paczki, baza, bez przeszkód. Test: czy PPO uczy się zbierać paczki i dostarczać je do bazy.
2. Dodanie granic mapy i kolizji ze ścianami.
3. Dodanie przeszkód wewnątrz mapy i czujników odległości w obserwacji.

Każdy etap powinien zostać przetestowany przez `gymnasium.utils.env_checker.check_env()` oraz ręczne uruchomienie kilku epizodów z losowymi akcjami.

---

## 16. Środowisko uruchomieniowe i zależności

Projekt korzysta z menedżera pakietów **`uv`**, który zarządza środowiskiem wirtualnym i zależnościami. Biblioteki są **już zainstalowane** w środowisku projektu – nie ma potrzeby ich osobnej instalacji.

Wszystkie skrypty uruchamiane są przez `uv run`, np.:
- trening: `uv run python train/train_ppo.py`,
- odtworzenie agenta: `uv run python render/play_agent.py`,
- testy środowiska: `uv run python -m env.warehouse_robot_env`,
- sprawozdanie (notebook): `uv run jupyter lab report/sprawozdanie.ipynb`.

Wymagane biblioteki (dostępne w środowisku `uv`):

| Biblioteka | Minimalna wersja | Zastosowanie |
|---|---|---|
| `gymnasium` | `>=0.29.0` | API środowiska RL |
| `stable-baselines3` | `>=2.3.0` | implementacja PPO |
| `numpy` | `>=1.24.0` | operacje wektorowe |
| `torch` | `>=2.0.0` | backend sieci neuronowych |
| `pygame` | `>=2.5.0` | tryb graficzny |
| `tensorboard` | `>=2.14.0` | logowanie treningu |
| `matplotlib` | `>=3.7.0` | wykresy do sprawozdania |
| `tqdm` | `>=4.65.0` | paski postępu |
| `jupyter` | `>=1.0.0` | uruchamianie sprawozdania w formie notebooka |

**Wersja API Gymnasium**: projekt używa **nowego API `gymnasium` (>=0.29)**, a nie przestarzałego `gym`. Oznacza to konkretnie:
- `reset(seed=None, options=None) -> (obs, info)` – zwraca krotkę dwuelementową,
- `step(action) -> (obs, reward, terminated, truncated, info)` – zwraca krotkę pięcioelementową (rozdzielenie `terminated` i `truncated`),
- importy w kodzie: `import gymnasium as gym`, `from gymnasium import spaces`.

---

## 17. Konfiguracja domyślna – tabela stałych

Wszystkie stałe środowiska są zebrane w jednym miejscu. Wartości są punktem startowym – można je dostrajać w eksperymentach.

### 17.1. Parametry mapy i obiektów

| Stała | Wartość | Opis |
|---|---|---|
| `map_size` | `10.0` | rozmiar mapy (kwadrat `10×10` jednostek) |
| `r_robot` | `0.3` | promień okręgu reprezentującego robota |
| `r_pickup` | `0.5` | promień zasięgu podnoszenia paczki |
| `r_delivery` | `0.7` | promień strefy bazy |
| `n_packages` | `3` | liczba paczek na starcie |
| `n_obstacles` | `2` | liczba prostokątnych przeszkód |

### 17.2. Parametry dynamiki

| Stała | Wartość | Opis |
|---|---|---|
| `dt` | `0.1` | krok czasowy symulacji |
| `v_max` | `2.0` | maksymalna prędkość robota (jednostki/s) |
| `a_max` | `1.0` | maksymalne przyspieszenie (skala akcji) |
| `v_pickup_max` | `0.5` | próg prędkości przy podnoszeniu/dostarczeniu paczki |

**Dlaczego `v_pickup_max = 0.5`**: stanowi 25% `v_max`, co wymusza na agencie zwolnienie przy paczce bez konieczności pełnego zatrzymania. Próg zbyt niski (np. `0.1`) byłby trudny do wyuczenia – agent musiałby praktycznie się zatrzymywać. Próg zbyt wysoki (np. `1.5`) czyniłby warunek trywialnym. Wartość `0.5` jest kompromisem – wymaga „strategicznego" zwolnienia, ale nie blokuje naturalnego ruchu.

### 17.3. Parametry epizodu

| Stała | Wartość | Opis |
|---|---|---|
| `max_steps` | `500` | maksymalna długość epizodu (po osiągnięciu – `truncated=True`) |
| `random_layout` | `False` | czy układ mapy jest losowany przy `reset()` |

**Uwaga**: `random_layout=False` na starcie oznacza ustalony deterministyczny układ mapy (patrz 17.5) – robi się zdecydowanie łatwiejsze debugowanie. Losowe układy wprowadzać dopiero po uzyskaniu działającej wersji bazowej, jako osobny eksperyment.

### 17.4. Parametry czujników

| Stała | Wartość | Opis |
|---|---|---|
| `n_sensors` | `8` | liczba czujników odległości |
| `sensor_range` | `3.0` | maksymalny zasięg czujnika (`d_max`) |
| `sensor_step` | `0.1` | krok próbkowania wzdłuż promienia |

### 17.5. Ustalony układ mapy (dla `random_layout=False`)

Współrzędne w jednostkach mapy (`0..10`):

- **Robot start**: `(1.0, 1.0)`, prędkość `(0, 0)`, `carrying=False`.
- **Baza**: środek `(9.0, 9.0)`, promień `r_delivery`.
- **Paczki** (3 sztuki):
  1. `(2.0, 8.0)` – lewy górny róg
  2. `(8.0, 2.0)` – prawy dolny róg
  3. `(5.0, 1.5)` – środek dolnej części mapy
- **Przeszkody** (prostokąty AABB `(x_min, y_min, x_max, y_max)`):
  1. `(3.0, 3.0, 4.0, 7.0)` – pionowy regał po lewej
  2. `(6.0, 3.0, 7.0, 7.0)` – pionowy regał po prawej

Układ tworzy strukturę dwóch pionowych regałów z szerokim korytarzem środkowym (szerokość 2 jednostki między `x=4` a `x=6`) oraz otwartymi przejściami górą (`y > 7`) i dołem (`y < 3`). Żadna paczka ani baza nie jest zamknięta przez przeszkody – wszystkie cele są osiągalne wieloma drogami. Robot jest zmuszony omijać regały jadąc z dolnej strefy (paczki 2, 3) do górnej (paczka 1, baza), co daje sensowne wyzwanie bez pułapek topologicznych.

### 17.6. Parametry nagrody

| Składnik | Wartość | Opis |
|---|---|---|
| `r_pickup_bonus` | `+20.0` | podniesienie paczki |
| `r_delivery_bonus` | `+100.0` | dostarczenie paczki do bazy |
| `r_completion_bonus` | `+200.0` | dostarczenie **wszystkich** paczek |
| `r_step` | `-0.1` | kara za każdy krok (zachęca do pośpiechu) |
| `r_collision` | `-5.0` | kara za kolizję (ze ścianą lub przeszkodą) |
| `beta_progress` | `1.0` | współczynnik reward shapingu `β * (d_prev - d_curr)` |

### 17.7. Parametry treningu PPO

| Hiperparametr | Wartość | Opis |
|---|---|---|
| `total_timesteps` | `1_000_000` | łączny budżet kroków treningu |
| `n_envs` | `4` | liczba równoległych środowisk (VecEnv) |
| `learning_rate` | `3e-4` | domyślna wartość dla PPO |
| `n_steps` | `2048` | długość rolloutu na środowisko |
| `batch_size` | `64` | rozmiar mini-batcha |
| `gamma` | `0.99` | współczynnik dyskontowania |
| `gae_lambda` | `0.95` | parametr GAE |

---

## 18. Specyfikacja API środowiska

### 18.1. Przestrzeń akcji

```python
action_space = spaces.Box(
    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
)
```

Interpretacja: `action = [ax, ay]`, gdzie każda składowa to przyspieszenie w danej osi w jednostkach `a_max`. Efektywne przyspieszenie to `a = action * a_max`.

### 18.2. Przestrzeń obserwacji

Wektor 17-wymiarowy, wszystkie składowe znormalizowane do `[-1, 1]`:

```python
observation_space = spaces.Box(
    low=-1.0, high=1.0, shape=(17,), dtype=np.float32
)
```

**Indeksy i znaczenie składowych**:

| Indeks | Składowa | Normalizacja |
|---|---|---|
| `0` | `x` – pozycja X robota | `/ map_size * 2 - 1` |
| `1` | `y` – pozycja Y robota | `/ map_size * 2 - 1` |
| `2` | `vx` – prędkość X | `/ v_max` |
| `3` | `vy` – prędkość Y | `/ v_max` |
| `4` | `carrying` – czy niesie paczkę | `0.0` lub `1.0` |
| `5` | `dx_target` – wektor do celu (X) | `/ map_size` |
| `6` | `dy_target` – wektor do celu (Y) | `/ map_size` |
| `7` | `dist_target` – odległość do celu | `/ (map_size * sqrt(2))` |
| `8..15` | 8 czujników odległości | `/ sensor_range` (już w `[0, 1]`) |
| `16` | `n_packages_left` – liczba pozostałych paczek | `/ n_packages_initial` |

**Wybór „aktualnego celu"**:
- jeśli `carrying=False`: cel = **najbliższa niezebrana paczka** (w sensie euklidesowym),
- jeśli `carrying=True`: cel = **baza**.

W przypadku remisu w odległości do paczek wybierana jest paczka o najniższym indeksie w liście. Cel jest przeliczany w każdym kroku.

### 18.3. Sygnatury metod

```python
class WarehouseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None): ...

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[np.ndarray, dict]: ...

    def step(
        self,
        action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]: ...

    def render(self) -> np.ndarray | None: ...

    def close(self) -> None: ...
```

### 18.4. Zawartość słownika `info`

`info` zwracane z `reset()` i `step()` zawiera:

```python
{
    "collision": bool,              # czy w tym kroku wystąpiła kolizja
    "pickup": bool,                 # czy w tym kroku podniesiono paczkę
    "delivery": bool,               # czy w tym kroku dostarczono paczkę
    "n_packages_left": int,         # liczba paczek do dostarczenia
    "carrying": bool,               # stan nośności
    "n_collisions_total": int,      # licznik kolizji w epizodzie
    "n_deliveries_total": int,      # licznik dostarczeń w epizodzie
}
```

### 18.5. Warunki `terminated` i `truncated`

- `terminated = True` gdy: wszystkie paczki dostarczone (sukces),
- `truncated = True` gdy: osiągnięto `max_steps`,
- w przeciwnym razie oba są `False`.

**Kolizja NIE kończy epizodu** (ani jako `terminated`, ani jako `truncated`).

### 18.6. Pełny pseudokod metody `step()`

```python
def step(self, action):
    # 1. Walidacja i skalowanie akcji
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    a = action * self.a_max  # przyspieszenie w jednostkach fizycznych

    # 2. ZAFIKSOWANIE celu na początku kroku
    #    (ważne: cel może się zmienić w trakcie kroku przez pickup/delivery
    #     lub przez zmianę najbliższej paczki - dlatego fiksujemy na start)
    target_idx_start = self._get_current_target_index()   # np. indeks paczki lub -1 dla bazy
    target_pos_start = self._get_target_position(target_idx_start)
    d_prev = np.linalg.norm(self.pos - target_pos_start)

    # 3. Obliczenie pozycji kandydata (prędkość ZAMIERZONA)
    v_intended = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
    pos_new = self.pos + v_intended * self.dt
    speed_intended = np.linalg.norm(v_intended)

    # 4. Detekcja kolizji (przeszkody + granice mapy)
    collision = self._check_collision(pos_new)

    # 5. Aplikacja ruchu lub zatrzymanie
    if collision:
        # pozycja się nie zmienia, prędkość zeruje
        self.v = np.zeros(2, dtype=np.float32)
        self.n_collisions_total += 1
    else:
        self.pos = pos_new
        self.v = v_intended

    # 6. Próba podnoszenia paczki
    #    UWAGA: próg prędkości sprawdzany na `speed_intended` (zamierzonej),
    #    nie na `self.v` po kolizji. Bez tego kolizja zerująca prędkość
    #    pozwalałaby "darmowe" podniesienie paczki obok ściany.
    #    Dodatkowo: pickup/delivery tylko gdy NIE było kolizji w tym kroku.
    pickup = False
    if not collision and not self.carrying and speed_intended < self.v_pickup_max:
        for i, pkg in enumerate(self.packages):
            if pkg.active and self._distance(self.pos, pkg.pos) < self.r_pickup:
                pkg.active = False
                self.carrying = True
                pickup = True
                break  # tylko jedna paczka naraz

    # 7. Próba dostarczenia do bazy (analogicznie)
    delivery = False
    if not collision and self.carrying and speed_intended < self.v_pickup_max:
        if self._distance(self.pos, self.base_pos) < self.r_delivery:
            self.carrying = False
            self.n_delivered += 1
            delivery = True

    # 8. Obliczenie nagrody składnik po składniku
    reward = self.r_step  # -0.1
    if collision:
        reward += self.r_collision  # -5
    if pickup:
        reward += self.r_pickup_bonus  # +20
    if delivery:
        reward += self.r_delivery_bonus  # +100

    # Reward shaping liczony względem ZAFIKSOWANEGO celu z początku kroku.
    # Pomijamy krok w którym wystąpił pickup/delivery, bo cel się właśnie zmienił
    # i sygnał postępu byłby bezsensowny.
    if not (pickup or delivery):
        d_curr = np.linalg.norm(self.pos - target_pos_start)
        reward += self.beta_progress * (d_prev - d_curr)

    # 9. Sprawdzenie warunków końca epizodu
    self.step_count += 1
    all_delivered = (self.n_delivered == self.n_packages_initial)
    if all_delivered:
        reward += self.r_completion_bonus  # +200

    terminated = all_delivered
    truncated = (self.step_count >= self.max_steps)

    # 10. Budowa obserwacji i info
    obs = self._build_observation()
    info = self._build_info(collision, pickup, delivery)

    return obs, reward, terminated, truncated, info
```

### 18.7. Pseudokod metody `reset()`

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)  # inicjalizacja self.np_random

    # Ustawienie stanu początkowego z konfiguracji (sekcja 17.5)
    self.pos = np.array([1.0, 1.0], dtype=np.float32)
    self.v = np.zeros(2, dtype=np.float32)
    self.carrying = False
    self.step_count = 0
    self.n_delivered = 0
    self.n_collisions_total = 0

    # Reset paczek (wszystkie active)
    for pkg in self.packages:
        pkg.active = True

    obs = self._build_observation()
    info = self._build_info(collision=False, pickup=False, delivery=False)
    return obs, info
```

### 18.8. Kontrakt `render_mode`

- `render_mode=None` (domyślnie w treningu): `render()` zwraca `None`, bez wyświetlania.
- `render_mode="human"`: otwiera okno pygame i rysuje stan środowiska, `render()` zwraca `None`.
- `render_mode="rgb_array"`: zwraca `np.ndarray` o kształcie `(H, W, 3)` z aktualnym frame'em (do zapisu wideo/gif).

---

## 19. Ostateczna koncepcja
Projekt będzie polegał na stworzeniu własnego środowiska **WarehouseRobotEnv**, w którym robot sterowany ciągłymi akcjami porusza się po magazynie, zbiera paczki i dostarcza je do bazy. Problem zostanie rozwiązany z użyciem algorytmu **PPO**, a końcowa wersja zostanie rozszerzona o prosty tryb graficzny i eksperymenty opisane w sprawozdaniu.

Jest to projekt realistyczny do wykonania, zgodny z wymaganiami i jednocześnie wystarczająco oryginalny.
