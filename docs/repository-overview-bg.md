# Пълно и подробно обяснение на репозитория (на български)

> Този документ е предназначен за хора, които искат да разберат **как работи цялата система**, как са свързани модулите, какви данни се генерират, как се валидират и как да стартират проекта от нула.

---

## 1) Какво представлява проектът

Това репо реализира **Synthetic Data Generator** за predictive maintenance на флот от търговски дизелови камиони.

### Основна цел
Да генерира реалистични (но синтетични) времеви данни за:
- вибрации,
- температури,
- operating state (режим на работа, RPM, натоварване),
- fault progression (степен/тежест на повреда),
- labels за ML задачи.

### Защо е нужно
В реални индустриални системи често липсват:
- достатъчно исторически данни,
- достатъчно редки fault случаи,
- добре етикетирани интервали за remaining useful life (RUL).

Този проект решава това с **контролирана симулация**, която може да произвежда голям обем консистентни данни за обучение и тестване на модели.

### Какво се очаква от системата
- детерминирана генерация (еднакъв seed ⇒ еднакви резултати),
- физически правдоподобни сигнали,
- ясна връзка между деградация и наблюдаеми features,
- стабилна схема на колоните за downstream ML pipeline.

---

## 2) Голямата картина: архитектура

Системата е описана като dual-path ML концепция:
- **Path A (Edge):** бърза класификация на дефекти в реално време.
- **Path B (Cloud):** прогноза за RUL (remaining useful life).

Това репо е фокусирано върху **Phase 1**: надеждна генерация на синтетични данни и labels, които после се използват за обучение на Path A/Path B.

Практически, кодът минава през следната верига:
1. Създаване на профил на камион/двигател.
2. Симулация на operating state за всеки 60s прозорец.
3. Прилагане на fault модели и деградация във времето.
4. Извличане на vibration/thermal features.
5. Сглобяване на финален feature vector.
6. Генериране на labels от вътрешното състояние на повредите.
7. Запис в Parquet + съхранение на термално състояние между дни.
8. Валидиране чрез range/progression/cross-feature checks.

---

## 3) Структура на проекта (подробно)

## `src/config/`
Конфигурационният слой е „договорът“ на системата.

- `constants.py`
  - съдържа домейн константи: operating modes (`idle`, `city`, `cruise`, `heavy`),
  - Markov transition matrix за режимите,
  - RPM и load диапазони,
  - дефиниции на сензори,
  - честотни ленти за вибрационните признаци,
  - window параметри,
  - baseline диапазони за температурния модел,
  - деградационни/label прагове.

- `schema.py`
  - описва структурите на данни и валидира вход/изход,
  - пази консистентност на типове и shape между модулите.

Практически: ако искаш да промениш физични диапазони или feature contract, най-често започваш от `config/`.

---

## `src/fleet/`
Отговаря за това **как изглежда един камион** като параметри.

- `engine_profile.py`
  - генерира параметри за конкретен engine profile,
  - включва variation между „modern“ и „older“.

- `bearing_geometry.py`
  - описва геометрични параметри на лагери,
  - изчислява характерни честоти (важни за fault signatures).

- `truck.py`
  - капсулира конкретен truck instance.

- `fleet_factory.py`
  - създава множество truck обекти с контролирана стохастика,
  - удобен entry point за batch генерация.

---

## `src/simulation/`
Симулира текущото operating състояние на двигателя.

- `markov_chain.py`
  - преминаване между operating режими по преходна матрица.

- `operating_state.py`
  - за всеки прозорец генерира режим + RPM + load.

- `ambient.py`
  - външни условия, които влияят на термалната картина.

Този слой е основата, върху която fault моделите „стъпват“.

---

## `src/faults/`
Сърцето на domain логиката за повреди.

- `fault_mode.py`
  - общ интерфейс/абстракция за fault режим.

- `degradation_model.py`
  - еволюция на severity във времето,
  - реализира плавен и ограничен растеж (без нефизични експлозии).

- `fault_schedule.py`
  - кога е активна дадена повреда,
  - позволява сценарии с multiple simultaneous faults.

- Конкретни fault режими:
  - `fm01_bearing.py` — лагер,
  - `fm02_cooling.py` — охлаждане,
  - `fm03_valve_train.py` — клапанен механизъм,
  - `fm04_oil.py` — масло,
  - `fm05_turbo.py` — турбо,
  - `fm06_injector.py` — инжектор,
  - `fm07_egr.py` — EGR cooler (вкл. leak event логика),
  - `fm08_dpf.py` — DPF.

### Важен принцип
Ефектите от fault режимите се комбинират контролирано (напр. капиране/максимум за някои измерения), за да не се получат нефизични стойности.

---

## `src/features/`
Превръща симулираните сигнали в ML-подходящи числови признаци.

- `conditioning.py`
  - базови operating feature-и (напр. `rpm_est`, `load_proxy`).

- `vibration_features.py`
  - time/frequency признаци (RMS, crest, kurtosis, band energies, spectral indicators),
  - използва sensor/axis/band логика.

- `thermal_features.py`
  - статистики за температури (mean/std/min/max и др.),
  - изчислява полезни диференциали между сензори.

- `feature_vector.py`
  - финално сглобяване на feature набора,
  - пази договор за фиксиран брой и имена на колони.

---

## `src/labels/`
- `ground_truth.py`
  - генерира labels **от вътрешното състояние на fault моделите**,
  - това е критично: labels не се извличат обратно от feature-ите,
  - включва fault mode/stage/severity/RUL и цели за Path A/Path B.

---

## `src/storage/`
Слой за устойчив запис на резултатите.

- `schema_definition.py`
  - дефинира колонна схема и типове за output dataset.

- `parquet_writer.py`
  - запис на truck/day parquet партиции,
  - гарантира съвместимост за downstream обработка.

- `thermal_state.py`
  - пази крайното термално състояние след деня,
  - зарежда го като initial state за следващия ден.

---

## `src/generator/`
Оркестрация на целия pipeline.

- `truck_day_generator.py`
  - най-важният orchestrator за един truck/day.

- `batch_generator.py`
  - мащабира към много камиони/много дни,
  - поддържа паралелизация и skip/resume поведения.

- `cli.py`
  - удобни команди за smoke, validation checkpoint и full run.

---

## `src/validation/`
Контрол на качество след генерацията.

- `range_checks.py`
  - стойностите да са в допустими физични/spec граници.

- `progression_checks.py`
  - severity/stage/RUL да е логически последователна във времето.

- `cross_feature.py`
  - валидира зависимости между свързани feature-и.

---

## 4) Тестове (`tests/`) — какво покриват

Тестовете са организирани по теми:
- геометрия и честоти (`test_bearing_geometry.py`),
- Маркови преходи (`test_markov_chain.py`),
- комбинирани повреди (`test_multi_fault.py`),
- labels (`test_labels.py`),
- feature vector contract (`test_feature_vector.py`),
- термален модел (`test_thermal_model.py`),
- parquet I/O (`test_parquet_output.py`),
- vibration extraction (`test_vibration_features.py`),
- деградация (`test_degradation.py`),
- интеграционни сценарии (`test_integration.py`).

### Какво гарантират интеграционните тестове
- end-to-end генерацията да работи,
- healthy и faulty сценарии,
- reproducibility при еднакъв seed,
- валиден запис/четене от Parquet.

---

## 5) Данни и колони: какво излиза като резултат

Проектът генерира данни в прозорци от 60 секунди.

Всеки ред (window) съдържа:
- метаданни (напр. truck/day/window),
- conditioning features,
- vibration features,
- thermal features,
- labels за fault/classification/RUL.

Схемата е фиксирана и е ключово да се пази стабилна, защото върху нея се тренират и сравняват модели в следващите фази.

---

## 6) Детерминизъм и repeatability

Един от най-важните инженерни аспекти в този проект е repeatability:
- seed-овете са организирани така, че генерацията да е възпроизводима,
- избягва се скрито mutable RNG състояние между процеси,
- това е критично за честен ML benchmarking.

Практически ефект: ако пуснеш същия сценарий със същите параметри, трябва да получиш същите числови резултати.

---

## 7) Физична правдоподобност

Сигналите не са произволен noise:
- operating режимите ограничават RPM/load,
- температурите следват first-order lag логика с time constants,
- стойностите се clamp-ват в реалистични граници,
- fault ефектите са съобразени с физическия смисъл на съответната повреда.

Това прави синтетичните данни значително по-полезни за реални модели.

---

## 8) Как се стартира проектът (практически)

### 8.1 Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 8.2 Пускане на тестовете
```bash
pytest tests/ -v
```

### 8.3 Smoke run (1 truck × 1 day)
```bash
python -m src.generator.cli --single-truck 1 --single-day 0 --output-dir output/test/
```

### 8.4 Validation checkpoint
```bash
python -m src.generator.cli --validation-checkpoint --output-dir output/validation/
python -m src.validation.range_checks output/validation/
python -m src.validation.progression_checks output/validation/
python -m src.validation.cross_feature output/validation/
```

### 8.5 Full generation (примерен command)
```bash
python -m src.generator.cli --trucks 200 --days 183 --seed 42 --output-dir output/full/ --workers 8 --skip-existing
```

---

### 8.6 Стартиране на красивия frontend + backend с автоматичен redirect
```bash
python run_front_and_backend.py
```
По подразбиране отваря браузър и пренасочва към `http://127.0.0.1:8787/` (което redirect-ва към `/app/`).


## 9) Как да четеш кода бързо (препоръчан ред)

Ако си нов в проекта, добър ред е:
1. `src/generator/cli.py` — виж какви са entry point-ите.
2. `src/generator/truck_day_generator.py` — разбери основната оркестрация.
3. `src/faults/` + `src/features/` — виж как се моделира „причина → наблюдаем ефект“.
4. `src/labels/ground_truth.py` — как се създават target-ите.
5. `src/storage/parquet_writer.py` — какво точно се записва.
6. `tests/test_integration.py` — как изглежда правилно end-to-end поведение.

---

## 10) Чести грешки при промени

- Промяна в feature имена/брой без ъпдейт на schema/tests.
- Добавяне на случайност без контрол на seed.
- Използване на feature-ите за labels (data leakage).
- Нереалистични диапазони, които чупят validator-ите.
- Неправилно комбиниране при multi-fault сценарии.

---

## 11) Какво следва след тази фаза

След надеждна генерация на данните:
- обучение на Path A и Path B,
- tuning и сравнение на модели,
- оценка върху sacred test split,
- експортиране на production артефакти.

С други думи: това репо изгражда „данните и истината“; следващите фази изграждат „моделите и внедряването“.

---

## 12) Кратко финално резюме

Това е добре структурирана synthetic-data платформа за predictive maintenance с:
- модулна архитектура,
- контролирана симулация на operating режими,
- осем fault режима с деградация,
- богато feature инженерство,
- коректни ground-truth labels,
- стабилен формат за запис,
- сериозен validation/testing слой.

Ако искаш да го обясниш с едно изречение:
> „Това репо генерира висококачествени, детерминирани и физически правдоподобни данни за обучение и оценка на модели за предиктивна поддръжка на камиони.“
