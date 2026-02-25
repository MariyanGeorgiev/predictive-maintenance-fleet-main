# Developer Onboarding Guide (BG)

> Това ръководство е за начинаещ developer. Целта е да разбереш **какво прави проектът**, **как е организиран кодът**,
> **какви са основните променливи/параметри** и **как безопасно да правиш промени**.

---

## 1) Какво решава този проект

Проектът генерира **синтетични данни** за predictive maintenance на дизелови камиони.

В реалния свят често липсват:
- достатъчно исторически fault данни,
- етикетирани примери за ранни повреди,
- стабилни данни за обучение на ML модели.

Затова тук симулираме целия процес:
1. създаваме профили на камиони,
2. симулираме поведение на двигателя по време,
3. добавяме fault деградация,
4. извличаме feature-и,
5. генерираме labels,
6. записваме dataset в parquet,
7. валидираме качеството.

**Крайният резултат:** dataset + labels + валидирана логика, върху които можеш да обучаваш модели.

---

## 2) Голямата картина (pipeline)

Логически поток:

`config + fleet + simulation + faults -> features + labels -> storage -> validation -> web demo`

Runtime (какво се случва при стартиране):
1. `src/generator/cli.py` приема параметри от командния ред.
2. `src/fleet/fleet_factory.py` прави флот от камиони.
3. `src/faults/fault_schedule.py` назначава fault сценарии.
4. `src/generator/truck_day_generator.py` симулира деня по прозорци.
5. `src/features/*` извличат числови признаци.
6. `src/labels/ground_truth.py` прави целевите labels.
7. `src/storage/parquet_writer.py` записва файл/партиция.
8. `src/validation/*` проверява логическа и физична консистентност.

---

## 3) Структура на репото (за начинаещ)

## Root
- `requirements.txt` — Python библиотеки, които трябва да инсталираш.
- `run_front_and_backend.py` — стартира локалния full-stack demo server.
- `docs/` — документация.
- `src/` — реалният код на системата.
- `tests/` — автоматични тестове.
- `web/frontend/` — HTML/CSS/JS интерфейс.

## `src/config/`
**Защо съществува:** всички „правила“ и диапазони да са на едно място.
- `constants.py` — константи (например режими, диапазони, размери).
- `schema.py` — как изглеждат колоните/данните.

## `src/fleet/`
**Защо съществува:** описва „какво е един камион“.
- профил на двигател,
- геометрия (лагери),
- фабрика за генериране на много камиони.

## `src/simulation/`
**Защо съществува:** симулира как работи камионът във времето.
- operating state,
- Markov преходи,
- ambient влияние.

## `src/faults/`
**Защо съществува:** моделира повредите и деградацията им.
- общ интерфейс на fault,
- деградационен модел,
- 8 конкретни fault режима.

## `src/features/`
**Защо съществува:** преобразува симулацията в ML вход.
- conditioning features,
- vibration features,
- thermal features,
- финален feature vector.

## `src/labels/`
**Защо съществува:** генерира „истината“ (ground truth) за ML.

## `src/storage/`
**Защо съществува:** запис/четене на стабилен dataset формат.

## `src/validation/`
**Защо съществува:** проверява дали резултатът е валиден.

## `src/web/` + `web/frontend/`
**Защо съществува:** демо интерфейс за визуализиране на резултата.

---

## 4) Обяснение на важни параметри и имена (защо са такива)

Тази секция е нарочно подробна, за да можеш да „четеш“ кода по-лесно.

### 4.1 Параметри в CLI
Типично в `src/generator/cli.py` ще видиш:
- `trucks` — брой камиони за генериране.
  - Името е множествено число, защото е count за fleet.
- `days` — брой дни за симулация.
- `seed` — random seed за повторяемост.
  - Същият `seed` => същите резултати.
- `output_dir` — къде се записват изходните файлове.
- `workers` — брой паралелни процеси/работници.
- `single_truck` / `single_day` — за ограничен debug run.
- `validation_checkpoint` — специален режим за контролирани сценарии.
- `skip_existing` — ако файлът вече съществува, прескача го.

### 4.2 Имена свързани с време
- `day_index` — индекс на ден (0, 1, 2, ...), а не календарна дата.
- `window` / `window_index` — малък времев интервал вътре в деня.
- `WINDOWS_PER_DAY` — колко такива интервала има на ден.

### 4.3 Имена за operating state
- `operating_mode` — категория на режим (`idle`, `city`, `cruise`, `heavy`).
- `rpm` / `rpm_est` — реална/оценена стойност на обороти.
- `load` / `load_proxy` — натоварване и негов proxy feature.

### 4.4 Имена за fault логика
- `fault_id` — уникален идентификатор на повреда (например FM-01).
- `severity` — непрекъсната стойност (колко напреднала е повредата).
- `stage` — дискретен етап (примерно Stage 1/2/3).
- `onset_hours` — кога е започнала повредата (в часове от референтен момент).
- `total_life_hours` — очакван живот на компонента до край.
- `time_since_onset` — колко време е минало от началото.

### 4.5 Имена за labels
- `rul` / `RUL` — Remaining Useful Life (оставащ ресурс).
- `label_*` — target полета за моделиране.
- „ground truth“ — етикети от вътрешния симулационен state, не от feature-ите.

### 4.6 Имена за storage/schema
- `feature_count` — брой feature колони.
- `output_columns` — feature + label + meta колони в крайния output.
- `schema_definition` — дефинира типовете/имената, които всички модули спазват.

---

## 5) Откъде идват стойностите (данни и променливи)

Когато видиш променлива в кода, обикновено идва от един от тези източници:

1. **Константи от `src/config/constants.py`**
   - домейн знания, диапазони, фиксирани параметри.
2. **CLI аргументи**
   - runtime настройки от потребителя.
3. **Генерирани fleet профили**
   - параметри специфични за конкретен truck.
4. **Изчисления от симулацията**
   - operating mode, temperatures, vibration и т.н.
5. **Fault модели**
   - severity/stage промени по време.
6. **Derived features**
   - статистики и трансформации върху симулационните сигнали.

Практичен съвет:
- Ако не знаеш откъде идва дадена стойност, търси името й с `rg "име_на_променлива" src tests`.

---

## 6) Как да стартираш проекта (стъпка по стъпка)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Пусни всички тестове
```bash
pytest tests/ -v
```

## Бърз smoke run
```bash
python -m src.generator.cli --single-truck 1 --single-day 0 --output-dir output/test/
```

## Validation checkpoint
```bash
python -m src.generator.cli --validation-checkpoint --output-dir output/validation/
python -m src.validation.range_checks output/validation/
python -m src.validation.progression_checks output/validation/
python -m src.validation.cross_feature output/validation/
```

## Стартирай GUI + API
```bash
python run_front_and_backend.py
```
Отвори: `http://127.0.0.1:8787/`

---

## 7) Как да правиш промени безопасно

1. Промени само една тема наведнъж (пример: само fault логика).
2. Пусни тестове след промяната.
3. Провери дали schema/колони не са счупени.
4. Ако има ново поведение, документирай го.
5. Не променяй `seed` логиката без причина.

---

## 8) Чести грешки при начинаещи

- Смесване на labels и features (data leakage).
- Промяна на имена на колони без update в schema/tests.
- Добавяне на randomness без seed.
- Нереалистични диапазони (чупят validation checks).
- Извеждане на твърде малко логове при дебъг.

---

## 9) Практичен roadmap (ако искаш да ъпгрейдваш)

### Краткосрочно
- Подобри frontend визуализация (графики за trend на severity/RUL).
- Добави endpoint за preview на един truck/day.

### Средносрочно
- Добави нов fault mode + unit/integration тестове.
- Добави нови диагностични feature-и.

### Дългосрочно
- Обучи модели за Path A/Path B върху синтетичния dataset.
- Добави MLOps pipeline.

---

## 10) Definition of Done (чеклист)

Промяната е завършена, ако:
- [ ] Тестовете минават.
- [ ] Няма schema regressions (или има документиран migration).
- [ ] Validation checks са успешни.
- [ ] Документацията е обновена.
- [ ] Промяната е обяснима за следващ developer.

---

## 11) Едно изречение за проекта

> Това е модулна платформа, която създава детерминирани и физически правдоподобни synthetic данни + labels за predictive maintenance на fleet.
