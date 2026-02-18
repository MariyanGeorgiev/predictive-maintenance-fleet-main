# 10.10 Maintenance Lifecycle Model

> **Purpose:** Models realistic fleet maintenance operations to produce synthetic data with
> operationally accurate class distributions. Without maintenance modeling, monotonic fault
> progression produces ~62% NORMAL / 36% CRITICAL — far from real-world fleet data where
> ~95% of operational windows are healthy. This section defines the detection, inspection,
> repair, and return-to-service mechanics that close this gap.

## 10.10.1 Motivation

In a well-maintained commercial fleet, fault degradation does not proceed uninterrupted from
onset to catastrophic failure. Fleet operators use a combination of onboard monitoring,
scheduled inspections, and driver-reported symptoms to detect emerging faults. Once detected,
a maintenance decision is made: repair immediately, schedule for the next available window,
or monitor and reassess. The result is that most fault episodes are resolved during early
degradation (Stage 2), before the fault signature becomes operationally significant.

The target class distribution for Path A labels reflects this reality:

| Label    | Target % | Description |
|----------|----------|-------------|
| NORMAL   | ~95%     | Healthy operation + Stage 1 + Stage 2 (early degradation, repaired before IMMINENT) |
| IMMINENT | ~4%      | Early Stage 3 — fault clearly visible, intervention urgently needed |
| CRITICAL | ~1%      | Late Stage 3 + Stage 4 — failure imminent or in progress |

The 5% combined IMMINENT + CRITICAL fraction represents faults that evade early detection —
precisely the cases the ML system is designed to catch.

## 10.10.2 Detection Model

Fault detection probability depends on degradation stage. Detection represents any mechanism
that flags the truck for inspection: automated monitoring alerts, driver-reported symptoms,
or routine scheduled maintenance checks.

| Stage | Daily Detection Probability | Mean Days to Detect | Rationale |
|-------|----------------------------|---------------------|-----------|
| Stage 1 (0–60% life) | 0 | N/A | Below sensor noise floor; invisible |
| Stage 2 (60–75% life) | 0.20–0.30 | 3–5 days | Subtle anomaly; intermittent alerts |
| Stage 3 (75–95% life) | 0.60–0.80 | 1–2 days | Clear anomaly; persistent alerts |
| Stage 4 (95–100% life) | 0.95 | <1 day | Critical alarm; driver-reported |

**Implementation:** At each simulated day boundary, for each active fault in Stage 2+,
sample a Bernoulli trial with the stage-appropriate detection probability. On the first
success, the fault is flagged as "detected" and an inspection is scheduled.

Detection probabilities are sampled once per fault at onset — one value per stage:

- `p_detect_stage2 = rng.uniform(0.20, 0.30)`
- `p_detect_stage3 = rng.uniform(0.60, 0.80)`
- `p_detect_stage4 = 0.95` (fixed, not sampled)

These three values remain constant for that fault episode. The Bernoulli trial on each day
uses whichever probability corresponds to the fault's *current* stage. This captures
truck-to-truck variability in monitoring system sensitivity and driver attentiveness while
preserving the stage-differentiated detection rates.

## 10.10.3 Inspection Scheduling

Once a fault is detected, an inspection is scheduled based on the urgency implied by the
detection stage:

| Detection Stage | Inspection Delay | Rationale |
|-----------------|-----------------|-----------|
| Stage 2 | 7–21 days | Low urgency; schedule at next convenient maintenance window |
| Stage 3 | 1–3 days | Urgent; pull truck from service soon |
| Stage 4 | 0–1 day | Emergency; immediate shop visit |

**Implementation:** Inspection day = detection day + `rng.integers(low, high+1)` using the
ranges above. The truck continues operating (generating data) until the inspection day.
If the fault progresses to a higher stage before the scheduled inspection, the inspection
is not accelerated — it proceeds on the originally scheduled day. (In practice, a new
detection at Stage 3 would trigger re-evaluation, but for generation simplicity we use
the conservative single-schedule model.)

**Note:** A fault detected at Stage 2 with a 14-day inspection delay may progress to Stage 3
or even Stage 4 before the inspection occurs. This is realistic and produces the IMMINENT /
CRITICAL windows in the dataset — these are the cases where early detection did not lead to
timely intervention.

## 10.10.4 Inspection Outcomes

On the inspection day, one of three outcomes is determined. Outcome probabilities depend
on the fault's current stage at inspection time:

| Outcome | Stage 2 | Stage 3 | Stage 4 | Description |
|---------|---------|---------|---------|-------------|
| **Repair** | 85% | 90% | 100% | Truck goes to shop, fault is resolved, returns to service |
| **Monitor** | 10% | 8% | 0% | Maintenance decides the fault is not severe enough to pull the truck |
| **False Positive** | 5% | 2% | 0% | Inspection finds no actionable fault |

**Stage 4 always repairs:** A Stage 4 fault is unambiguously critical — no competent
maintenance team would monitor or dismiss it. The 100% repair rate at Stage 4 is a
hard constraint, not a tunable parameter.

### 10.10.4.1 Repair Outcome

The truck is taken offline for repair. Repair duration depends on the fault's current stage
at the time of the inspection (not the detection stage):

| Current Stage at Inspection | Repair Duration | Rationale |
|-----------------------------|----------------|-----------|
| Stage 2 | 1–2 days | Early wear; quick fix (filter swap, bearing adjustment, fluid top-up) |
| Stage 3 | 2–5 days | Component replacement; more involved diagnostics and parts |
| Stage 4 | 5–10 days | Major repair or overhaul; possible parts ordering, collateral damage assessment |

**During repair:** No data is generated for this truck on repair days. The resulting gaps
in the Parquet timeline are realistic — trucks in the shop do not produce telemetry.
Repair duration is in calendar days: a 2-day repair starting on day 52 means days 52–53
have no data, and the truck returns to service on day 54. Return day = repair_start +
repair_duration.

**"Offline" means repair days only.** The truck continues operating and generating data
during the inspection delay period (between detection and the scheduled inspection).
Only the repair days are offline. The validation constraint "No truck offline > 14 days"
(§10.10.7) refers to repair duration only, not detection-to-return elapsed time.

**After repair:** The fault is fully resolved. The truck returns to service in a healthy
state (severity = 0, no active faults). A new fault episode *may* be assigned later
(see §10.10.5).

**Multi-fault trucks:** When a truck has multiple concurrent faults (e.g., FM-01 + FM-05)
and one fault is detected and repaired, the repair addresses **all active faults** — not
just the detected one. This reflects real-world shop practice: when a truck is pulled for
service, mechanics perform a comprehensive inspection and address all known issues. The
truck returns to service fully healthy (all faults cleared, severity = 0 for all).
This is a simplification — in practice some secondary faults might be missed — but it
avoids the complexity of partial-repair state tracking and is conservative (it produces
more NORMAL windows, aligning with the 95/4/1 target).

### 10.10.4.2 Monitor Outcome

The maintenance team decides the condition is not urgent enough to warrant pulling the truck.
The truck continues operating with the active fault. Two sub-outcomes are equally likely:

| Sub-outcome | Probability (of Monitor) | Behavior |
|-------------|-------------------------|----------|
| **Continue degrading** | 50% | Fault life resumes normally; no change to progression |
| **Improve** | 50% | Severity decays back toward 0 over 200–500 hours (see §7.2 note on improving conditions: DPF regen clearing a blockage, partial repair resetting progression) |

**Improvement model:** When a monitored fault improves, its severity decays exponentially:
`severity(t) = severity_at_decision * exp(-t_since_decision / tau_improve)`, where
`tau_improve = rng.uniform(200, 500)` hours. When severity drops below 0.01, the fault is
considered resolved and removed. The truck returns to healthy state and may receive a new
fault assignment.

**Continued degradation:** The fault progresses normally. It may trigger a *second* detection
event (with the same per-day probability). A second detection always leads to repair (no
further monitor decisions for the same fault episode).

### 10.10.4.3 False Positive Outcome

The inspection finds no actionable issue. The fault state is **not** actually reset — the
underlying degradation continues — but the detection flag is cleared. The fault must be
re-detected through the normal detection model. This simulates the real-world scenario where
an intermittent early-stage fault is dismissed on inspection but continues to develop.

**Implementation note:** A false-positive outcome resets the detection state for the
*triggering* fault to "undetected." The fault continues progressing and may be re-detected
later at a higher stage (with higher detection probability). A re-detection after a false
positive is treated as a *first* detection for that fault — the Monitor and False Positive
outcomes remain available (this is not a "second detection" per §10.10.4.2).

**Multi-fault trucks:** When a multi-fault truck is inspected due to one fault's detection,
the inspection covers the whole truck. If the outcome is False Positive, the detection flag
is cleared for the triggering fault only. Other active faults retain their detection state
(detected or undetected). If the outcome is Repair, all active faults are cleared per
§10.10.4.1.

## 10.10.5 Post-Repair Fault Assignment

After a truck returns from repair, it is eligible for a new fault episode. This models the
reality that fleet trucks accumulate different failure modes over their operational life.

**Constraints:**
- **Healthy buffer:** Minimum 30 days (720 hours) after return to service before a new fault
  can onset. This prevents unrealistic back-to-back fault episodes.
- **Different fault type:** The new fault must be a different FM-XX type than any fault
  that was active at the time of repair. A truck that had FM-01 + FM-05 repaired will not
  immediately develop another FM-01 or FM-05. (Different failure modes on the same truck
  are realistic — e.g., a bearing and turbo repair does not prevent injector degradation.)
- **Assignment probability:** Not every truck gets a new fault after repair. Use the same
  ~70% probability as initial assignment to determine whether a new fault is assigned.
- **New fault onset:** If assigned, onset time = return_to_service_hours + healthy_buffer +
  `rng.uniform(0, remaining_sim_hours * 0.5)`, where `remaining_sim_hours` is measured from
  `return_to_service_hours` to simulation end (i.e., `total_sim_hours - return_to_service_hours`).
- **Insufficient time guard:** If `remaining_sim_hours < healthy_buffer`, no new fault is
  assigned. The truck completes the simulation in healthy state. This avoids assigning faults
  whose onset would fall beyond the simulation window.

## 10.10.6 Expected Class Distribution

With the maintenance lifecycle model applied, the expected class distribution emerges from
the mechanics rather than being directly controlled:

1. **Stage 1 windows** (~60% of fault life): All labeled NORMAL. Most faults spend their
   longest phase here, invisible to detection.
2. **Stage 2 windows** (~15% of fault life, labeled NORMAL): Most faults are detected and
   repaired during this phase. With mean detection time of 3–5 days (§10.10.2) and
   inspection delay of 7–21 days (§10.10.3), most Stage 2 faults are resolved before
   reaching Stage 3.
3. **Stage 3 windows** (IMMINENT/CRITICAL): Only faults that passed through Stage 2 without
   repair reach here. With 85% repair rate at Stage 2 inspection, this is a small fraction.
4. **Stage 4 windows** (CRITICAL): Very rare — only faults where both Stage 2 and Stage 3
   interventions failed or were delayed.

**Monte Carlo estimate:** With the parameters above, the expected distribution is approximately
93–96% NORMAL, 3–5% IMMINENT, 0.5–2% CRITICAL. The exact values depend on the random seed
and are validated during generation (see §10.10.7). The target framing of "~95/4/1" in
§10.10.1 is an ideal; the validation bounds in §10.10.7 define the acceptable range.

## 10.10.7 Validation

After full fleet generation, verify:

1. **Class distribution:** NORMAL 93–96%, IMMINENT 3–5%, CRITICAL 0.5–2%. These bounds
   must sum to ~100% (±0.5% for rounding). If outside these bounds, investigate detection
   probabilities and inspection delays.
2. **Repair gap realism:** Mean repair duration 2–4 days. No truck offline > 14 days.
3. **Fault episode count:** Mean 1–3 fault episodes per faulted truck over the 183-day window.
4. **No simultaneous dual-repair:** A truck can only be in one repair at a time (enforced by
   the single active fault model per inspection cycle).
5. **Temporal continuity:** Parquet files are absent for repair days (realistic data gaps).
   Thermal state is reset to healthy baselines on return-to-service day.
6. **End-of-simulation faults:** Some faults may remain active (undetected or unrepaired)
   when the 183-day simulation ends. This is expected and realistic — not every fault
   resolves within the observation window. These are recorded in the maintenance log with
   `"outcome": "simulation_end"` and no repair event.
7. **Concurrent inspection collision:** If a truck has a scheduled inspection on day X but
   enters repair for a different fault on day X-2 (returning on day X+1), the scheduled
   inspection is cancelled — the repair already addressed all active faults (§10.10.4.1).

---

# Amendments to Existing Sections

The following changes bring existing spec sections into alignment with the maintenance
lifecycle model (§10.10). Each amendment is labeled with the section it modifies.

---

## Amendment A: §7.2 Path B — Degradation Tracking

**Current text (final paragraph):**
> Note on "improving": Some modes genuinely recover. DPF regen clears a blockage; a partial
> repair can reset bearing wear progression. Showing improvement gives maintenance staff
> confirmation that their intervention worked.

**Replace with:**

> **Trend direction — "improving":** The "improving" trend is not an edge case — it is a
> first-class operational state driven by the fleet maintenance lifecycle (§10.10). When a
> fault is detected and the maintenance decision is to monitor rather than repair (~10% of
> inspections per §10.10.4.2), approximately half of monitored faults improve naturally.
> Mechanisms include: DPF regeneration clearing a blockage, partial interventions resetting
> bearing wear progression, oil changes arresting thermal degradation, and turbo fouling
> clearing under sustained highway load.
>
> The Path B model must correctly identify improving trends and reflect them in the RUL
> estimate (RUL increasing over time) and Health Index (trending upward). Failure to model
> improvement will produce false alarms on trucks that are self-correcting or have been
> partially serviced.
>
> After a full repair (§10.10.4.1), the truck returns to service in a healthy state. The
> RUL resets to infinity and the Health Index returns to 100. This is a discontinuous jump,
> not a gradual improvement — the model should learn to distinguish repaired-healthy from
> never-faulted-healthy (they are functionally identical from the sensor perspective).

---

## Amendment B: §7.4 Health Index — RUL Mapping

**Append after existing content:**

> **Repair resets:** When a truck undergoes repair (§10.10.4.1), the Health Index resets
> discontinuously to 100 and RUL resets to infinity. The mapping function must handle this
> gracefully — the post-repair Health Index is identical to a never-faulted truck.
>
> **Improving faults:** When a monitored fault improves (§10.10.4.2), the Health Index
> increases gradually as severity decays. The RUL estimate increases correspondingly. The
> mapping function should be symmetric — the same severity value produces the same Health
> Index regardless of whether the fault is worsening or improving.

---

## Amendment C: §9.4 Feature Vector Summary

**Current text references 191 features.**

**Replace with:**

> The feature extraction pipeline produces **221 features** per 60-second window:
>
> | Category | Count | Details |
> |----------|-------|---------|
> | Conditioning | 2 | RPM estimate, load proxy |
> | Vibration | 180 | 3 sensors × 3 axes × (6 time-domain + per-band freq-domain stats) + 2 SK per sensor |
> | Thermal | 39 | 6 sensors × 6 statistics + 3 differential features |
> | **Total** | **221** | |
>
> The increase from the original 191-feature estimate to 221 reflects the inclusion of
> aggregation statistics (standard deviation and max values) for vibration time-domain and
> frequency-domain features. These additional statistics capture within-window variability
> that improves fault detection sensitivity, particularly for intermittent faults (FM-07
> EGR leak events) and early-stage bearing wear (FM-01 kurtosis max vs. mean).
>
> The feature count is enforced by contract assertion at module load time:
> `assert N_FEATURES == 221`

---

## Amendment D: §10.2.3 Fault Progression Model

**Append after existing content:**

> **Maintenance interruption:** The degradation model described above represents uninterrupted
> fault progression. In practice, the maintenance lifecycle (§10.10) can interrupt or reset
> this progression:
>
> - **Repair (§10.10.4.1):** Severity resets to 0. The degradation curve is terminated. A
>   new fault (different type) may be assigned later with its own independent degradation model.
> - **Monitor — improving (§10.10.4.2):** Severity decays exponentially from its current
>   value: `severity(t) = severity_at_decision × exp(-t / τ_improve)`, where τ_improve is
>   sampled from [200, 500] hours. This overrides the logistic growth curve for the duration
>   of improvement. If severity drops below 0.01, the fault is considered resolved.
> - **Monitor — continuing (§10.10.4.2):** The original degradation model resumes unchanged.
>
> **Implementation note:** The degradation model implementation uses a logistic growth curve
> `(exp(k·t_frac) - 1) / (exp(k) - 1)` with k=5.0, replacing the original Wiener process
> specification. The Wiener process produced noise-dominated trajectories where stochastic
> variation overwhelmed the drift term, making stage progression unreliable. The logistic
> curve with bounded mean-reverting noise preserves monotonic progression while maintaining
> realistic variability. This is flagged as an implementation deviation from the original
> spec formula.

---

## Amendment E: §10.3.1 Baseline Thermal Model

**Append after existing content:**

> **Thermal state after repair:** When a truck returns from repair (§10.10.4.1), the thermal
> state is reset to healthy baselines. Specifically:
>
> - All 6 sensor temperatures (T1–T6) are re-initialized to idle baseline values from the
>   truck's engine profile, as if the truck were starting fresh on day 0.
> - The end-of-day thermal state JSON for the last pre-repair day is discarded. The
>   return-to-service day uses fresh idle baselines.
> - Any accumulated fault-related thermal offsets are cleared (fault no longer active).
>
> This models the physical reality that a repaired engine returns to its nominal thermal
> operating envelope. The thermal time constant (τ) ensures the temperature transitions
> smoothly from the reset baselines when the truck resumes operation.

---

## Amendment F: §10.5 Synthetic Data Generation Pipeline

**Replace step 2 ("Initialize fault states") and add new step after step 7:**

> **Step 2 — Initialize fault states and maintenance schedule:**
> For each truck, assign initial fault modes per the fault distribution (§10.7):
> ~30% healthy, ~40% single fault, ~20% double, ~10% triple.
> For each assigned fault: sample onset time, degradation parameters, and total life.
> Additionally, initialize the maintenance lifecycle state (§10.10): detection probabilities,
> inspection scheduling parameters, and outcome probabilities per fault.
>
> **Step 7a — Maintenance lifecycle check (per day boundary):**
> At the end of each simulated day, before advancing to the next day:
>
> 1. For each active fault in Stage 2+, run detection Bernoulli trial (§10.10.2).
> 2. If newly detected, schedule inspection day (§10.10.3).
> 3. If today is a scheduled inspection day, determine outcome (§10.10.4):
>    - **Repair:** Mark truck as offline for repair duration. Skip generating Parquet files
>      for repair days. On return-to-service day, reset fault state and thermal baselines.
>      Optionally assign new fault (§10.10.5).
>    - **Monitor:** Set fault to improving or continuing per §10.10.4.2.
>    - **False positive:** Clear detection flag, continue generating normally.
> 4. If truck is in repair (offline), skip data generation for this day.
>
> This step runs between day N and day N+1, after the day's Parquet file has been written
> (or skipped if the truck is offline).

**Update step 6 ("Extract features"):**

> Extract features: Run feature extraction pipeline (§9) on the generated signals. Produces
> **221** features per 60-second window → 1440 feature vectors for 24 hours.

---

## Amendment G: §10.7 Ground Truth Labeling Strategy

**Append after existing content:**

> **Labels after repair:** When a truck returns from repair (§10.10.4.1):
> - `fault_mode` = "HEALTHY"
> - `fault_severity` = "HEALTHY"
> - `rul_hours` = ∞ (infinity, stored as a sentinel value, e.g., 99999.0)
> - `path_A_label` = "NORMAL"
>
> **Labels during repair:** No data is generated while the truck is in the shop. There are
> no label rows for repair days — the Parquet file for that truck-day simply does not exist.
> This creates realistic temporal gaps in the dataset that the ML model must handle.
>
> **Labels during improvement:** When a monitored fault is improving (§10.10.4.2), the labels
> reflect the *current* severity, which is decaying. The `fault_mode` remains set to the
> active fault ID (e.g., "FM-01") as long as severity > 0.01. The `rul_hours` is set to
> infinity (the fault is improving, not progressing toward failure). The `path_A_label`
> is determined by the current stage, which may regress from IMMINENT back to NORMAL as
> severity decreases.
>
> **False positive labels:** After a false positive inspection (§10.10.4.3), the labels
> continue to reflect the actual underlying fault state. The false positive is a
> maintenance-side event — the ground truth labels always reflect physical reality.

---

## Amendment H: §11.1–11.2 Storage Sizing

**Append note after existing content:**

> **Adjusted for maintenance gaps:** With the maintenance lifecycle model (§10.10), not all
> 200 × 183 = 36,600 truck-days produce data. Trucks in the shop for repair generate no
> Parquet files. Estimated data reduction: ~3–5% fewer files (roughly 1,000–1,800 missing
> truck-days across the fleet), depending on fault prevalence and repair durations.
> Updated estimate: ~34,800–35,600 Parquet files, ~61–63 GB total storage.

---

## Amendment I: §7.1/§7.2/§7.3 — MAINTENANCE as a Formal Operational State

**Add to §7.1 (Path A) and §7.2 (Path B):**

> **MAINTENANCE state:** In addition to the fault-driven states (NORMAL, IMMINENT, CRITICAL),
> the system recognizes MAINTENANCE as a formal operational state. A truck enters MAINTENANCE
> when it is taken offline for repair (§10.10.4.1).
>
> **State definition:**
>
> ```
> Operational States:
>   HEALTHY     — no active faults, normal operation
>   DEGRADING   — active fault in Stage 1–2, sensors may show early signs
>   IMMINENT    — active fault in early Stage 3, intervention needed
>   CRITICAL    — active fault in late Stage 3 / Stage 4, failure imminent
>   MAINTENANCE — truck offline for repair, no sensor data produced
> ```
>
> **System behavior during MAINTENANCE:**
>
> | Subsystem | Behavior | Rationale |
> |-----------|----------|-----------|
> | Alert engine (§7.9) | Suppressed — no alerts generated or escalated | Truck is already being serviced; alerts are meaningless |
> | Path A inference | Paused — no classification produced | No sensor data available; inference would produce garbage |
> | Path B RUL | Undefined (null) — not computed | No degradation trajectory to predict; RUL is meaningless during repair |
> | Health Index | Undefined (null) — displayed as "In Service" | Operator dashboard shows maintenance status, not health |
> | Edge device | Idle — no data collection or inference | Truck engine is off or in shop diagnostic mode |
> | Cloud pipeline | Marks window as non-operational | Prevents repair gaps from being interpreted as sensor failure |
>
> **Transition rules:**
> - Entry: Inspection outcome = Repair (§10.10.4.1)
> - Exit: Repair complete, truck returns to service → state = HEALTHY
> - Duration: Determined by repair duration (1–10 days per §10.10.4.1)
>
> **Why this must be explicit:** Without a formal MAINTENANCE state, the system would
> interpret post-repair sensor data as a sudden improvement from CRITICAL to HEALTHY —
> producing false "improvement" signals, anomalous RUL spikes, and potential data leakage
> during model retraining. The MAINTENANCE state creates a clean boundary between
> operational episodes.
>
> **Synthetic data vs. production:** In the synthetic dataset, MAINTENANCE manifests as
> *missing Parquet files* — no rows exist for repair days. The MAINTENANCE state is not
> stored as a label in the Parquet schema. Instead, the `episode_id` column (§9.6.1) and
> `maintenance_log.json` encode when repairs occurred. The state machine above describes
> **production system behavior** — how the deployed inference pipeline should handle trucks
> known to be in service. The ML models never see MAINTENANCE-labeled rows during training;
> they learn to handle post-repair resets via episode segmentation (Amendment L).

**Add to §7.3 (Output Layer — Two Audiences):**

> **Maintenance status in operator view:** When a truck is in MAINTENANCE state, the
> operator dashboard should display:
> - Current status: "In Maintenance" (not a health score)
> - Repair type and estimated completion (from maintenance log)
> - Last known health state before repair (for reference)
> - Predicted post-repair status: HEALTHY (assuming successful repair)

---

## Amendment J: §10.4 / §10.5 — Simulation Pause During Maintenance

**Add to §10.4 (Engine RPM and Load Profiles):**

> **Maintenance pause:** When a truck is in MAINTENANCE state (§10.10.4.1, Amendment I):
>
> - The Markov operating mode chain is **suspended**. No state transitions occur.
>   The chain resumes from the IDLE state on the return-to-service day.
> - The engine-hour counter **stops**. Repair days do not accumulate engine hours.
>   This ensures degradation models (which use engine-hours as input) do not advance
>   during repair.
> - All degradation clocks **freeze**. No severity progression occurs while the truck
>   is offline. (For the repaired fault, this is moot — severity is reset to 0. For
>   any secondary faults on multi-fault trucks, degradation pauses during the shop visit
>   since the engine is not running.)
> - Ambient temperature simulation **continues** (calendar time advances), but thermal
>   sensor state is irrelevant (no data generated). On return-to-service, thermal state
>   is re-initialized from idle baselines at the current ambient temperature.
>
> **Implementation:** The day-level generation loop checks the truck's maintenance state
> before generating data. If `truck.state == MAINTENANCE`, the day is skipped entirely:
> no Markov simulation, no feature generation, no Parquet output. The simulation advances
> the calendar day counter but nothing else.

---

## Amendment K: §10.7 — RUL Semantics Across Maintenance Events

**Add to §10.7 (Ground Truth Labeling Strategy), after Amendment G content:**

> **RUL formalization across lifecycle states:**
>
> RUL (Remaining Useful Life) is defined as the time remaining until the active fault
> reaches end-of-life (Stage 4 completion). Its value depends on the truck's lifecycle
> state:
>
> | State | RUL Value | Rationale |
> |-------|-----------|-----------|
> | HEALTHY (no fault) | ∞ (sentinel: 99999.0) | No fault to predict failure for |
> | DEGRADING (Stage 1–2) | `onset + total_life - t_current` | Deterministic from degradation model |
> | IMMINENT (Stage 3 early) | `onset + total_life - t_current` | Same formula, smaller value |
> | CRITICAL (Stage 3 late / Stage 4) | `onset + total_life - t_current` | Approaching 0 |
> | MAINTENANCE | Undefined (null) | No data produced; RUL is not computable |
> | Improving (monitor-improve) | ∞ (sentinel: 99999.0) | Fault is receding, not approaching failure |
>
> **RUL is deterministic in synthetic data.** It is computed directly from the fault's onset
> time and total life span — both known quantities in the generator. There is no learned
> component. The ML model (Path B) must *learn* to predict this value from sensor features
> alone; the ground truth RUL is the training target.
>
> **RUL after repair:** RUL = ∞. The repair eliminates the fault. There is no residual
> degradation, no repair-type-dependent offset, and no usage-history adjustment. This is
> a simplification — in reality, a repaired component may have shorter life than a new one.
> This is flagged as **EXTRAPOLATION (E20)**: post-repair RUL assumes full restoration.
> Must be validated against field data if available.
>
> **RUL during improvement:** RUL = ∞. The fault severity is decaying toward 0. Since the
> trajectory is moving away from failure, predicting "time to failure" is undefined. The
> Path B model should learn that improving features correspond to ∞ RUL — equivalently,
> that the truck is returning to healthy baseline.

---

## Amendment L: §9.6 / §10.5 — Operational Episode Segmentation

**Add new subsection §9.6.1 (or append to §9.6):**

> ### 9.6.1 Operational Episode Segmentation
>
> Each truck's operational history is divided into **episodes** — contiguous periods of
> operation separated by maintenance events. An episode begins when a truck enters service
> (day 0 or return from repair) and ends when the truck enters MAINTENANCE state or the
> simulation ends.
>
> **Episode ID column:** Each Parquet row includes an `episode_id` (int32) metadata field.
> The episode ID is an integer that increments each time the truck returns from repair:
>
> | Event | episode_id |
> |-------|-----------|
> | Truck starts simulation (day 0) | 0 |
> | First repair complete, returns to service | 1 |
> | Second repair complete, returns to service | 2 |
> | ... | ... |
>
> Healthy trucks that never undergo repair have `episode_id = 0` for all rows.
>
> **Natural resolution (monitor-improve) does NOT increment episode_id.** When a fault
> resolves naturally through severity decay (§10.10.4.2), there is no repair gap and no
> discontinuous state jump. The transition from degrading-to-healthy is gradual and visible
> in the sensor data. The sequence model can safely span this transition — it represents
> a physically continuous trajectory (improvement), not an artificial reset. The
> `maintenance_log.json` records monitor-improve events with `"outcome": "monitor_improve"`
> for traceability, but they do not create episode boundaries.
>
> **Parquet schema update:** The schema gains one metadata column:
>
> ```
> Columns: 230 total (was 229)
>   5 metadata:  timestamp, truck_id, engine_type, day_index, episode_id (int32)
>   221 features
>   4 labels
> ```
>
> **Code update required:** `src/storage/schema_definition.py` and
> `src/features/feature_vector.py` must be updated to reflect 230 columns and include
> `episode_id` in the metadata column list. The stale "201 columns" comment in
> `schema_definition.py` must also be corrected.
>
> **Purpose — preventing data leakage in sequence models:**
>
> Path B (CNN-LSTM-Attention) uses sliding windows of 50 consecutive 60-second samples.
> These sequences must **never cross episode boundaries**. A sequence spanning
> `[...degrading → repair gap → healthy...]` would teach the model that "failure causes
> improvement" — a physically impossible relationship that will break inference.
>
> **Data loader contract:**
> - Sequences are constructed **within** a single episode only
> - When an episode has fewer than 50 windows, it is either padded (with masking) or
>   discarded, depending on configuration
> - The episode boundary acts as a hard segmentation point — no lookback across it
> - Path A (row-level XGBoost) is not affected by this constraint since each row is
>   independent, but the `episode_id` is still available for stratified analysis
>
> **Maintenance event log:** In addition to the `episode_id` column, each truck's metadata
> directory contains a `maintenance_log.json` file:
>
> ```json
> {
>   "truck_id": 42,
>   "events": [
>     {
>       "episode_id_before": 0,
>       "episode_id_after": 1,
>       "fault_repaired": "FM-01",
>       "detection_day": 45,
>       "detection_stage": "stage2",
>       "inspection_day": 52,
>       "outcome": "repair",
>       "repair_start_day": 52,
>       "repair_end_day": 54,
>       "return_to_service_day": 54
>     },
>     {
>       "episode_id_before": 1,
>       "episode_id_after": 2,
>       "fault_repaired": "FM-05",
>       "detection_day": 110,
>       "detection_stage": "stage3",
>       "inspection_day": 112,
>       "outcome": "repair",
>       "repair_start_day": 112,
>       "repair_end_day": 115,
>       "return_to_service_day": 115
>     }
>   ]
> }
> ```
>
> This log enables:
> - Post-hoc analysis of maintenance patterns
> - Verification that the ML pipeline correctly segments episodes
> - Training data augmentation (e.g., using repair history as auxiliary features)
> - Distinguishing maintenance gaps from sensor outages in production deployment

---

## Amendment M: §7.9 — Alert State Machine MAINTENANCE Transition

**Add to §7.9 (Alert Fatigue Prevention — Per-Mode State Machine):**

> **MAINTENANCE transition:** When a truck enters MAINTENANCE state (§10.10.4.1), all
> active alert state machines for that truck transition to INACTIVE:
>
> ```
> ACTIVE/ACKNOWLEDGED/SHELVED → INACTIVE    [truck enters MAINTENANCE]
> ```
>
> This transition:
> - Clears all pending alerts for the truck
> - Cancels any shelve timers
> - Resets escalation counters
> - Does **not** generate a "condition cleared" notification (the condition was not
>   cleared by improvement — it was cleared by repair, which is a different event)
>
> When the truck returns from MAINTENANCE to HEALTHY, alert state machines are
> re-initialized in the INACTIVE state. New alerts can only fire when the detection
> model (§10.10.2) identifies a new fault condition.
>
> **Designed suppression during MAINTENANCE:** Per ISA 18.2, alerts from a truck known
> to be in the shop are classified as "designed suppression" — the system knows the
> alerts would be spurious (no engine running, diagnostic equipment may trigger false
> readings). This is analogous to the DPF regen suppression already described in §7.9.

---

## Amendment N: Additional 191→221 Feature Count References

**The following paragraphs in the spec reference "191 features" and must be updated to 221:**

> **Document Organization (paragraph [22]):**
> Change: "Sections 7-8: Feature extraction (191 features)" →
> "Sections 7-8: Feature extraction (221 features)"
>
> **§9.4 Feature Vector Summary (paragraph [269]):**
> Change: "Final unified feature vector: 191 features per 60-second window" →
> "Final unified feature vector: 221 features per 60-second window"
>
> **§9.6 Feature Storage Format (paragraph [274]):**
> Change: "Total column count: ~200 (191 features + metadata + label)" →
> "Total column count: 230 (221 features + 5 metadata + 4 labels)"
>
> **§10.5 Step 6 (paragraph [462]):**
> Change: "Produces 191 features per 60-second window" →
> "Produces 221 features per 60-second window"
>
> **§13.1 Phase 1 (paragraph [785]):**
> Change: "Implement 191-feature extraction from raw sensors" →
> "Implement 221-feature extraction from raw sensors"
>
> These are all instances of the same update covered by Amendment C (§9.4). They are
> listed here explicitly to ensure none are missed during the docx merge.
