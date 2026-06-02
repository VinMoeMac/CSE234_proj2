# Experiment Log — CSE234 Project 2: Schema Linking

All experiments use Qwen2.5-1.5B-Instruct as the base model unless noted.
Evaluation metric: Leaderboard Score = 0.5 × Table Score + 0.5 × Column Score on validation set (101 questions).

---

## Baseline (exp2-5ep-lr1e4)
**Score: 0.206 → 0.313 (with keyword fallback)**

- Model: Qwen2.5-1.5B-Instruct
- LoRA r=32, target: q/k/v/o_proj
- Epochs: 5, LR: 1e-4, batch: 1, grad_accum: 16
- Max seq len: 1024 (local), 2048 (DSMLP)
- Schema: basic (table + columns + types + PK/FK annotations)

**Findings:**
- 57/101 predictions empty — model outputting nothing valid
- Root cause: prompts truncated from left, cutting off system prompt
- SAP databases score 0.000 — cryptic table names (OCPN, OEGP) have no semantic match to NL questions
- NTSB scores 0.129 — large schema (40 tables, 1161 cols) exceeds context
- Biodiversity DBs score 0.5–0.6 — readable table/column names match well

**Keyword fallback added:** if model output is empty, use token matching between question and schema identifiers. Boosted score from 0.206 to 0.313.

---

## RapidFire Grid Experiments (schema-linking-v1 through v4)

RapidFire multi-config grid: 2 models × 2 LoRA ranks × 2 LRs = 8 configs per experiment.

**schema-linking-v1:** num_chunks=4 (default) — each config only trained on ~75/301 examples per chunk. Result: undertrained models, near-zero leaderboard scores.

**schema-linking-v3:** num_chunks=1 — each config trained on full dataset but only 1 epoch due to RapidFire epoch/chunk interaction. Steps: 19/config. Result: token accuracy ~0.60 but hallucinated table names → empty predictions after validation.

**schema-linking-v4:** num_chunks=1, 5 epochs — same issue: RapidFire gave 1 epoch per config regardless of --epochs flag. Steps: 19/config. token accuracy ~0.62.

**Key insight:** RapidFire's chunk-based interleaved scheduling doesn't straightforwardly map epochs to full dataset passes. With small datasets (301 examples), num_chunks=1 still only trains each config for ~19 steps. Switched to train_simple.py (plain TRL SFTTrainer) for reliable epoch control.

---

## Schema Serialization Experiments (run_schema_experiments.sh)

All trained with: Qwen2.5-1.5B-Instruct, 5 epochs, LR=1e-4, batch=1, grad_accum=16, seq_len=2048.

| Approach | Description | Score |
|---|---|---|
| approach1-table-desc | Table hints derived from training questions | 0.297 |
| approach2-fk-links | FK relationships shown per table | **0.334** |
| approach3-col-sort | Columns sorted by question relevance | 0.295 |
| approach-combined | All three together | 0.311 |

**Findings:**
- FK links (approach2) best — showing which tables are connected helps model understand join patterns
- Table descriptions hurt slightly — keywords derived from training are noisy
- Column sorting alone not helpful — model wasn't trained to exploit the ordering
- Combined worse than FK links alone — the extra noise from approaches 1+3 hurts

---

## Inference Improvements (no retraining)

After approach2 training:

**Table sorting at inference:** Sort tables by column-name overlap with question before feeding to model. Most relevant tables appear first before truncation.
- Score: 0.334 → 0.354

**Fix truncation direction:** Original code truncated from left (removing system prompt). Fixed to truncate schema body from bottom, keeping system prompt + question always visible.
- Score: 0.354 → **0.398**

**Key insight:** The model was outputting SQL instead of JSON for SAP questions — it never saw the system prompt due to left truncation. After fix, model correctly outputs JSON but still picks wrong SAP tables (e.g. predicts OSSG instead of ORSC for "resource name" query).

**Root cause of SAP failure:** SAP table names are opaque abbreviations. Column names ARE meaningful (empID, salary, roleID) but the model still picks wrong tables because many SAP tables share similar column patterns. Needs either more training data or retraining with table-sorted schema so model learns to trust ordering.

---

## Current Best: 0.398

Per-database breakdown at 0.398:
| Database | n | Avg F1 |
|---|---|---|
| SBODemoUS-* (all 9) | 20 | 0.000–0.125 |
| NTSB | 20 | 0.129 |
| NYSED_SRC2022 | 13 | 0.440 |
| ATBI | 8 | 0.362 |
| PacificIslandLandbirds | 8 | 0.418 |
| KlamathInvasiveSpecies | 8 | 0.357 |
| NorthernPlainsFireManagement | 8 | 0.622 |
| CratersWildlifeObservations | 8 | 0.604 |
| ASIS_20161108_HerpInv_Database | 8 | 0.720 |

**Next steps:**
1. Retrain with FK links + table sorting together (fk-sorted-v1) — match training to inference distribution
2. Add synthetic SAP training data — generate NL questions from SAP column names
3. Oversample SAP/NTSB during training

---

## In Progress

- **fk-sorted-v1:** Training with both FK links and table-by-column sorting. Expected to improve SAP/NTSB by teaching model to use table ordering.
- **Synthetic SAP data:** Generating ~60 additional NL→schema_links pairs for SAP databases using column names as semantic anchors.
