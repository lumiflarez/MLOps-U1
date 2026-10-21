# Jailbreak Prompt Detection - Requirements v1.1

Version: 1.1
Date: 2026-10-21
Author: Saule Gylyte

## 1.1 Problem
LLM chat applications receive user prompts that try to make the model ignore its safety rules
("jailbreaks"). One way to defend against them is a filter that checks prompts before they
are sent to the LLM.

This project builds the core of such a filter: a **classifier model** that labels a prompt as
**jailbreak** (1) or **benign** (0).

## 2. Functional requirements (FR)
| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The model takes a prompt (text) and returns a label: 1 = jailbreak, 0 = benign | Must |
| FR-02 | The pipeline (prepare -> train -> evaluate) rebuilds the model from raw data with one command (`dvc repro`) | Must |
| FR-03 | Every training run is logged to the MLflow server in the cluster: parameters, metrics, model | Must |
| FR-04 | Any past run can be reproduced from its MLflow run ID (`python reproduce.py <run_id>`) | Should |

## 3. Non-functional requirements (NFR)
| ID | Category | Requirement | KPI | 
|---|---|---|---|
| NFR-01 | Quality | Catch jailbreak prompts | Recall >= 0.95 |
| NFR-02 | Quality | Few false alarms | Precision >= 0.95 | 
| NFR-03 | Quality | Balanced overall quality | F1 >= 0.95 |
| NFR-04 | Generalisation | No strong overfitting | train_accuracy - test_accuracy <= 0.03 | 
| NFR-05 | Performance | Full pipeline runs fast on CPU | <= 2 min | 
| NFR-06 | Reproducibility | A past run is rebuilt with `reproduce.py` | < 2 min and metric delta <= 0.005 | 
| NFR-07 | Traceability | Every run is linked to its code and data | 100 % of runs tagged `git_commit` + `dvc_raw_md5` | 
| NFR-08 | Tracking | Runs are stored in the cluster MLflow | 100 % of runs in cluster |

## 4. Change history
| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-10-21 | Saule Gylyte | Initial document: problem, FR-01..04, NFR-01..05 |
| 1.1 | 2026-10-21 | Saule Gylyte | NFR-06..08 |