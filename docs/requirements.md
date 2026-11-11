# Jailbreak Prompt Detection - Requirements v1.2

Version: 1.2
Date: 2026-11-11
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
| FR-05 | REST API: `POST /predict` returns label and jailbreak probability; `GET /health` returns the model version in use | Must |
| FR-06 | The API loads the model from the MLflow Model Registry by alias `@champion`, not from a file | Must |
| FR-07 | Training can run in the cluster as a Kubernetes Job | Must |
| FR-08 | A model reaches production only after passing a quality gate; the previous model version can be restored | Must |

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

## 3.1 Inference and CI/CD NFR (measured)
| ID | Category | Requirement | KPI | Measured | 
|---|---|---|---|---|
| NFR-09 | Latency | `/predict` response time | p95 <= 50 ms | 12.03 ms | 
| NFR-10 | Throughput | Requests served per second | >= 50 rps | 99.3 rps | 
| NFR-11 | Reliability | Failed requests under load | 0 % | 0 / 100 | 
| NFR-12 | Model size | Serialized model (`model.pkl`) | <= 5 MB | 481 KB | 
| NFR-13 | Image size | API Docker image | <= 1 GB | 845 MB | 
| NFR-14 | Quality gate | Deploy only if test accuracy >= threshold | threshold 0.95 | 0.973 passed; threshold 0.99 blocked deploy | 
| NFR-15 | CI speed | Full CI/CD pipeline (lint, test, gate, build, push) | <= 10 min | 2 min 56 s | 
| NFR-16 | Rollback | Restore previous model version in the cluster | <= 5 min | 14.7 s | 

## 4. Change history
| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-10-21 | Saule Gylyte | Initial document: problem, FR-01..04, NFR-01..05 |
| 1.1 | 2026-10-21 | Saule Gylyte | NFR-06..08 |
| 1.2 | 2026-11-11 | Saule Gylyte | FR-05..08, NFR-09..16 |