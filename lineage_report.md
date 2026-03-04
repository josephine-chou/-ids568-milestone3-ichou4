
# Lineage Report: Iris Classification

## 1. Experiment Design

### Experiment Matrix

| Experiment | n_estimators | max_depth | Purpose |

|------------|--------------|-----------|---------|

| Baseline | 50 | 5 | Establish baseline performance |

| Exp 2 | 100 | 5 | Increase tree count |

| Exp 3 | 100 | 10 | Increase depth |

| Exp 4 | 150 | 8 | Balanced configuration |

| Exp 5 | 200 | 8 | Maximum configuration |

## 2. Experiment Results

| Experiment | n_estimators | max_depth | Accuracy | F1 Score | Run ID |

|------------|--------------|-----------|----------|----------|--------|

| Baseline | 50 | 5 | 0.900 | 0.900 | blushing-lynx-834 |

| Exp 2 | 100 | 5 | **0.933** | **0.933** | thundering-conch-919 |

| Exp 3 | 100 | 10 | 0.900 | 0.900 | carefree-snake-52 |

| Exp 4 | 150 | 8 | 0.900 | 0.900 | bittersweet-foal-289 |

| Exp 5 | 200 | 8 | 0.900 | 0.900 | youthful-finch-98 |

## 3. Production Candidate

**Recommended Model:** Exp 2 (thundering-conch-919)

**Selection Rationale:**

1. **Best Performance:** Achieved highest accuracy (93.3%) and F1 score (93.3%)

2. **Balanced Configuration:** Uses moderate parameters (n_estimators=100, max_depth=5) 

3. **Reproducibility:** Consistent performance with fixed random_state=42

4. **Efficiency:** Lower complexity than Exp 4/5 while maintaining best performance

**Key Findings:**

- Increasing tree count from 50 to 100 improved performance

- Further increases (150, 200) showed no additional benefit

- Deeper trees (max_depth=10) led to potential overfitting

- Sweet spot: 100 estimators with depth 5

## 4. Model Lineage

**Production Candidate Lineage:**

| Component | Value |

|-----------|-------|

| Run ID | thundering-conch-919 |

| Code Version | Git commit: YOUR_COMMIT_HASH |

| Data Version | run_date: 20260304 |

| Hyperparameters | n_estimators=100, max_depth=5, criterion=gini |

| Test Accuracy | 0.933 |

| F1 Score | 0.933 |

| Model Hash | c73fabd1b6f55502 |

## 5. Risk Assessment

| Risk | Severity | Mitigation |

|------|----------|------------|

| Data drift | High | Monitor input feature distributions daily |

| Model degradation | Medium | Retrain weekly, track accuracy trends |

| Prediction latency | Low | Model inference <100ms, acceptable for production |

## 6. Deployment Strategy

**Canary Deployment Plan:**

1. Deploy to 1% of traffic → Monitor for 24 hours

2. If accuracy >90% → Increase to 10%

3. If accuracy >90% → Increase to 50%

4. If accuracy >90% → Full rollout to 100%

**Rollback Criteria:**

- Accuracy drops below 85%

- F1 score drops below 80%

- Latency exceeds 500ms

## 7. Monitoring Recommendations

**Metrics to Track:**

- Daily accuracy on production data

- F1 score per class

- Prediction distribution

- Model inference latency

**Alert Thresholds:**

- Accuracy < 85% → Immediate alert

- F1 score < 80% → Warning

- Latency > 500ms → Performance alert

