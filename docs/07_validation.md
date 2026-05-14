# Limitations & Validation

| Metric | Target | Current |
| --- | --- | --- |
| Answer Accuracy (human eval) | > 90% | 87% |
| Hallucination Rate | < 3% | 4.2% |
| User Satisfaction (CSAT) | > 4.0/5 | 4.1/5 |

**Known Limitations:**
1. Cannot reason across multiple policy documents simultaneously
2. Performance degrades for questions about policies updated within the last 48 hours (RAG index refresh lag)

{{findings:credit_risk}}
