# Experiment Log

AI 코딩 에이전트 다음 행동 예측 실험 결과 기록이다.

## Score Terms

- `Local CV`: 로컬 train 데이터를 나눠 검증한 Macro-F1.
- `Public LB`: 대회 서버 public leaderboard Macro-F1.
- `Artifact`: 로컬 제출 zip 또는 공유 링크.

## Results

| Date | Model / Method | Input | Validation | Local CV | Public LB | Artifact | Note |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-07-01 | TF-IDF + Logistic Regression | `current_prompt` only | stratified holdout 20%, seed 42 | 0.4383 | - | `model/tfidf_logreg.pkl` | 배포 baseline을 `.py`로 변환해 실행 |
| 2026-07-01 | `intfloat/multilingual-e5-small` fine-tuning | `current_prompt` + recent history + session/workspace meta | stratified holdout 15%, seed 42 | 0.49345 | - | `model/e5-small-router` | 초기 E5 실험 |
| 2026-07-01 | `intfloat/multilingual-e5-small` fine-tuning | `current_prompt` + recent history + session/workspace meta | stratified holdout 20%, seed 42 | 0.48948 | - | `submissions/submit_e5-small-val20_f1-0.48948_20260701.zip` | baseline과 같은 split 비율로 비교 |
| 2026-07-01 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning | `[META] + [HIST] + [CUR]`, recent 6 user-action pairs | GroupKFold 5, fold0, session id group | 0.73255 | - | `submissions/submit_granite-311m-fold0_f1-0.73255_20260701.zip` | ModernBERT 기반 Granite 재현 |
| 2026-07-01 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning + logit bias tuning | same as above | GroupKFold 5, fold0, session id group | 0.73697 | 0.73078 | `submissions/submit_granite-311m-fold0_bias_f1-0.73697_20260701.zip` / [Drive](https://drive.google.com/file/d/1nIw48xZB1kVZmsO1v3o18E_3TR2NCQmu/view?usp=drive_link) | Current SOTA |
| 2026-07-02 | `BAAI/bge-m3` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.69687 | - | `model/bge-m3-router-fold0` | XLM-R 계열 multilingual encoder 비교. 3 epochs best |
| 2026-07-02 | `BAAI/bge-m3` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.71227 | - | `submissions/submit_bge-m3-fold0_bias_f1-0.71227_20260702.zip` | Bias tuning improves substantially, but still trails Granite |
| 2026-07-02 | `FacebookAI/xlm-roberta-large` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.58929 | - | `model/xlm-roberta-large-router-fold0-e1` | 1 epoch screening. Below BGE-M3 epoch1, not worth extending |
| 2026-07-02 | `FacebookAI/xlm-roberta-large` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.61100 | - | `model/xlm-roberta-large-router-fold0-e1/logit_bias.json` | Bias helps, still clearly behind BGE-M3/Granite |
| 2026-07-02 | `answerdotai/ModernBERT-base` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.32973 | - | `model/modernbert-base-router-fold0-e1` | 1 epoch screening. English-centric base model is a poor fit |
| 2026-07-02 | `intfloat/multilingual-e5-large` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.57579 | - | `model/e5-large-router-fold0-e1` | 1 epoch screening. Large E5 does not close the Granite gap |
| 2026-07-02 | `intfloat/multilingual-e5-large` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.59895 | - | `model/e5-large-router-fold0-e1/logit_bias.json` | Bias helps but remains below XLM-R large |
| 2026-07-02 | `Alibaba-NLP/gte-multilingual-base` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.56063 | - | `model/gte-multilingual-base-router-fold0-e1` | 1 epoch screening. Requires `trust_remote_code` and mismatched classifier head reset |
| 2026-07-02 | `Alibaba-NLP/gte-multilingual-base` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.60119 | - | `model/gte-multilingual-base-router-fold0-e1/logit_bias.json` | Similar to E5-large, not a submission candidate |

## Current Best

Current best submission candidate:

```text
submissions/submit_granite-311m-fold0_bias_f1-0.73697_20260701.zip
```

Summary:

- Base model: `ibm-granite/granite-embedding-311m-multilingual-r2`
- Local CV: `0.73697`
- Public LB: `0.73078`
- Method: fine-tuning + class logit bias tuning
- Model size: about `629M` unpacked
- Submit zip size: about `496M`

## Notes

- E5-small improved over the TF-IDF baseline, but Granite gave a much larger jump.
- The next comparison should keep Granite's serialization/split/training recipe fixed and only swap the base encoder. This isolates model family/scale from feature engineering.
- `BAAI/bge-m3` is the first non-Granite candidate because it is a strong multilingual encoder at a comparable scale.
- BGE-M3 reached `0.69687` after 3 epochs and `0.71227` with bias tuning. This suggests Granite's edge is not just from using a larger multilingual embedding model; the ModernBERT/Granite backbone or pretraining mixture is likely helping this action-routing task.
- XLM-R large underperformed BGE-M3 in the 1 epoch screen (`0.58929`, `0.61100` with bias), so extending it to 3 epochs is unlikely to beat Granite.
- Plain `answerdotai/ModernBERT-base` performed poorly (`0.32973` at epoch1), likely because it is not a multilingual/code-agent tuned checkpoint. The useful signal is not generic ModernBERT architecture alone.
- E5-large (`0.59895` with bias) and GTE multilingual base (`0.60119` with bias) both underperform XLM-R large/BGE-M3 in 1 epoch screens. Larger retrieval encoders are not enough by themselves.
- Granite requires `transformers==4.48.3` because it loads as `ModernBertForSequenceClassification`.
- The evaluation server default is `transformers==4.46.3`; Granite submissions should include a pinned `requirements.txt`.
- Logit bias tuning improved fold0 Macro-F1 from `0.732565` to `0.736965` without retraining.
