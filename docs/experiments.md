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
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning, 5 epochs, lr `1e-5` | same as Granite | GroupKFold 5, fold0, session id group | 0.71821 | - | `model/granite-311m-fold0-e5-lr1e-5` | Lower lr + more epochs underperformed original 3 epoch run |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning, 5 epochs, lr `1e-5` + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.72698 | - | `model/granite-311m-fold0-e5-lr1e-5/logit_bias.json` | Bias helps but remains below current SOTA |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h16/l512 | `[META]+[HIST]+[CUR]`, recent 8 user-action pairs | GroupKFold 5, fold0, session id group | 0.73251 | - | `model/granite-311m-fold0-h16-l512` | Raw similar to original h12/l512 |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h16/l512 + logit bias tuning | same as above | GroupKFold 5, fold0, session id group | 0.73906 | - | `model/granite-311m-fold0-h16-l512/logit_bias.json` | Best fold0 local result so far |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h20/l768 | `[META]+[HIST]+[CUR]`, recent 10 user-action pairs | GroupKFold 5, fold0, session id group | 0.73282 | - | `model/granite-311m-fold0-h20-l768` | Longer context did not beat h16 after tuning |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h20/l768 + logit bias tuning | same as above | GroupKFold 5, fold0, session id group | 0.73796 | - | `model/granite-311m-fold0-h20-l768/logit_bias.json` | Above original, below h16/l512 |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h8/l512 | `[META]+[HIST]+[CUR]`, recent 4 user-action pairs | GroupKFold 5, fold0, session id group | 0.72240 | - | `model/granite-311m-fold0-h8-l512` | Too little history hurts |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h8/l512 + logit bias tuning | same as above | GroupKFold 5, fold0, session id group | 0.72692 | - | `model/granite-311m-fold0-h8-l512/logit_bias.json` | Not competitive |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h16/l512, fold1 | `[META]+[HIST]+[CUR]`, recent 8 user-action pairs | GroupKFold 5, fold1, session id group | 0.74430 | - | `model/granite-311m-fold1-h16-l512` | Ensemble candidate |
| 2026-07-02 | `ibm-granite/granite-embedding-311m-multilingual-r2`, h16/l512, fold1 + logit bias tuning | same as above | GroupKFold 5, fold1, session id group | 0.75159 | - | `model/granite-311m-fold1-h16-l512/logit_bias.json` | Strong fold1 validation score |
| 2026-07-02 | `BAAI/bge-m3` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.69687 | - | `model/bge-m3-router-fold0` | XLM-R 계열 multilingual encoder 비교. 3 epochs best |
| 2026-07-02 | `BAAI/bge-m3` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.71227 | - | `submissions/submit_bge-m3-fold0_bias_f1-0.71227_20260702.zip` | Bias tuning improves substantially, but still trails Granite |
| 2026-07-02 | `FacebookAI/xlm-roberta-large` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.58929 | - | `model/xlm-roberta-large-router-fold0-e1` | 1 epoch screening. Below BGE-M3 epoch1, not worth extending |
| 2026-07-02 | `FacebookAI/xlm-roberta-large` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.61100 | - | `model/xlm-roberta-large-router-fold0-e1/logit_bias.json` | Bias helps, still clearly behind BGE-M3/Granite |
| 2026-07-02 | `answerdotai/ModernBERT-base` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.32973 | - | `model/modernbert-base-router-fold0-e1` | 1 epoch screening. English-centric base model is a poor fit |
| 2026-07-02 | `intfloat/multilingual-e5-large` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.57579 | - | `model/e5-large-router-fold0-e1` | 1 epoch screening. Large E5 does not close the Granite gap |
| 2026-07-02 | `intfloat/multilingual-e5-large` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.59895 | - | `model/e5-large-router-fold0-e1/logit_bias.json` | Bias helps but remains below XLM-R large |
| 2026-07-02 | `Alibaba-NLP/gte-multilingual-base` fine-tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.56063 | - | `model/gte-multilingual-base-router-fold0-e1` | 1 epoch screening. Requires `trust_remote_code` and mismatched classifier head reset |
| 2026-07-02 | `Alibaba-NLP/gte-multilingual-base` fine-tuning + logit bias tuning | same as Granite | GroupKFold 5, fold0, session id group | 0.60119 | - | `model/gte-multilingual-base-router-fold0-e1/logit_bias.json` | Similar to E5-large, not a submission candidate |
| 2026-07-03 | Granite h16/l512 fold0+fold1 logit ensemble packaging/throughput | same as h16 Granite | GPU throughput benchmark on 3,000 fold0 validation rows | - | - | `submissions/submit_granite_h16_fold01_ensemble_20260703.zip` | Zip `992M`; 2-model inference estimated `164s` for 30,000 rows on GPU |
| 2026-07-03 | Granite LoRA integration smoke, r16 alpha32 targets `Wqkv,Wo,Wi` | same as h16 Granite | 32 train / 32 val smoke only | 0.23114 | - | `model/granite-lora-smoke` | PEFT integration, adapter save, merged full-model save, and inference load all verified. Score is not meaningful |
| 2026-07-03 | Granite LoRA r16 alpha32, 1 epoch + logit bias | same as h16 Granite | GroupKFold 5, fold0, session id group | 0.70433 | - | `model/granite-311m-lora-fold0-e1/merged/logit_bias.json` | 1 epoch is below BGE-M3 tuned and not enough |
| 2026-07-03 | Granite LoRA r16 alpha32, 3 epochs + logit bias, fp16 merged | same as h16 Granite | GroupKFold 5, fold0, session id group | 0.75337 | - | `submissions/submit_granite_lora_fold0_e3_fp16_bias_f1-0.75337_20260703.zip` | New best fold0 local. Zip `491M`; fp32 tuned was `0.75367`, fp16 tuned `0.75337` |

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
- Granite 5 epoch with lower lr (`1e-5`) dropped to `0.71821` raw / `0.72698` tuned, so the original `3 epochs`, `2e-5` schedule remains better.
- History sweep: h16/l512 is the best fold0 setting so far (`0.73906` tuned). h20/l768 improves over original but not h16, and h8 is clearly worse.
- Fold1 h16/l512 is strong (`0.75159` tuned on fold1 validation), so a 2-fold h16/l512 ensemble is the next submission candidate to test against runtime/zip limits.
- Broader next-step plan is in `docs/next_experiments.md`. Priority is h16 fold0/fold1 ensemble first, dynamic INT8 quantization probe second, then distillation/LoRA for cheaper extra folds. Pruning is lower priority unless structured layer dropping is paired with distillation.
- h16 fold0/fold1 ensemble fits the 10 minute runtime constraint by throughput estimate (`~164s` for 30k rows on GPU), but the zip is large (`992M`). If the competition accepts ~1GB submissions, this is the next public LB candidate.
- LoRA is now wired through `scripts/train_granite_lora_router.py`. Smoke run confirmed `1.2587%` trainable parameters and successful `merge_and_unload()` output, so full fold2/fold3 LoRA training is feasible.
- LoRA 3 epoch fold0 beat the previous fold0 local best after bias tuning (`0.75337` fp16 merged vs `0.73906` full fine-tune h16). This is the strongest next Public LB candidate before adding more folds.
