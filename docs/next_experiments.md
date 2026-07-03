# Next Experiment Plan

현재 최고 성능 축은 Granite h16/l512 + logit bias다. 이미 base encoder swap, lr/epoch, history length, fold 차이는 확인했으므로 다음 실험은 단순 파라미터 변경보다 `성능 유지 + 제출 비용 절감` 또는 `추가 일반화 신호`에 집중한다.

## Priority

| Priority | Idea | Expected Upside | Main Risk | First Check |
| --- | --- | --- | --- | --- |
| P0 | h16/l512 fold0 + fold1 logit ensemble | Public LB 상승 가능성이 가장 큼. 이미 모델 산출물이 있음 | zip 약 2배, 추론 시간 약 2배 | 공개 5건 smoke + hidden 30k 예상 시간 측정 |
| P0 | Dynamic INT8 quantization probe | CPU 추론/메모리 절감 가능. 새 학습 불필요 | GPU 제출에서는 이득이 작고 CPU 30k가 느릴 수 있음 | `scripts/probe_granite_dynamic_quantization.py --limit 1000` |
| P1 | Distillation to smaller encoder | Granite의 decision boundary를 작은 모델로 압축 가능 | 학생 모델이 Granite gap을 못 따라갈 수 있음 | Granite validation logits 캐시 후 E5-small/BGE-base KL + CE 학습 |
| P1 | LoRA/partial fine-tuning | fold 추가 학습 비용 감소, fold 다양성 확보 | 제출은 base+adapter만으로는 오프라인 로드 불가. merge 필요 | PEFT 설치 후 Granite LoRA rank 8/16 학습, merge 저장 |
| P2 | Layer/head pruning | 모델 크기와 runtime 직접 절감 가능 | ModernBERT 구조에서 무작위/비구조 pruning은 zip 감소가 거의 없고 성능 손실 큼 | layer drop 22->18 후 짧은 재학습 |
| P3 | Unstructured magnitude pruning | 구현은 쉬움 | sparse kernel 미사용이면 runtime/zip 이득 거의 없음 | 후순위 |

## Immediate Candidate: 2-Fold Ensemble

이미 존재하는 모델:

```text
model/granite-311m-fold0-h16-l512
model/granite-311m-fold1-h16-l512
```

추론 스크립트:

```bash
conda run -n digital python scripts/infer_granite_ensemble.py \
  --data-dir ../data \
  --model-dirs \
    ./model/granite-311m-fold0-h16-l512 \
    ./model/granite-311m-fold1-h16-l512 \
  --output-path ./output/submission_granite_h16_fold01_ensemble.csv \
  --batch-size 64 \
  --max-length 512 \
  --max-history-events 16
```

제출 템플릿은 `packaging/granite_ensemble_submit_script.py`다. 메모리 폭증을 피하기 위해 모델을 한 번에 하나씩 로드하고 logits를 누적한다.

주의: fold0/fold1 모델을 같은 fold validation에서 평균 내면 학습 데이터 누수가 생긴다. 이 앙상블은 로컬 CV보다 public/private 제출로 판단하는 후보에 가깝다.

## Quantization

가장 먼저 볼 것은 학습 후 동적 INT8 양자화다. 새 학습이 필요 없고, hidden 30,000건 추론 제약에 대한 비용 측정이 가능하다.

```bash
conda run -n digital python scripts/probe_granite_dynamic_quantization.py \
  --data-dir ../data \
  --model-dir ./model/granite-311m-fold0-h16-l512 \
  --fold 0 \
  --max-history-events 16 \
  --limit 1000
```

판정 기준:

- Macro-F1 하락이 `0.003` 이하이고 CPU speedup이 있으면 full validation으로 확대한다.
- CPU에서 느리거나 F1 하락이 크면 제출 경로에는 넣지 않는다.
- bitsandbytes 8-bit/4-bit는 평가 서버 오프라인 의존성 리스크가 커서, `requirements.txt` 포함 설치가 허용되는지 확인한 뒤에만 진행한다.

## LoRA

LoRA는 제출물 크기를 줄이기보다는 fold 추가 학습 비용을 줄이는 목적에 가깝다. 평가 서버가 오프라인이므로 adapter-only 제출은 부적합하고, 최종 제출은 `merge_and_unload()`로 병합한 full model을 저장하는 방식이어야 한다.

로컬 Granite/ModernBERT 모듈명 기준 target 후보:

```text
Wqkv, Wo, Wi
```

추천 시작점:

```text
rank=8 or 16
alpha=16 or 32
dropout=0.05
target_modules=["Wqkv", "Wo", "Wi"]
freeze base, train classifier
```

LoRA가 raw fold0 h16/l512의 `0.7325`에 근접하면 fold2/fold3을 싸게 늘려 ensemble diversity를 확보한다. `0.72` 아래면 full fine-tuning 대비 손실이 커서 중단한다.

## Pruning

pruning은 마지막에 본다. unstructured pruning은 PyTorch/HF 일반 추론에서 실제 속도나 zip 크기 이득으로 잘 이어지지 않는다. 한다면 layer drop 같은 구조적 pruning이 낫다.

추천 실험:

1. 22 layers에서 18 layers로 drop한 student를 초기화한다.
2. Granite teacher logits + label CE로 1 epoch distillation 한다.
3. h16/l512 조건에서 fold0 validation을 측정한다.

성공 기준은 `0.72+` Macro-F1과 30,000건 추론 시간의 확실한 감소다.
