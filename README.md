# AI Agent Action Prediction

AI 코딩 에이전트 세션 상태에서 다음 행동(action)을 14개 클래스 중 하나로 예측하는 대회용 작업 저장소다.

## Experiments

시도한 모델과 성능 기록은 `docs/experiments.md`에 정리한다. 단순 모델/파라미터 변경 이후의 압축, LoRA, pruning, ensemble 후보는 `docs/next_experiments.md`에 정리한다.

## Baseline

현재 코드는 배포 baseline notebook을 Python script로 변환한 초기 버전이다.

- `scripts/baseline_train.py`: `current_prompt`만 사용해 TF-IDF + Logistic Regression 모델을 학습하고 `model/tfidf_logreg.pkl`로 저장한다.
- `scripts/baseline_inference.py`: 저장된 모델을 불러와 `data/test.jsonl`을 예측하고 `output/submission.csv`를 생성한다.

실행:

```bash
python scripts/baseline_train.py
python scripts/baseline_inference.py
```

산출물:

```text
model/tfidf_logreg.pkl
output/submission.csv
```

## Transformer Experiment

`intfloat/multilingual-e5-small` 기반 action router 실험을 추가했다. IBM Granite embedding 모델은 제외하고, 먼저 다국어 E5-small을 sequence classification으로 fine-tuning하는 구성이다.

선택 이유:

- 한국어/영어/mixed 입력을 모두 처리할 수 있다.
- small급 모델이라 T4 16GB, hidden test 30,000건 추론 제한 안에서 운영하기 쉽다.
- `current_prompt`만 쓰는 baseline보다 `history`, `session_meta`, workspace 상태를 함께 넣을 수 있다.

구성:

- `src/action_router/features.py`: JSONL 샘플을 모델 입력 텍스트로 렌더링한다.
- `scripts/train_e5_router.py`: E5-small sequence classifier fine-tuning.
- `scripts/infer_e5_router.py`: fine-tuned 모델로 `submission.csv` 생성.
- `requirements-transformer.txt`: transformer 실험용 의존성.

입력 텍스트에는 다음 정보를 포함한다.

- `current_prompt`
- 최근 history의 user 발화와 assistant action/result
- `user_tier`, `language_pref`, token budget bucket, turn index, elapsed bucket
- workspace language mix, LOC, git dirty, open files, last CI status

학습:

```bash
pip install -r requirements-transformer.txt
python scripts/train_e5_router.py \
  --data-dir ./data \
  --model-name intfloat/multilingual-e5-small \
  --output-dir ./model/e5-small-router \
  --epochs 3 \
  --batch-size 16 \
  --max-length 512 \
  --max-history 8
```

추론:

```bash
python scripts/infer_e5_router.py \
  --data-dir ./data \
  --model-dir ./model/e5-small-router \
  --output-path ./output/submission.csv \
  --batch-size 64
```

학습된 모델 디렉터리는 제출 시 `model/e5-small-router` 형태로 포함하고, 평가 서버에서는 인터넷 없이 로컬 모델만 로드해야 한다.

## Granite Experiment

현재 주력 모델은 `ibm-granite/granite-embedding-311m-multilingual-r2` 기반 sequence classifier다.

구성:

- `scripts/train_granite_router.py`: Granite fine-tuning.
- `scripts/infer_granite_router.py`: Granite 모델 추론.
- `scripts/tune_granite_bias.py`: validation logits 기반 class logit bias tuning.
- `packaging/granite_submit_script.py`: 제출 zip에 들어갈 self-contained `script.py` 템플릿.
- `requirements-granite.txt`: Granite 제출/학습용 의존성. `transformers==4.48.3` 필요.

입력 직렬화:

```text
[META] tier=... pref=... turn=... budget=... lang=... ci=... git=... open=...
[HIST] U: ... | A[action] args -> result | ...
[CUR] current_prompt
```

학습:

```bash
conda run -n digital python scripts/train_granite_router.py \
  --data-dir ../data \
  --model-name ibm-granite/granite-embedding-311m-multilingual-r2 \
  --output-dir ./model/granite-311m-fold0 \
  --fold 0 \
  --n-splits 5 \
  --max-length 512 \
  --max-history-events 12 \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --grad-accum 4 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --seed 42
```

추론:

```bash
conda run -n digital python scripts/infer_granite_router.py \
  --data-dir ../data \
  --model-dir ./model/granite-311m-fold0 \
  --output-path ./output/submission_granite.csv \
  --batch-size 64
```

Logit bias tuning:

```bash
conda run -n digital python scripts/tune_granite_bias.py \
  --data-dir ../data \
  --model-dir ./model/granite-311m-fold0 \
  --fold 0 \
  --n-splits 5 \
  --max-length 512 \
  --max-history-events 12 \
  --batch-size 64
```

위 명령은 `model/granite-311m-fold0/logit_bias.json`을 생성한다. `scripts/infer_granite_router.py`와 제출용 `packaging/granite_submit_script.py`는 이 파일이 있으면 자동으로 logits에 bias를 더한다.

## Encoder Model Sweep

Granite가 E5-small보다 크게 앞선 뒤에는 같은 입력 직렬화와 GroupKFold 조건에서 더 강한 multilingual encoder를 비교한다. 우선 후보는 `BAAI/bge-m3`다. 크기는 Granite와 비슷하지만 XLM-R 계열이라 inductive bias가 달라, Granite 점수가 단순 모델 크기 효과인지 ModernBERT/Granite 계열 효과인지 가르는 데 유용하다.

구성:

- `scripts/train_encoder_router.py`: 임의 Hugging Face encoder sequence classifier fine-tuning.
- `scripts/infer_encoder_router.py`: fine-tuned encoder 모델 추론.
- `scripts/tune_encoder_bias.py`: validation logits 기반 class logit bias tuning.
- `packaging/encoder_submit_script.py`: 제출 zip에 들어갈 generic encoder `script.py` 템플릿. zip 안 모델 경로는 `model/encoder-router`로 맞춘다.

BGE-M3 학습:

```bash
conda run -n digital python scripts/train_encoder_router.py \
  --data-dir ../data \
  --model-name BAAI/bge-m3 \
  --output-dir ./model/bge-m3-router-fold0 \
  --fold 0 \
  --n-splits 5 \
  --max-length 512 \
  --max-history-events 12 \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --seed 42
```

추론 및 bias tuning:

```bash
conda run -n digital python scripts/infer_encoder_router.py \
  --data-dir ../data \
  --model-dir ./model/bge-m3-router-fold0 \
  --output-path ./output/submission_bge_m3.csv \
  --batch-size 32

conda run -n digital python scripts/tune_encoder_bias.py \
  --data-dir ../data \
  --model-dir ./model/bge-m3-router-fold0 \
  --fold 0 \
  --n-splits 5 \
  --max-length 512 \
  --max-history-events 12 \
  --batch-size 32
```

다음 후보:

- `Alibaba-NLP/gte-multilingual-base`: multilingual retrieval encoder. 필요하면 `--trust-remote-code`를 켠다.
- `FacebookAI/xlm-roberta-large`: 강한 범용 multilingual baseline. 추론 시간과 zip 크기를 먼저 확인한다.

Screening 결과:

- `FacebookAI/xlm-roberta-large`, 1 epoch: `0.58929`, bias tuning 후 `0.61100`.
- `answerdotai/ModernBERT-base`, 1 epoch: `0.32973`.
- `intfloat/multilingual-e5-large`, 1 epoch: `0.57579`, bias tuning 후 `0.59895`.
- `Alibaba-NLP/gte-multilingual-base`, 1 epoch: `0.56063`, bias tuning 후 `0.60119`.

모두 Granite/BGE-M3보다 낮아서 제출 후보로 패키징하지 않는다.

## Packaging Submit Zip

대회 제출은 prediction CSV가 아니라 `script.py`, `requirements.txt`, 학습된 모델을 포함한 zip이다.

Granite 제출 zip 생성:

```bash
rm -rf /tmp/granite_submit
mkdir -p /tmp/granite_submit/model submissions
cp packaging/granite_submit_script.py /tmp/granite_submit/script.py
cp requirements-granite.txt /tmp/granite_submit/requirements.txt
cp -a model/granite-311m-fold0 /tmp/granite_submit/model/
cd /tmp/granite_submit
zip -qr /home/seongmin/research/projects/digital/agent-action-prediction/submissions/submit_granite-311m-fold0_bias.zip .
```

패키지 검증:

```bash
rm -rf /tmp/granite_submit_test
mkdir -p /tmp/granite_submit_test
unzip -q /home/seongmin/research/projects/digital/agent-action-prediction/submissions/submit_granite-311m-fold0_bias.zip -d /tmp/granite_submit_test
ln -s /home/seongmin/research/projects/digital/data /tmp/granite_submit_test/data
cd /tmp/granite_submit_test
conda run -n digital python script.py
```

정상 실행되면 아래 파일이 생성된다.

```text
output/submission.csv
```

Generic encoder 제출 zip 생성:

```bash
rm -rf /tmp/encoder_submit
mkdir -p /tmp/encoder_submit/model submissions
cp packaging/encoder_submit_script.py /tmp/encoder_submit/script.py
cp requirements-granite.txt /tmp/encoder_submit/requirements.txt
cp -a model/bge-m3-router-fold0 /tmp/encoder_submit/model/encoder-router
cd /tmp/encoder_submit
zip -qr /home/seongmin/research/projects/digital/agent-action-prediction/submissions/submit_bge-m3-fold0_bias_f1-0.71227_20260702.zip .
```

Granite h16/l512 fold0+fold1 앙상블 추론:

```bash
conda run -n digital python scripts/infer_granite_ensemble.py \
  --data-dir ../data \
  --model-dirs \
    ./model/granite-311m-fold0-h16-l512 \
    ./model/granite-311m-fold1-h16-l512 \
  --output-path ./output/submission_granite_h16_fold01_ensemble.csv \
  --batch-size 64 \
  --max-history-events 16
```

앙상블 제출 zip 생성:

```bash
rm -rf /tmp/granite_ensemble_submit
mkdir -p /tmp/granite_ensemble_submit/model
cp packaging/granite_ensemble_submit_script.py /tmp/granite_ensemble_submit/script.py
cp requirements-granite.txt /tmp/granite_ensemble_submit/requirements.txt
cp -a model/granite-311m-fold0-h16-l512 /tmp/granite_ensemble_submit/model/
cp -a model/granite-311m-fold1-h16-l512 /tmp/granite_ensemble_submit/model/
cd /tmp/granite_ensemble_submit
zip -qr /home/seongmin/research/projects/digital/agent-action-prediction/submissions/submit_granite_h16_fold01_ensemble.zip .
```

앙상블 GPU throughput 측정:

```bash
conda run -n digital python scripts/benchmark_granite_ensemble.py \
  --data-dir ../data \
  --model-dirs \
    ./model/granite-311m-fold0-h16-l512 \
    ./model/granite-311m-fold1-h16-l512 \
  --max-history-events 16 \
  --limit 3000 \
  --batch-size 64
```

Granite LoRA smoke/full 학습:

```bash
conda run -n digital python scripts/train_granite_lora_router.py \
  --data-dir ../data \
  --model-name ibm-granite/granite-embedding-311m-multilingual-r2 \
  --output-dir ./model/granite-311m-lora-fold2 \
  --fold 2 \
  --max-history-events 16 \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --grad-accum 4 \
  --learning-rate 5e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-merged
```

## Expected Data Layout

실행 시 데이터는 저장소 루트 기준 아래 위치에 둔다. 데이터 파일은 용량과 대회 규정상 git에 포함하지 않는다.

```text
data/
  train.jsonl
  train_labels.csv
  test.jsonl
  sample_submission.csv
```

## Action Classes

`read_file`, `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`, `respond_only`

## Run

주요 실행 명령은 위의 Baseline, Transformer Experiment, Granite Experiment, Packaging Submit Zip 섹션을 참고한다.
