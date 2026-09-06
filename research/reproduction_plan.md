# ZeroMRG Implementation and Reproduction Plan

## 0. Scope, evidence labels, and non-negotiable constraints

This is the Phase 3 plan for reproducing the base ZeroMRG paper using only the supplied **COV-CTR** and **IU-Xray** datasets. It does not implement code, preprocess full datasets, download checkpoints, or start training.

The source archives remain immutable:

- `data/COV-CTR with English reports.zip`
- `data/IU-Xray.zip`

COV-CTR and IU-Xray will not be merged. COV-CT is not part of this reproduction scope despite stale COV-CT path references in `research/AGENTS.md`.

Every important decision uses one of these labels:

- **[PAPER-STATED]** — explicitly stated in the paper.
- **[INFERRED]** — the most direct interpretation of the paper or named checkpoint, but not stated explicitly.
- **[IMPLEMENTATION-ASSUMPTION]** — a necessary reproducibility choice where the paper is silent or inconsistent.
- **[RESOURCE-CONSTRAINED]** — an engineering adaptation for the current machine; it must be recorded and never presented as the paper configuration.

`MISSING FROM PAPER` identifies an unresolved omission. It is not presented as a decision.

### 0.1 Primary assumption registry

| ID | Decision | Basis |
|---|---|---|
| A01 | Keep the two datasets, manifests, vocabularies, prompt banks, checkpoints, and results separate. | **[IMPLEMENTATION-ASSUMPTION]** The paper reports dataset-specific experiments and gives no joint-training schedule. |
| A02 | Use the 714 unambiguous COV-CTR image/report keys; quarantine all six conflicting duplicate keys and 26 unannotated images. | **[IMPLEMENTATION-ASSUMPTION]** This avoids inventing a conflict winner or repeating pixels with different targets. |
| A03 | Use the 3,331 IU-Xray UIDs with both impression and findings for the primary experiment; retain excluded records in an audit manifest. | **[IMPLEMENTATION-ASSUMPTION]** The paper requires their concatenation but gives no missing-section policy. |
| A04 | Use seed 42 for splits and prompt sampling, with separately derived seeds for each operation/dataset. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: all random seeds. |
| A05 | Split COV-CTR 8:1:1 at unique image-key level and IU-Xray 70:10:20 at UID/study level. | Ratios are **[PAPER-STATED]**; grouping levels are **[IMPLEMENTATION-ASSUMPTION]** required to prevent image/view leakage. |
| A06 | Form IU-Xray target text as trimmed `impression + " " + findings`, in that order. | Order is **[PAPER-STATED]**; the single-space separator is **[IMPLEMENTATION-ASSUMPTION]**. |
| A07 | Use a training-only, dataset-specific, case-preserving word/punctuation vocabulary with `<pad>`, `<bos>`, `<eos>`, and `<unk>`. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: decoder tokenizer and vocabulary. |
| A08 | Use a standard causal Transformer decoder with cross-attention to one conditioning memory vector. Provisional configuration: 6 layers, `d_model=768`, 8 heads, FFN 3072, dropout 0.1, GELU, pre-norm, learned positional embeddings, maximum 512 report tokens. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: all decoder internals. The 768 dimension is **[INFERRED]** from the named checkpoint’s official specification. |
| A09 | For IU-Xray, encode every listed view independently, compute prompt enrichment per view, masked-mean the enriched view embeddings, then L2-normalize to one study vector. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: multi-view handling. This adds no trainable fusion module and retains study-level pairing. |
| A10 | Use validation BLEU-4 for early stopping and best-checkpoint selection. | **[IMPLEMENTATION-ASSUMPTION]** The paper states validation BLEU and patience 50 but not the BLEU order. |
| A11 | Use maximum 200 epochs, constant learning rate, and no weight decay unless evidence is later recovered. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: epoch limit, schedule, and weight decay. Early stopping remains authoritative. |
| A12 | Use greedy autoregressive decoding for the primary run. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: decoding algorithm and its hyperparameters. |
| A13 | Use the canonical preprocessing object returned with OpenAI CLIP `ViT-L/14`; assert its output shape rather than manually duplicating constants. | **[INFERRED]** from the official companion checkpoint instructions; image preprocessing is `MISSING FROM PAPER`. |
| A14 | Build 10%-paired and 100%-paired models independently from the same best text-only checkpoint, rather than fine-tuning 100% from the 10% model. | **[IMPLEMENTATION-ASSUMPTION]** `MISSING FROM PAPER`: checkpoint lineage for supervised variants. |

Assumptions A07–A13 must be isolated in configuration so later documentary evidence can replace them without rewriting the pipeline.

## 1. Exact implementation pipeline

### 1.1 Dataset-independent flow

```text
Immutable ZIP archive
    ↓
Read-only archive + CSV integrity validation
    ↓
Dataset-specific image/report pairing
    ↓
Audit manifest (valid, excluded, conflict, missing, duplicate flags)
    ↓
Deterministic train / validation / test manifest
    ├─────────────────────────────────────────────────────────────┐
    │                                                             │
    │ Text-only auto-encoding branch                              │ Image branch
    │                                                             │
    ├─ training reports                                           ├─ image(s) for one sample/study
    ├─ report normalization                                       ├─ canonical CLIP image preprocessing
    ├─ frozen M-CLIP text encoder                                 ├─ frozen/trainable CLIP ViT-L/14
    ├─ L2-normalized text embedding ψ(S)                          └─ per-image embedding f_M(I)
    └─ train Transformer decoder D_m with reconstruction CE                 │
          │                                                                 │
          └─ best text-only decoder checkpoint                              │
                                                                            │
Training reports ──random fixed selection──> prompt report IDs              │
    ↓                                                                       │
Frozen M-CLIP text encoder                                                  │
    ↓                                                                       │
Cached prompt bank f_p [N_p, d] ────────────────────────────────────────────┤
                                                                            ↓
                                  image→prompt scaled dot-product attention
                                                                            ↓
                                  f_M(I) + attended prompt embedding f_p(I)
                                                                            ↓
                                             L2-normalized embedding φ(I)
                                                                            ↓
                                  IU only: documented masked view mean + L2
                                                                            ↓
                                       trained autoregressive decoder D_m
                                                                            ↓
                                  generated tokens until END/EOS or max length
                                                                            ↓
                                           detokenized generated medical report
                                                                            ↓
                      dataset-specific BLEU/CIDEr/ROUGE-L/METEOR evaluation
```

- **[PAPER-STATED]** Text-only auto-encoding trains report reconstruction from normalized Multilingual-CLIP text embeddings.
- **[PAPER-STATED]** Image inference uses Multilingual-CLIP image features, report prompts, Eqs. (5)–(6), and the previously trained decoder.
- **[PAPER-STATED]** Zero-shot inference has no trainable parameters.
- **[IMPLEMENTATION-ASSUMPTION]** Only training-split reports may enter text-only training or prompt selection. This prevents validation/test text leakage, which the paper does not explicitly address.

### 1.2 Artifacts produced by the later pipeline

The implementation will create processed artifacts, never overwrite raw archives:

```text
data/processed/
├── cov_ctr/
│   ├── audit_manifest.jsonl
│   ├── valid_samples.jsonl
│   ├── splits.json
│   ├── prompt_ids.json
│   ├── vocabulary.json
│   └── embedding_cache/            # optional resource adaptation
└── iu_xray/
    ├── audit_manifest.jsonl
    ├── valid_studies.jsonl
    ├── splits.json
    ├── prompt_ids.json
    ├── vocabulary.json
    └── embedding_cache/

checkpoints/{cov_ctr,iu_xray}/{autoencode,pairs_10,pairs_100}/
results/{cov_ctr,iu_xray}/{zero_shot,pairs_10,pairs_100}/
```

- **[IMPLEMENTATION-ASSUMPTION]** Manifests use relative archive/member paths and stable IDs; no machine-specific absolute paths are stored.
- **[RESOURCE-CONSTRAINED]** Prefer reading images from immutable archives or extracting one processed copy only. Do not create multiple full IU-Xray image copies.

## 2. Mapping paper components to code

| Paper Component | Purpose | Input | Output | Pretrained Model | Frozen/Trainable | Dimensions | Dependencies / Loss | Proposed Code File |
|---|---|---|---|---|---|---|---|---|
| Multilingual encoder `E_m` | Encode report semantics for auto-encoding and prompt creation | List of report strings | Text embedding `ψ(S)` | `M-CLIP/XLM-Roberta-Large-Vit-L-14` **[PAPER-STATED]** | Frozen in text-only stage **[INFERRED]** from Fig. 1 | Paper omits exact `d`; official checkpoint indicates `[B,768]` **[INFERRED]** | `multilingual-clip`, Transformers; reconstruction CE acts on decoder output | `src/zeromrg/models/mclip_text.py` |
| L2 text normalization | Put report features on normalized embedding manifold | `[B,d]` | `[B,d]`, unit norm | None | No parameters | Same shape | PyTorch; Eq. (3), no separate loss | `src/zeromrg/models/normalization.py` |
| Vision encoder `E_v` / `f_M` | Encode medical raster image | `[B,V,3,H,W]` after collation | Per-view image embeddings `[B,V,d]` | OpenAI CLIP `ViT-L/14`, paired with M-CLIP **[INFERRED]** from named checkpoint | Frozen in zero-shot; paper says train encoder in paired stage | Paper gives `N_M×d_M`; checkpoint indicates `d=768` **[INFERRED]** | OpenAI CLIP, TorchVision/Pillow; supervised CE reaches it in paired stage | `src/zeromrg/models/clip_vision.py` |
| Prompt bank `f_p` | Supply fixed medical-report knowledge to visual features | Training report IDs/text | Cached prompt matrix `[N_p,d]` | M-CLIP text encoder | Frozen during inference **[PAPER-STATED]** | COV `N_p=500`, IU `N_p=1000`; `d=768` inferred | M-CLIP; no prompt loss | `src/zeromrg/models/prompt_bank.py` |
| Prompt attention `f_p(I)` | Retrieve a weighted prompt embedding for each image/view | Image embeddings `[B,V,d]`, prompts `[N_p,d]` | Attention `[B,V,N_p]`; attended prompt `[B,V,d]` | None | No parameters in printed equation | Softmax over `N_p` **[INFERRED]** | PyTorch; Eq. (6), scale `sqrt(d)`; no separate loss | `src/zeromrg/models/alignment.py` |
| Residual alignment `φ(I)` | Combine image and prompt semantics | `f_M(I)`, `f_p(I)` | L2-normalized `[B,V,d]` | None | No parameters | `d_M=d_p` **[PAPER-STATED]** | PyTorch; Eq. (5), no separate loss | `src/zeromrg/models/alignment.py` |
| Study/view reduction | Turn IU’s multiple enriched views into one study condition | `[B,V,d]`, view mask | `[B,d]` | None | No parameters | Masked mean over `V`, then unit norm | `MISSING FROM PAPER`; **[IMPLEMENTATION-ASSUMPTION]** A09 | `src/zeromrg/models/view_pooling.py` |
| Multilingual decoder `D_m` | Reconstruct/generate report autoregressively | Conditioning `[B,1,d]`; shifted token IDs `[B,T]` | Logits `[B,T,|Vocab|]` | None stated | Trainable in auto-encoding; frozen zero-shot; trainable paired stage | Exact architecture `MISSING FROM PAPER`; A08 | PyTorch Transformer; token CE | `src/zeromrg/models/report_decoder.py` |
| ZeroMRG wrapper | Route text embeddings or image/prompt embeddings into the same decoder | Text or image batch, tokens, masks | Logits or generated IDs | Encoders above | Stage-dependent | Preserves shapes above | All model modules | `src/zeromrg/models/zeromrg.py` |
| Reconstruction loss | Teach decoder to invert text embeddings | Decoder logits and report targets | Scalar CE | None | Gradients to decoder only | Ignore PAD **[IMPLEMENTATION-ASSUMPTION]** | Eq. (2), repaired as ordinary English NLL | `src/zeromrg/training/losses.py` |
| Image-conditioned loss | Train paired-data variants | Image-conditioned logits and targets | Scalar CE | None | Gradients to `E_v` and `D_m` per Sec. 3.4 | Ignore PAD **[IMPLEMENTATION-ASSUMPTION]** | Eq. (7) | `src/zeromrg/training/losses.py` |
| Autoregressive generator | Produce reports | `φ(I)`, BOS, max length | Token IDs and text | Trained decoder | Frozen/eval | Up to `T_max` | Greedy decoding A12; EOS/END stop | `src/zeromrg/evaluation/generation.py` |
| Metric evaluator | Compare generated and reference reports | Prediction/reference strings keyed by sample ID | Metric dictionary | None | No parameters | Corpus-level outputs | COCO-caption-compatible evaluation | `src/zeromrg/evaluation/metrics.py` |

## 3. COV-CTR implementation plan

### 3.1 Population and pairing

- **[IMPLEMENTATION-ASSUMPTION]** Primary population: 714 unique image IDs having exactly one annotation row.
- Quarantine, but preserve in the audit manifest:
  - six image IDs with two conflicting annotations;
  - 26 physical images with no annotation;
  - `.DS_Store`.
- Join `reports_ZH_EN.csv:image_id` to the exact archive member basename.
- Never group filenames by `%` suffix or publication stem; the dataset has no trustworthy study ID.
- Create an internal `sample_id` from a stable hash of the original image basename plus source-row number. Do not rename the image.

This population differs from the paper’s 728 images. **[IMPLEMENTATION-ASSUMPTION]** Results must be labeled “supplied-archive reproduction” and must not claim identical paper splits.

### 3.2 Fields

| Field | Plan |
|---|---|
| `reports_En` | Model target and text-only/prompt source **[PAPER-STATED]** |
| `image_id` | Pairing key and output provenance |
| Chinese `findings` | Retain in manifest; do not use as English target or model input |
| Chinese `impression` | Retain in manifest; do not append/translate |
| Chinese `terminologies` | Retain as auxiliary audit metadata; not a model input |
| `COVID` | Retain as auxiliary label and validate; not a model input/loss |

- There are no missing values among the retained 714 records.
- Strip the one trailing line feed from `reports_En`; preserve case and clinical punctuation.
- **[IMPLEMENTATION-ASSUMPTION]** Apply Unicode NFC and collapse only runs of whitespace to one space. Do not lowercase, stem, translate, or remove source text.

### 3.3 Image preprocessing

1. Decode by content with Pillow; do not trust suffix because 34 `.png` members contain JPEG.
2. Composite RGBA deterministically on black before RGB conversion. **[IMPLEMENTATION-ASSUMPTION]** Black avoids injecting a white border; record this choice.
3. Convert grayscale/RGB/RGBA to RGB.
4. Apply the canonical preprocess returned by OpenAI `clip.load("ViT-L/14")`. **[INFERRED]** Expected result is `[3,224,224]`, but assert the processor’s actual output rather than hard-code its statistics.
5. Do not apply DICOM windowing; these are rendered 8-bit rasters.
6. Do not remove publication annotations/circles or recrop lesions.

### 3.4 Split and prompt plan

- **[PAPER-STATED]** Split ratio: 8:1:1.
- **[IMPLEMENTATION-ASSUMPTION]** Shuffle the 714 unique keys with dataset split seed 42, then allocate by largest-remainder/declared deterministic rule: 571 train, 71 validation, 72 test.
- **[IMPLEMENTATION-ASSUMPTION]** Do not stratify by COVID because the paper says random split, not stratified split.
- **[PAPER-STATED]** Prompt count: 500.
- **[IMPLEMENTATION-ASSUMPTION]** Sample 500 distinct report IDs without replacement from the 571 training records using a separately derived prompt seed; persist ordered IDs.
- **[IMPLEMENTATION-ASSUMPTION]** The 10%-paired subset contains 57 distinct training keys, sampled once from and nested within the 571-key 100% pool. The zero-shot model uses zero image/report pairs but may use all training reports for text-only training and prompts.

### 3.5 Conceptual sample and loader behavior

```text
sample_id: str
dataset: "cov_ctr"
image: FloatTensor[3,H_clip,W_clip]
images: FloatTensor[1,3,H_clip,W_clip]
image_mask: BoolTensor[1] == True
image_ids: [original basename]
view_labels: [null]
study_id: null
raw_report: original reports_En
processed_report: normalized English text
decoder_input_ids: LongTensor[T]
decoder_target_ids: LongTensor[T]
token_mask: BoolTensor[T]
covid_label: 0|1                  # metadata only
raw_auxiliary_fields: {...}
validity_flags: {...}
```

- **[IMPLEMENTATION-ASSUMPTION]** Loader samples are unique physical-image keys, never raw CSV rows.
- Text-only mode must not open the image.
- Image mode loads one image and preserves a singleton view axis so the model interface matches IU-Xray.
- Collation pads report tokens and views separately and returns masks.
- Training shuffles; validation/test do not. **[IMPLEMENTATION-ASSUMPTION]**

## 4. IU-Xray implementation plan

### 4.1 Population and pairing

- Join `indiana_reports.uid` to all `indiana_projections.uid` rows, then resolve each projection filename under `images/images_normalized/`.
- **[IMPLEMENTATION-ASSUMPTION]** Primary population: the 3,331 UIDs with nonblank impression and findings.
- Exclude from primary training/evaluation but retain in audit manifests:
  - 6 findings-only UIDs;
  - 489 impression-only UIDs;
  - 25 UIDs with neither target section;
  - four physical images absent from `indiana_projections.csv`.
- Keep all authoritative projection rows for included UIDs, including the three byte-identical pairs. Flag duplicates; do not delete or silently reweight them. **[IMPLEMENTATION-ASSUMPTION]**

### 4.2 Study-level representation and leakage prevention

- One sample equals one `uid`/report/study, not one projection image.
- One sample carries a list of 1–5 images and aligned `Frontal`/`Lateral` labels.
- Split UIDs before any image-level expansion.
- Assert that the train, validation, and test UID sets are pairwise disjoint and that every image filename occurs in exactly one split.
- Never repeat the report as separate training samples for each view.

This preserves the real one-report/multiple-image relation. `MISSING FROM PAPER`: how the authors fed multiple images into singular `I`.

### 4.3 Report construction and fields

- **[PAPER-STATED]** Target report is the concatenation of impression and findings.
- **[IMPLEMENTATION-ASSUMPTION]** Construct `impression.strip() + " " + findings.strip()` after Unicode NFC and whitespace normalization.
- Preserve case, punctuation, and `XXXX` redaction placeholders.
- Do not treat strings such as `None.`/`None available` as CSV-null unless they occur in target sections and an explicit later audit supports doing so.

| Field | Plan |
|---|---|
| `uid` | Study ID, report ID, split key, and join key |
| `impression` | First target section |
| `findings` | Second target section |
| `filename` | Image ID from projection table |
| `projection` | View metadata used for audit/collation, not a paper model feature |
| `MeSH` | Retain original and semicolon-parsed list; not model input/loss |
| `Problems` | Retain original and semicolon-parsed list; not model input/loss |
| `image`, `indication`, `comparison` | Retain as metadata; do not concatenate into target |

### 4.4 Image preprocessing and view reduction

1. Decode `.dcm.png` members as PNG rasters, not DICOM.
2. Convert 8-bit grayscale to RGB.
3. Apply the canonical OpenAI CLIP `ViT-L/14` preprocess to each view independently.
4. Collate to `[B,V_max,3,H_clip,W_clip]`, where dataset-observed `V_max=5`, with `[B,V_max]` mask.
5. Encode only valid views, restore `[B,V,d]`, and compute prompt attention per valid view.
6. **[IMPLEMENTATION-ASSUMPTION]** Masked-mean the enriched valid-view embeddings and L2-normalize the study vector. Do not add a learned view-fusion module because the paper does not specify one.

### 4.5 Split and prompt plan

- **[PAPER-STATED]** Split ratio: 70:10:20.
- **[IMPLEMENTATION-ASSUMPTION]** Shuffle 3,331 UIDs using split seed 42 and deterministic largest-remainder allocation: 2,332 train, 333 validation, 666 test.
- **[PAPER-STATED]** Prompt count: 1,000.
- **[IMPLEMENTATION-ASSUMPTION]** Sample 1,000 distinct training UIDs without replacement with a separate prompt seed and persist their order.
- **[IMPLEMENTATION-ASSUMPTION]** The 10%-paired pool contains 233 training UIDs, sampled once and nested within the 2,332-UID 100% pool.
- A missing patient ID means patient-disjoint splitting cannot be guaranteed; this must appear in result metadata.

### 4.6 Conceptual sample and loader behavior

```text
sample_id: uid
dataset: "iu_xray"
images: FloatTensor[V,3,H_clip,W_clip]       # 1 <= V <= 5 before collation
image_ids: list[str]
view_labels: list["Frontal"|"Lateral"]
image_mask: BoolTensor[V]
study_id: uid
report_id: uid
raw_impression: str
raw_findings: str
raw_report: {impression, findings}
processed_report: impression + " " + findings
decoder_input_ids: LongTensor[T]
decoder_target_ids: LongTensor[T]
token_mask: BoolTensor[T]
mesh_terms: list[str]                       # metadata only
problem_labels: list[str]                   # metadata only
indication/comparison/image_description: str
validity_flags: {...}
```

- Text-only mode returns one report per UID without loading images.
- Image mode returns the complete view list.
- Collation pads views and report tokens independently.
- The loader never flattens studies into image-level samples.

## 5. Dataset experiment strategy

### Decision: A — train and evaluate separately

- **[IMPLEMENTATION-ASSUMPTION]** Run COV-CTR and IU-Xray as independent experiments with separate splits, vocabularies, text-only decoder checkpoints, prompt banks, optional paired fine-tunes, predictions, and metrics.
- **[PAPER-STATED]** The paper uses dataset-specific split ratios, prompt counts, target construction, and result tables.
- The paper claims a unified multimodal/multilingual framework, but `MISSING FROM PAPER`: a joint sampling schedule, shared checkpoint evidence, loss weighting, or combined train corpus.
- Combining the datasets would introduce an undocumented modality mixture and could leak evaluation vocabulary/report style across datasets. It is therefore not part of the primary reproduction.
- Neither dataset is assigned as pretraining data for the other. Each dataset’s training reports feed its own Stage 1 and prompt bank.

Per dataset, run these model conditions:

1. **0% paired / zero-shot:** text-only Stage 1, then frozen image inference.
2. **10% paired:** initialize from the same Stage-1 decoder, use the fixed 10% nested paired pool.
3. **100% paired:** initialize independently from the same Stage-1 decoder, use all training pairs.

These three ratios are **[PAPER-STATED]**. Their exact subset selection is **[IMPLEMENTATION-ASSUMPTION]**.

## 6. Model architecture plan and tensor flow

### 6.1 Encoder assets

- Text: `M-CLIP/XLM-Roberta-Large-Vit-L-14` **[PAPER-STATED]**.
- Image: corresponding OpenAI CLIP `ViT-L/14` **[INFERRED]**. The official M-CLIP model card states that the named Hugging Face checkpoint contains the multilingual text encoder and that the image model is loaded separately.
- Shared projected dimension: 768 **[INFERRED]** from the official M-CLIP model table; assert at runtime.

Official verification sources for later environment pinning:

- [M-CLIP model card](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)
- [Multilingual-CLIP repository/model table](https://github.com/FreddeFrallan/Multilingual-CLIP)
- [OpenAI CLIP repository](https://github.com/openai/CLIP)

No checkpoint is downloaded in this phase.

### 6.2 Text-only path

Let `B` be batch size, `T` padded report length, `d=768`, and `Vocab` the dataset-specific output vocabulary.

```text
processed reports: list[str], length B
    ↓ frozen M-CLIP text encoder, eval/no dropout
ψ_raw: FloatTensor[B,d]
    ↓ L2 normalize on last dimension
ψ: FloatTensor[B,d]
    ↓ unsqueeze memory axis
memory: FloatTensor[B,1,d]

decoder input IDs: LongTensor[B,T]
    ↓ token embedding + learned position + causal Transformer decoder
logits: FloatTensor[B,T,|Vocab|]
    ↓ CE against shifted targets, PAD ignored
scalar reconstruction loss
```

- **[INFERRED]** Teacher forcing and causal masking follow Eqs. (2)/(7), which condition token `t` on the ground-truth prefix.
- **[IMPLEMENTATION-ASSUMPTION]** A single normalized embedding is one cross-attention memory token.
- **[IMPLEMENTATION-ASSUMPTION]** Decoder input is `<bos>, y_1, …, y_{T-1}` and target is `y_1, …, <eos>`; `<eos>` implements the paper’s `END` token.

### 6.3 Image/prompt path

Let `V=1` for COV-CTR and `1≤V≤5` for IU-Xray; `N_p=500` or `1000`.

```text
images: FloatTensor[B,V,3,H_clip,W_clip]
view mask: BoolTensor[B,V]
    ↓ flatten valid B×V and encode with CLIP ViT-L/14
f_M: FloatTensor[B,V,d]

prompt report strings
    ↓ frozen M-CLIP text encoder once
f_p: FloatTensor[N_p,d]

scores = f_M @ f_p.T / sqrt(d): FloatTensor[B,V,N_p]
weights = softmax(scores, dim=-1): FloatTensor[B,V,N_p]
f_prompt = weights @ f_p: FloatTensor[B,V,d]
φ_view = normalize(f_M + f_prompt): FloatTensor[B,V,d]
    ↓ COV: select singleton; IU: masked mean then normalize (A09)
φ_study: FloatTensor[B,d]
    ↓ unsqueeze memory axis and decode autoregressively
generated IDs: LongTensor[B,<=T_max]
```

Required invariants:

- prompt and visual `d` must match;
- invalid/padded views must never contribute to attention or pooling;
- softmax weights must sum to one over `N_p` for every valid view;
- all normalized embeddings must be finite and unit-norm within tolerance;
- prompts remain fixed during zero-shot and paired generation unless contrary evidence is found.

### 6.4 Decoder specification gap

`MISSING FROM PAPER`: layers, conditioning mechanism, vocabulary, tokenization, hidden/FFN dimensions, attention heads, activation, normalization order, position representation, dropout rate, initialization, and weight tying.

**[IMPLEMENTATION-ASSUMPTION]** A08 is the primary transparent configuration. It is a conventional Transformer decoder, not a claimed recovery of the authors’ undisclosed decoder. The configuration must be printed in every result and ablation table.

**[IMPLEMENTATION-ASSUMPTION]** Initialize embeddings/linear layers with standard PyTorch Transformer initialization, share/tie decoder input and output embeddings only if shapes allow, and record the choice. Provisional primary choice: tied embeddings to reduce parameters.

## 7. Training stages

### 7.1 Stage 0 — validation, manifests, vocabulary

| Item | Plan |
|---|---|
| Input | Immutable archive and CSV files |
| Model components | None |
| Objective | Reproduce dataset-analysis counts; produce valid/excluded manifests and splits |
| Output | Audit manifest, valid manifest, split IDs, train-only vocabulary |
| Gate | All joins resolve; excluded counts match; split sets are disjoint; target text nonempty |

### 7.2 Stage 1 — text-only auto-encoding

| Property | Plan |
|---|---|
| Input dataset | Training reports from one dataset only **[IMPLEMENTATION-ASSUMPTION]** |
| Components | Frozen M-CLIP text encoder, L2 normalization, trainable decoder |
| Objective | Reconstruct the input report from its normalized text embedding **[PAPER-STATED]** |
| Loss | Autoregressive token cross-entropy **[PAPER-STATED]** |
| Frozen | M-CLIP text encoder **[INFERRED]** from Fig. 1 |
| Trainable | Decoder `D_m` **[INFERRED]** |
| Optimizer | SGD **[PAPER-STATED]** |
| Learning rate | `1e-3` **[PAPER-STATED]** |
| Momentum | `0.99` **[PAPER-STATED]** |
| Batch size | 6 **[PAPER-STATED]** |
| Dropout | Used **[PAPER-STATED]**; rate `MISSING FROM PAPER`; 0.1 **[IMPLEMENTATION-ASSUMPTION]** |
| Epochs | `MISSING FROM PAPER`; maximum 200 **[IMPLEMENTATION-ASSUMPTION]** |
| Stopping | No validation BLEU improvement for 50 epochs **[PAPER-STATED]**; BLEU-4 **[IMPLEMENTATION-ASSUMPTION]** |
| Checkpoint | Best validation BLEU-4 and last epoch, including decoder, vocab, optimizer, configuration, splits, RNG state **[IMPLEMENTATION-ASSUMPTION]** |

The malformed bilingual product in Eq. (2) is irrelevant to these English-only separate runs. **[INFERRED]** Implement the ordinary English negative log-likelihood, averaged over non-PAD target tokens.

### 7.3 Stage 2 — prompt-bank construction

| Property | Plan |
|---|---|
| Input | Fixed training report IDs only |
| Counts | COV-CTR 500; IU-Xray 1,000 **[PAPER-STATED]** |
| Encoder | Frozen M-CLIP text encoder |
| Output | Ordered IDs and `[N_p,768]` embedding matrix |
| Training | None; prompts are fixed **[PAPER-STATED]** for inference |
| Gate | IDs are unique/subset of train; no val/test IDs; embeddings finite; checksum persisted |

### 7.4 Stage 3 — 0%-paired zero-shot inference

| Property | Plan |
|---|---|
| Input | Validation/test images, fixed prompt bank, best Stage-1 decoder |
| Frozen | CLIP image encoder, prompt embeddings, decoder **[PAPER-STATED]** |
| Trainable | None |
| Objective/loss | No training loss; generate reports |
| Generation | Greedy, `<bos>` to `<eos>` or 512 tokens **[IMPLEMENTATION-ASSUMPTION]** |
| Evaluation | Dataset-specific metrics in Section 10 |

### 7.5 Stage 4 — optional 10%-paired training

| Property | Plan |
|---|---|
| Input | Nested 10% subset of training image/report samples |
| Initialization | Best Stage-1 decoder; pretrained image encoder; fixed prompt bank **[IMPLEMENTATION-ASSUMPTION]** |
| Frozen | Prompt embeddings; text encoder not in paired forward path **[IMPLEMENTATION-ASSUMPTION]** |
| Trainable | Image encoder `E_v` and decoder `D_m` **[PAPER-STATED]** |
| Objective/loss | Image-conditioned autoregressive CE, Eq. (7) **[PAPER-STATED]** |
| Optimizer/LR/momentum/batch | SGD / `1e-3` / `0.99` / 6 **[PAPER-STATED]** |
| Epochs | `MISSING FROM PAPER`; max 200 **[IMPLEMENTATION-ASSUMPTION]** |
| Stopping | Validation BLEU patience 50 **[PAPER-STATED]**; BLEU-4 selection **[IMPLEMENTATION-ASSUMPTION]** |
| Checkpoint | Best and last; never overwrite Stage-1 checkpoint |

### 7.6 Stage 5 — optional 100%-paired training

Same settings as Stage 4, but use the complete training pair pool and initialize independently from the same Stage-1 checkpoint. **[IMPLEMENTATION-ASSUMPTION]** Do not continue from the 10% checkpoint.

### 7.7 Checkpoint contents and lineage

Every checkpoint directory must include:

- model state for trainable modules;
- exact pretrained model identifiers and revisions/checksums, without needlessly duplicating frozen weights;
- optimizer and epoch state;
- best metric and patience counter;
- dataset name and regime (`autoencode`, `0`, `10`, `100`);
- split manifest hash, prompt manifest hash, vocabulary hash;
- full resolved configuration;
- Python/PyTorch/CUDA/dependency versions;
- random-number-generator states;
- execution date and hardware summary.

## 8. Pretrained models and dependencies

### 8.1 Checkpoints to obtain later

| Model/artifact | Role | Framework/library | Expected checkpoint | Paper explicit? | Approximate size | Frozen/trainable |
|---|---|---|---|---|---|---|
| M-CLIP XLM-RoBERTa-Large text encoder + projection | Encode reports/prompts to visual-compatible space | `multilingual-clip`, Hugging Face Transformers, PyTorch | `M-CLIP/XLM-Roberta-Large-Vit-L-14` | Yes **[PAPER-STATED]** | Official table: ~344M parameters; HF PyTorch artifact currently ~2.24 GB **[INFERRED]** from official distribution, not paper | Frozen in Stage 1/2/inference; not used in paired image forward |
| OpenAI CLIP ViT-L/14 visual encoder | Encode CT/X-ray rasters | OpenAI `clip`, PyTorch, Pillow | `ViT-L/14` loaded through official CLIP package | Companion visual model is **[INFERRED]** from named M-CLIP checkpoint/model card | Visual tower approximately ~304M parameters **[INFERRED]**; verify after loading | Frozen zero-shot; trainable 10%/100% per Sec. 3.4 |
| M-CLIP tokenizer | Tokenize text-encoder input only | Transformers | tokenizer files from the same M-CLIP revision | Checkpoint named, tokenizer not discussed | Small relative to weights | No parameters |
| Report decoder | Generate English report tokens | Native PyTorch | No pretrained checkpoint; trained in Stage 1 | Decoder type stated, checkpoint absent | Depends on A08 and per-dataset vocabulary; log exact count | Trainable Stage 1 and paired stages; frozen inference |

Important checkpoint clarification: the official M-CLIP model card says the Hugging Face repository contains the multilingual text encoder, while the corresponding `ViT-L/14` image model is obtained separately from OpenAI CLIP. The implementation must verify that both outputs are 768-dimensional and aligned before proceeding.

### 8.2 Required software, to be pinned during implementation

- Python 3.10 environment
- PyTorch with a CUDA build compatible with the installed NVIDIA driver
- TorchVision
- `multilingual-clip`
- Hugging Face `transformers` and tokenizer dependencies (`sentencepiece`)
- official OpenAI CLIP package
- Pillow
- NumPy
- pandas or Python CSV utilities
- PyYAML
- tqdm
- `pycocoevalcap`/COCO-caption-compatible evaluator
- Java runtime required by the standard METEOR wrapper
- pytest

Current environment observation: Python 3.10.0 is present, but PyTorch, TorchVision, Transformers, `multilingual_clip`, OpenAI `clip`, Pillow, and pandas are not installed. NumPy is present. No package installation or model download occurs in this phase.

## 9. Loss functions

### 9.1 Text reconstruction cross-entropy

- **[PAPER-STATED]** Used during text-only auto-encoding (Eq. 2).
- Purpose: maximize the likelihood of the original report token sequence given its text embedding.
- Inputs: logits `[B,T,Vocab]`, shifted target IDs `[B,T]`, token mask.
- Output: one finite scalar minimized by SGD.
- Gradient recipients: decoder only; text encoder frozen.
- **[IMPLEMENTATION-ASSUMPTION]** Use `CrossEntropyLoss(ignore_index=pad_id, reduction="sum") / valid_token_count`.
- The paper’s bilingual expression is malformed. Since both scoped datasets use English target reports, **[INFERRED]** use a single English NLL rather than inventing a bilingual product.

### 9.2 Image-conditioned cross-entropy

- **[PAPER-STATED]** Used for optional paired 10%/100% training (Eq. 7).
- Purpose: maximize report likelihood conditioned on `φ(I)`.
- Inputs/output/reduction: same token-level form as reconstruction CE, but logits are conditioned on image-plus-prompt embeddings.
- Gradient recipients: image encoder and decoder **[PAPER-STATED]**; prompt bank remains fixed **[IMPLEMENTATION-ASSUMPTION]**.

### 9.3 Explicit exclusions

- **[PAPER-STATED]** No additional ZeroMRG contrastive, triplet, classification, retrieval, reinforcement, clinical-label, or view-fusion loss is specified.
- Do not add losses using COV COVID labels or IU MeSH/Problems.
- CLIP’s historical pretraining objective is not rerun.

## 10. Evaluation plan

Use the standard COCO-caption-compatible evaluation family cited by the paper. Store raw `[0,1]`-like scores where applicable and display `×100` values alongside them because the paper tables appear scaled. The scaling interpretation is **[INFERRED]** and must be documented.

| Metric | Dataset/regime | What it measures | Reference | Prediction | Planned implementation | Timing |
|---|---|---|---|---|---|---|
| BLEU-1 | Both; 0/10/100% | Unigram precision with brevity penalty | One held-out target report per sample | Generated report | COCO-caption BLEU | Validation and final test |
| BLEU-2 | Both; 0/10/100% | Up-to-bigram precision | Same | Same | COCO-caption BLEU | Validation and final test |
| BLEU-3 | Both; 0/10/100% | Up-to-trigram precision | Same | Same | COCO-caption BLEU | Validation and final test |
| BLEU-4 | Both; 0/10/100% | Up-to-4-gram precision | Same | Same | COCO-caption BLEU | Every validation checkpoint; primary early-stop assumption; final test |
| ROUGE-L | Both; 0/10/100% | Longest-common-subsequence overlap/recall balance | Same | Same | COCO-caption ROUGE | Validation summaries and final test |
| CIDEr | COV-CTR; 0/10/100% | TF-IDF-weighted n-gram consensus | Same | Same | COCO-caption CIDEr | Validation summaries and final test |
| METEOR | IU-Xray; 0/10/100% | Token alignment with precision/recall and matching rules | Same | Same | COCO-caption METEOR with pinned Java dependency | Validation summaries and final test |

- **[PAPER-STATED]** These dataset-specific table metrics match Tables 1 and 3.
- **[IMPLEMENTATION-ASSUMPTION]** Use the evaluator’s standard PTB tokenization for both predictions and references; retain pre-tokenized raw strings for audit.
- Test metrics are computed once after model/checkpoint selection, never used for early stopping.
- Each output JSONL contains sample ID, image IDs, prediction, reference, termination reason, generated length, regime, and checkpoint hash.
- Do not add unrelated clinical/factuality metrics at this stage.

## 11. Hardware feasibility

### 11.1 Observed development machine

| Resource | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM | 4,096 MiB total; 3,964 MiB free at inspection |
| Compute capability | 8.6 |
| CPU | AMD Ryzen 5 5600H, 12 logical processors |
| System RAM | 7.35 GiB total; 0.69 GiB free at inspection |
| Workspace drive free | 59.19 GiB |
| Raw archive size | ~14.22 GB total on disk, dominated by IU-Xray |
| Uncompressed image payload | ~14.28 GB total |

The very low currently free system RAM must be corrected by closing other applications before any model load. Running into swap will make checkpoint loading and data workers unreliable.

### 11.2 Original paper configuration

- **[PAPER-STATED]** Batch size 6, SGD, LR `1e-3`, momentum 0.99, Multilingual-CLIP XLM-R Large/ViT-L/14, Transformer decoder.
- `MISSING FROM PAPER`: GPU model/count, VRAM, RAM, storage, precision, gradient accumulation, activation checkpointing, and training time.
- Full paired training requires gradients and optimizer momentum for the large ViT-L/14 visual tower plus decoder. A 4 GiB GPU is not expected to hold parameters, gradients, momentum state, and activations at batch 6.
- A practical full-configuration target is at least 16 GiB VRAM and preferably 24 GiB, with at least 32 GiB system RAM. **[INFERRED]** This is a feasibility recommendation, not a paper-reported requirement.

### 11.3 Resource-constrained reproduction configuration

- **[RESOURCE-CONSTRAINED]** Stage 1: run the frozen M-CLIP text encoder in evaluation/no-grad mode with micro-batches, cache normalized CPU embeddings, unload it, then train only the decoder from the cache. With deterministic frozen encoding, this preserves the mathematical Stage-1 inputs.
- **[RESOURCE-CONSTRAINED]** Decoder training: use automatic mixed precision, micro-batch 1, gradient accumulation 6 to preserve effective batch size 6, and zero/one data worker because RAM is limited.
- **[RESOURCE-CONSTRAINED]** Prompt bank: encode once in micro-batches, persist `[N_p,768]`, unload the text encoder before image inference.
- **[RESOURCE-CONSTRAINED]** Zero-shot inference: batch one study at a time; encode IU views sequentially if the entire view tensor does not fit; keep only the visual encoder, prompt matrix, and decoder resident as feasible.
- **[RESOURCE-CONSTRAINED]** Do not duplicate frozen model weights inside every checkpoint; store identifiers/hashes and trainable states.
- **[RESOURCE-CONSTRAINED]** Keep only best and last checkpoints per stage and dataset.
- **[RESOURCE-CONSTRAINED]** Stream images or make one processed extraction. The 59 GiB free drive can accommodate one ~14.3 GB extraction plus model/checkpoint artifacts, but multiple image copies or many full-model checkpoints can exhaust it.

Full 10%/100% paired training with the paper-stated trainable visual encoder is **not safely feasible on 4 GiB VRAM** without aggressive offload or changing the trainable scope. Freezing the visual encoder would contradict Sec. 3.4 and must not be labeled faithful. The plan is therefore:

1. run data, embedding, forward, gradient, and checkpoint smoke tests locally;
2. run Stage 1 and likely zero-shot inference locally with the adaptations above;
3. run full paired Stage 4/5 on ≥16 GiB VRAM hardware, preferably 24 GiB; or explicitly label any locally frozen-encoder experiment resource-constrained and non-equivalent.

Expected bottlenecks, in order:

1. ViT-L/14 gradients/SGD momentum during paired training;
2. decoder activations for 512-token reports;
3. loading the 344M-parameter M-CLIP text encoder with only ~7.35 GiB RAM;
4. decoding large IU PNG files and ZIP random access;
5. METEOR’s Java dependency and evaluation runtime.

## 12. Exact project code structure

The following structure should be created during implementation, not in this planning phase:

```text
configs/
├── base.yaml                       # paper-stated defaults and shared schema
├── cov_ctr.yaml                    # COV population, 8:1:1, 500 prompts
├── iu_xray.yaml                    # UID population, 70:10:20, 1000 prompts
└── resource_4gb.yaml               # explicitly labeled engineering overrides

src/zeromrg/
├── __init__.py
├── data/
│   ├── archive_io.py               # read-only ZIP/member access
│   ├── schemas.py                  # typed sample/manifest records
│   ├── validate_cov_ctr.py         # counts, signatures, duplicate conflicts
│   ├── validate_iu_xray.py         # UID/projection joins, section/view checks
│   ├── build_cov_ctr_manifest.py
│   ├── build_iu_xray_manifest.py
│   ├── split.py                    # deterministic group-level allocation
│   ├── report_text.py              # whitespace/NFC/section construction
│   ├── vocabulary.py               # train-only output vocabulary
│   ├── image_transforms.py         # canonical CLIP preprocessing wrapper
│   ├── datasets.py                 # text-only and image-conditioned views
│   └── collate.py                  # token/view padding and masks
├── models/
│   ├── mclip_text.py               # frozen multilingual text encoder
│   ├── clip_vision.py              # CLIP ViT-L/14 visual tower
│   ├── normalization.py            # safe L2 normalize
│   ├── prompt_bank.py              # fixed report IDs/embeddings
│   ├── alignment.py                # Eqs. (5) and (6)
│   ├── view_pooling.py             # isolated IU assumption A09
│   ├── report_decoder.py           # causal Transformer decoder
│   └── zeromrg.py                  # stage-aware complete model
├── training/
│   ├── losses.py
│   ├── trainer.py
│   ├── stages.py
│   └── checkpoint.py
├── evaluation/
│   ├── generation.py
│   ├── metrics.py
│   └── evaluator.py
└── utils/
    ├── config.py
    ├── seed.py
    ├── logging.py
    └── provenance.py

scripts/
├── validate_data.py
├── build_manifests.py
├── cache_text_embeddings.py
├── build_prompt_bank.py
├── train_autoencoder.py
├── evaluate_zero_shot.py
├── train_paired.py
└── evaluate.py

tests/
├── test_cov_ctr_manifest.py
├── test_iu_xray_manifest.py
├── test_splits.py
├── test_report_text.py
├── test_datasets.py
├── test_collate.py
├── test_encoders.py
├── test_alignment.py
├── test_decoder.py
├── test_losses.py
├── test_checkpoint.py
├── test_generation.py
└── test_metrics.py
```

The shared model/training pipeline is reused across datasets. Only validation, manifest construction, target construction, prompt count, split ratio, and view behavior are dataset-specific.

## 13. Numbered implementation order

### Step 1 — Environment specification

- Files: `pyproject.toml` or pinned requirements, `configs/base.yaml`, environment documentation.
- Output: reproducible CPU/CUDA environment specification; no model weights yet.
- Test: import-only smoke test and CUDA device report.
- Proceed when: all pinned packages import and tests can run offline after installation.

### Step 2 — Configuration and provenance layer

- Files: `utils/config.py`, `utils/seed.py`, `utils/provenance.py`, config YAMLs.
- Output: validated resolved configuration and deterministic seed derivation.
- Test: same config/seed gives identical hashes and random sequences.
- Proceed when: paper values, assumptions, and resource overrides are printed separately.

### Step 3 — Read-only archive validators

- Files: `data/archive_io.py`, both validator modules, `scripts/validate_data.py`.
- Output: machine-readable validation report matching `dataset_analysis.md`.
- Test: counts, formats, missing references, duplicate keys/pixels, and field-null counts.
- Proceed when: raw archives remain byte-identical and all known anomalies are reproduced.

### Step 4 — COV-CTR manifest builder

- Files: `schemas.py`, `build_cov_ctr_manifest.py`.
- Output: 714 valid samples plus excluded/conflict audit entries.
- Test: one-to-one filename joins, no duplicate valid image key, no blank target.
- Proceed when: exact 714/6/26 partition is accounted for.

### Step 5 — IU-Xray manifest builder

- Files: `build_iu_xray_manifest.py`.
- Output: 3,331 strict valid study records with 1–5 mapped views plus excluded audit entries.
- Test: UID joins, filenames, view counts, section presence, four unlisted images.
- Proceed when: every valid UID has both sections and at least one image.

### Step 6 — Deterministic split manifests

- Files: `data/split.py`, dataset config files.
- Output: COV 571/71/72 and IU 2332/333/666 IDs.
- Test: disjointness, completeness, repeatability; IU image filenames occur in one split only.
- Proceed when: split and population hashes are persisted.

### Step 7 — Report normalization and target construction

- Files: `data/report_text.py`, `tests/test_report_text.py`.
- Output: COV trimmed English targets; IU impression-space-findings targets.
- Test: golden examples, no invented text, no empty target, raw text preserved, no unexpected control characters.
- Proceed when: normalization is deterministic and audited.

### Step 8 — Decoder vocabulary/tokenizer

- Files: `data/vocabulary.py`, per-dataset `vocabulary.json`.
- Output: training-only vocabulary and tokenized sequences with special tokens.
- Test: encode/decode round-trip, unknown-token behavior, padding/shift, length distribution.
- Proceed when: no validation/test text influenced vocabulary and truncation count is zero at `T_max=512`; otherwise increase `T_max` and document memory effect.

### Step 9 — Image transform wrapper

- Files: `data/image_transforms.py`.
- Output: finite RGB tensors using canonical CLIP preprocess.
- Test: COV PNG/JPEG/mislabeled/RGBA/grayscale samples and IU grayscale samples all yield identical expected shape/dtype/range contract.
- Proceed when: all format variants pass without raw writes.

### Step 10 — Dataset classes and collation

- Files: `data/datasets.py`, `data/collate.py`.
- Output: text-only batches and `[B,V,3,H,W]` image batches with masks/metadata.
- Test: load one COV and IU batch; assert shapes, split IDs, UID-level views, token shifts.
- Proceed when: no study/image leakage and RAM use is bounded.

### Step 11 — Pretrained encoder wrappers

- Files: `models/mclip_text.py`, `models/clip_vision.py`, `models/normalization.py`.
- Output: text/image embeddings with asserted common dimension.
- Test: frozen flags, eval determinism, finite `[B,768]` vectors, unit norms after normalization.
- Proceed when: checkpoint revisions/hashes and canonical preprocess are recorded.

### Step 12 — Prompt bank

- Files: `models/prompt_bank.py`, `scripts/build_prompt_bank.py`.
- Output: 500/1000 training-only prompt IDs and embeddings.
- Test: no non-train IDs, correct shapes/counts/order/checksum, deterministic rebuilding.
- Proceed when: bank is frozen and reloadable.

### Step 13 — Alignment equations

- Files: `models/alignment.py`, `models/view_pooling.py`.
- Output: attention weights, per-view `φ`, and study condition.
- Test: compare with hand-computed tiny tensors; weights sum to one; masks work; normalized outputs; gradients behave as configured.
- Proceed when: Eqs. (5)–(6) pass numerical unit tests.

### Step 14 — Report decoder

- Files: `models/report_decoder.py`.
- Output: `[B,T,Vocab]` logits under teacher forcing.
- Test: causal-mask isolation, memory conditioning changes logits, PAD behavior, all shapes, parameter count logged.
- Proceed when: a two-sample batch forward is finite.

### Step 15 — Complete model and losses

- Files: `models/zeromrg.py`, `training/losses.py`.
- Output: text-conditioned and image-conditioned forward paths with scalar CE.
- Test: finite loss; correct frozen/trainable parameters; backward gradients only to allowed modules.
- Proceed when: all stage-specific gradient assertions pass.

### Step 16 — Checkpoint and trainer

- Files: `training/trainer.py`, `training/stages.py`, `training/checkpoint.py`.
- Output: resumable best/last checkpoints and structured logs.
- Test: one optimizer step, save/reload, identical logits after reload, resume epoch/optimizer/RNG.
- Proceed when: round-trip and early-stopping tests pass.

### Step 17 — Tiny overfit/smoke experiment

- Files: training scripts/config smoke override.
- Output: two-to-eight sample run with decreasing CE and generated termination.
- Test: overfit text reconstruction first; then one paired batch; monitor VRAM/RAM.
- Proceed when: loss decreases, no NaN/OOM, and no forbidden gradients occur.

### Step 18 — Generation

- Files: `evaluation/generation.py`, `scripts/evaluate_zero_shot.py`.
- Output: deterministic generated report JSONL.
- Test: BOS/EOS, maximum-length termination, batch-vs.-single equivalence, no reference text enters generation.
- Proceed when: outputs are reproducible after checkpoint reload.

### Step 19 — Metrics

- Files: `evaluation/metrics.py`, `evaluation/evaluator.py`.
- Output: dataset-specific raw/scaled metric JSON.
- Test: identity predictions score at/near expected maxima; empty/hypothesis edge cases; stable sample ordering.
- Proceed when: BLEU orders, ROUGE-L, COV CIDEr, and IU METEOR match pinned toolkit fixtures.

### Step 20 — Full Stage-1 and zero-shot experiments

- Files: resolved run configs; no source changes unless a verified bug is found.
- Output: separate COV/IU text-only checkpoints, prompt banks, 0%-paired predictions and metrics.
- Test: all preflight gates rerun; best checkpoint chosen only on validation.
- Proceed when: provenance is complete and zero-shot inference is reproducible.

### Step 21 — 10% and 100% paired experiments

- Hardware condition: use sufficient VRAM for paper-stated trainable encoder+decoder.
- Output: separate paired checkpoints/results per dataset and ratio.
- Test: subset nesting, initialization lineage, trainable visual gradients, validation-only selection.
- Proceed when: hardware feasibility and all smoke tests pass; otherwise do not launch.

### Step 22 — Final evaluation and results packaging

- Output: `results/cov_ctr/...` and `results/iu_xray/...` with predictions, raw/scaled metrics, configs, hashes, and paper-vs-reproduction count caveats.
- Test: test set used only after selection; artifact schema validation; no fabricated paper numbers in result files.
- Completion condition: all requested regimes that hardware permits are reproducible from recorded commands/configs.

## 14. Sanity-check strategy and mandatory gates

Full training is prohibited until every applicable check below passes.

| Area | Test | Acceptance condition |
|---|---|---|
| Dataset validation | Recompute archive counts and anomalies | Exactly matches `dataset_analysis.md` |
| COV pairing | Join all valid keys | 714 valid; 6 conflict keys; 26 unannotated images; zero missing referenced files |
| IU pairing | Join reports→projections→files | 3,331 strict UIDs; every included UID has 1–5 images; zero missing listed files |
| Split leakage | Intersections across splits | Empty ID intersections; IU image filenames confined to parent UID split |
| Prompt leakage | Prompt IDs vs splits | Exact count; all prompts in train; none in validation/test |
| Image decoding | Exercise every signature/channel class | Finite RGB tensor with canonical CLIP shape |
| Text normalization | Golden raw/processed pairs | Only configured whitespace/NFC/section join changes occur |
| Tokenization | Encode/decode and length audit | Special-token contract holds; no primary target truncation |
| Collation | Mixed lengths/views | Correct token/view masks; padded values never contribute |
| Encoder dimensions | Text/image smoke inputs | Both output finite dimension 768 or execution stops with explicit incompatibility |
| Normalization | Norm checks | Norm approximately 1 (`atol=1e-5`) for nonzero vectors |
| Prompt attention | Tiny manual case | Shape `[B,V,Np]`; last-axis sums approximately 1; invalid views masked |
| View pooling | Known vectors/masks | Equals manual masked mean then normalization |
| Decoder causality | Perturb future target tokens | Earlier logits unchanged |
| Forward pass | One batch per dataset/path | Logits `[B,T,Vocab]`, no NaN/Inf |
| Loss | Manual small logits and real batch | Matches manual CE; PAD excluded; finite scalar |
| Stage-1 gradients | Backward one batch | Decoder grads finite; text encoder has no grads |
| Zero-shot gradients | Inference | No parameter grads and graph disabled |
| Paired gradients | Backward one batch | Visual encoder and decoder grads finite; prompt bank/text encoder no grads |
| Optimizer | One step | Parameters change only in allowed modules |
| Checkpoint | Save/reload/resume | Same eval logits/generation; optimizer/RNG/epoch restored |
| Tiny overfit | 2–8 reports | Reconstruction loss decreases substantially without NaN/OOM |
| Generation | Known checkpoint/batch | Starts BOS, stops EOS or max, no PAD in detokenized output |
| Metric fixtures | Identical and deliberately different strings | Expected ordering/max behavior; deterministic values |
| Resource monitoring | Peak VRAM/RAM/disk | Within configured budget with safety margin; otherwise stop and document override |

Any failed gate blocks progression. A resource failure permits only an explicitly labeled **[RESOURCE-CONSTRAINED]** configuration; it does not authorize an undocumented architecture change.

## 15. Final implementation checklist

- [ ] Environment — pinned Python/PyTorch/CUDA and metric dependencies; import smoke test
- [ ] Dataset validation — raw archive counts, keys, duplicates, and hashes reproduced
- [ ] COV-CTR preprocessing — 714-primary manifest, exclusions, split, normalized English targets
- [ ] IU-Xray preprocessing — 3,331-primary UID manifest, all views, impression+findings targets
- [ ] Dataset loaders — text-only and image/study modes with token/view masks
- [ ] Model components — M-CLIP text, CLIP vision, prompt bank, Eqs. (5)–(6), view adapter, decoder, wrapper
- [ ] Loss functions — reconstruction CE and paired image-conditioned CE only
- [ ] Training pipeline — Stage 1 plus independent 10%/100% paired stages, checkpoint/resume/logging
- [ ] Sanity test — every mandatory gate in Section 14 passes
- [ ] COV-CTR experiment — separate 0%/10%/100% artifacts under `results/cov_ctr/`
- [ ] IU-Xray experiment — separate 0%/10%/100% artifacts under `results/iu_xray/`
- [ ] Inference — frozen prompt-enriched image-to-report generation with EOS/max stop
- [ ] Evaluation — COV BLEU-1..4/CIDEr/ROUGE-L; IU BLEU-1..4/METEOR/ROUGE-L
- [ ] Results — predictions, metrics, configurations, hashes, hardware, date, and count/version caveats

No checklist item is completed by this planning document. Implementation, preprocessing, checkpoint download, and training begin only in later explicitly authorized phases.

## 16. Resource-constrained IU-Xray-500 execution profile

**[RESOURCE-CONSTRAINED]** This operational profile supplements, and does not replace or revise, the canonical full-IU reproduction plan above. COV-CTR remains the complete 714-sample validated population with its 571/71/72 split, 57 paired-training IDs, and 500 prompt IDs. The canonical IU-Xray population remains 3,331 strict study UIDs; its existing configuration and derived artifacts must not be overwritten.

For resource-limited Kaggle execution, `configs/iu_xray_kaggle_500.yaml` defines a deterministic selection of 500 UIDs from the validated 3,331-UID population using base seed 42 and a dedicated versioned seed namespace. Selection occurs before splitting and preserves all projection rows and physical image views belonging to every selected UID. The isolated profile split is exactly 350 train, 50 validation, and 100 test. Its 35 paired-training IDs and 250 prompt IDs are drawn only from the 350 training UIDs. Its decoder vocabulary and report-length statistics are built independently from the subset, with vocabulary construction restricted to subset training reports.

Subset selection provenance is stored at `data/processed/iu_xray/subset_500_ids.json`; all other subset-derived data artifacts are stored below `data/processed/iu_xray/kaggle_500/`. Future checkpoints, predictions, metrics, and logs for this profile must use the same `iu_xray/kaggle_500` namespace so that neither full-IU nor COV artifacts can be mixed with subset runs.

The portable `IU-Xray-500` package preserves the original IU-compatible filenames and layout, report field values, and selected PNG bytes. It includes filtered report/projection tables, every selected view, selection provenance, a package summary, and an explicit resource-constrained README. Package acceptance requires independent validation of exactly 500 valid UIDs, exact metadata/image filtering, no missing/unmapped/corrupt images, unchanged image hashes, correct view grouping, and the recorded 350/50/100, 35, and 250 profile counts. A deterministic ZIP may be used for Kaggle upload.

Results from this profile are not directly equivalent to full-IU paper results. Every table, checkpoint, or metric produced from it must carry the `RESOURCE-CONSTRAINED` label and the subset artifact hash. No architectural, loss, preprocessing, or evaluation-method change is implied by the smaller execution population.
