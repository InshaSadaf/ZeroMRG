# ZeroMRG COV-CTR Kaggle Implementation: Methodology and Progress

> **Gate 3 preparation, 2026-10-11 IST:** a separate recovery workflow preserves the original85/56.44423922571333 and5/50 stopping evidence, initializes a recoverable selection window by freshly measuring physical90 weights on validation71, and retains original stopping for subsequent91+ continuation. Historical source copies are hash-verified/read-only; original Gate 1/2 notebooks and reports remain unchanged. New notebook training is disabled by default. Local synthetic recovery tests and static compilation pass; **no Gate 3 Kaggle run, new baseline measurement, training, image generation, final test scoring or remote publication has occurred**. Conditional readiness is for a separately approved baseline-only run; training/inference and durable new-checkpoint retrieval remain pending. See [recovery decision](zeromrg_gate3_recovery_decision.md), [protocol](zeromrg_gate3_training_protocol.md), [dependency audit](zeromrg_gate3_dependencies.md) and [persistence](zeromrg_gate3_persistence.md). Earlier Gate 1/2 selected-best blockers remain historical facts, not erased by this recovery policy.

> **Gate 2 result, 2026-10-11 IST:** the explicitly authorized private bounded Kaggle preflight ran and its report/log were retrieved. Mounted dataset/dependency assertions, epoch90 decoder/SGD restoration (112/112 momentum buffers), finite single-training-report forward and disposable save/restore all passed on Tesla T4. The intended best-selection guard found epoch8 instead of history-selected85 and raised after saving a BLOCKED report. **Overall Gate 2 remains BLOCKED.** Historical best56.44423922571333/patience5 remain unchanged. No training, image generation, final test scoring or checkpoint overwriting occurred. See [Gate 2 execution record](zeromrg_gate2_preflight.md). No further remote execution is authorized.

> **Gate 1 update, 2026-10-10: BLOCKED.** Epoch90 latest model/SGD/history is verified, with 112/112 intact momentum buffers. History selects epoch85 (text-reconstruction BLEU4 56.44423922571333), but both the local and independently downloaded remote `text_decoder_best.pt` contain **epoch8** weights. Epoch85 weights have not been recovered. New checkpoint-driven continuation and separate preflight notebooks are statically verified; 51 CPU tests passed. No Kaggle execution/training or final test evaluation has been performed. Current scope is COV-CTR 0% only. See [Gate 1 readiness](zeromrg_gate1_readiness.md) and [recovery evidence](zeromrg_checkpoint_recovery.md). Earlier runtime details below are historical logs, not verification of current mounts or exact encoder revisions.

## 1. Purpose and scope

This document records the COV-CTR English-report implementation of ZeroMRG completed so far in Kaggle. It describes the exact dataset audit, preprocessing, deterministic partitions, vocabulary, pretrained encoders, prompt bank, report decoder, text-autoencoder training, checkpoint recovery, results, assumptions, and remaining work.

The implementation is a paper-guided reproduction rather than an exact reproduction. The paper reports 728 COV-CTR images, whereas the available archive produces 714 unambiguous image-report pairs. The paper also omits the decoder architecture, tokenizer, random seed and split membership, decoding strategy, exact package versions, and several preprocessing choices.

No final image-to-report result has been produced yet. The current BLEU-4 values measure validation-set text reconstruction by the text autoencoder.

## 2. Execution environment

- Platform: Kaggle Notebook
- Accelerator used during the recorded run: NVIDIA Tesla T4
- GPU memory reported by Kaggle: 14.56 GiB
- PyTorch: 2.10.0+cu128
- pandas: 2.3.3
- NumPy: 2.0.2
- Pillow: 11.3.0
- Primary device: `cuda:0`
- Writable artifact directory: `/kaggle/working/zeromrg_covctr_preprocessed`
- Random seed: 42
- Prompt-selection seed: 42

The raw dataset is attached read-only below `/kaggle/input`. All derived artifacts and checkpoints are written below `/kaggle/working` because Kaggle input datasets are immutable.

## 3. Raw dataset discovery

The attached dataset root used in Kaggle was:

```text
/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports
```

The report table was discovered at:

```text
/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports/COV-CTR/reports_ZH_EN.csv
```

The implementation recursively searched the dataset tree for the report CSV and supported image extensions:

```text
.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
```

The discovered raw population was:

| Quantity | Count |
|---|---:|
| CSV rows | 726 |
| Physical image files | 746 |
| Duplicate image identifiers | 6 |
| Valid unambiguous image-report pairs | 714 |

The six duplicate identifiers account for 12 CSV rows. Excluding those duplicate groups yields 714 unambiguous rows. The archive also contains physical images that are not selected as valid annotated pairs. The paper reports 728 images, so the local 714-pair population is not identical to the paper population.

## 4. Dataset validation and preprocessing

### 4.1 Required columns

The English COV-CTR pipeline uses:

```text
image_id
reports_En
```

`image_id` is treated as a string and stripped of surrounding whitespace. `reports_En` is the target English medical report.

### 4.2 Report normalization

Each report was normalized by:

1. Converting missing values to an empty string.
2. Applying Unicode NFC normalization.
3. Collapsing repeated whitespace to a single space.
4. Stripping leading and trailing whitespace.
5. Preserving the original letter case (`LOWERCASE_REPORTS=False`).

No medical terms were rewritten, corrected, translated, expanded, or removed. No report was truncated.

### 4.3 Image matching

Physical images were indexed by lowercase filename. Each normalized `image_id` was matched to an image by exact case-insensitive filename equality.

A sample was excluded when:

- its image identifier occurred in more than one CSV row;
- its normalized report was empty;
- no physical image matched the identifier;
- more than one physical image ambiguously matched the identifier; or
- Pillow could not decode and verify the image.

The raw dataset was never modified. Validation only read files and wrote derived manifests below `/kaggle/working`.

### 4.4 Validation result

The executed validation produced:

```text
CSV rows:                726
Physical images:         746
Valid image-report pairs:714
Excluded duplicate IDs:    6
```

The validated manifest contains one row per usable pair with:

```text
image_id
image_path
report
```

It was saved as `validated_manifest.csv` together with `validation_audit.json`.

## 5. Deterministic partitions and sampling

### 5.1 Train, validation, and test split

The 714 validated samples were shuffled with NumPy's `default_rng` using seed 42. An 80:10:10 allocation was applied:

| Partition | Count |
|---|---:|
| Train | 571 |
| Validation | 71 |
| Test | 72 |
| Total | 714 |

The code asserted that the three image-ID sets were mutually disjoint and that their union covered the complete validated population.

The seed and exact memberships are implementation choices because the paper does not disclose its split seed or sample membership. All memberships were persisted so future stages reuse the same split.

### 5.2 Prompt selection

Exactly 500 prompt reports were sampled without replacement from the 571 training samples using seed 42. Assertions verified that every prompt belonged to the training set and that no prompt came from validation or test.

The resulting prompt population was saved as `prompt_manifest.csv`.

### 5.3 Ten-percent paired subset

The 10% paired condition contains:

```text
round(0.10 * 571) = 57 samples
```

These 57 samples were selected without replacement from the training partition using seed 42 and saved as `paired_10_manifest.csv`. No validation or test sample is included.

The prompt set and 10% paired set are separately sampled training subsets and are allowed to overlap.

## 6. Decoder tokenization and vocabulary

### 6.1 Tokenization

The output reports use a deterministic word-and-punctuation tokenizer:

```regex
\w+|[^\w\s]
```

This keeps word-like sequences as tokens and separates punctuation into individual tokens.

### 6.2 Special tokens

Four special tokens occupy fixed IDs:

| Token | ID | Purpose |
|---|---:|---|
| `<pad>` | 0 | Batch padding and ignored loss positions |
| `<bos>` | 1 | Beginning of report |
| `<eos>` | 2 | End of report |
| `<unk>` | 3 | Token absent from the training vocabulary |

### 6.3 Vocabulary construction

The vocabulary was built exclusively from the 571 training reports. Validation and test text did not influence it. All unique training tokens were sorted after the four fixed special tokens.

The resulting vocabulary size was 368.

### 6.4 Sequence-length audit

Lengths include the beginning and end tokens:

| Statistic | Tokens |
|---|---:|
| Minimum | 21 |
| Mean | 78.28 |
| 95th percentile | 106 |
| Maximum | 129 |
| Configured maximum | 160 |

No report required truncation. Encoding/decoding was checked on a real training report and reproduced the normalized report text, including punctuation spacing.

The vocabulary and length statistics were saved as `decoder_vocabulary.json` and `report_length_statistics.json`.

## 7. Pretrained representation models

### 7.1 M-CLIP text encoder

The report encoder is:

```text
M-CLIP/XLM-Roberta-Large-Vit-L-14
```

The frozen encoder maps a report to a 768-dimensional vector. The package's default forward method left tokenizer tensors on CPU while model weights were on CUDA, causing a device mismatch. The implementation corrected this by explicitly moving every tokenizer tensor to `cuda:0` before executing the XLM-R transformer.

The text representation follows the M-CLIP implementation:

1. Tokenize the report with the pretrained tokenizer.
2. Run XLM-R to obtain token hidden states.
3. Mean-pool hidden states using the attention mask.
4. Apply M-CLIP's `LinearTransformation` projection.
5. L2-normalize the projected vector for text-autoencoder conditioning.

A probe produced:

```text
Text embedding shape: (1, 768)
Text embedding norm:  1.0
Device:               cuda:0
```

### 7.2 Cached text features

M-CLIP is frozen, so repeatedly encoding the same reports during decoder training would waste memory and compute. Features were encoded once in micro-batches of two and cached on CPU:

| Cache | Count | Representation |
|---|---:|---|
| Training reports | 571 | L2-normalized 768-D vectors |
| Validation reports | 71 | L2-normalized 768-D vectors |

Test text embeddings were not required for training and were not used to condition test image generation.

The cache was saved as `mclip_text_features.pt`.

### 7.3 Prompt bank

The 500 training-only prompt reports were encoded using the same M-CLIP text encoder. The prompt bank has shape:

```text
(500, 768)
```

Raw projected M-CLIP vectors, rather than individually L2-normalized prompt vectors, were retained for prompt attention. The paper does not explicitly specify prompt normalization, so this is a documented implementation assumption.

The prompt bank and ordered prompt IDs were saved as `mclip_prompt_bank.pt`.

### 7.4 OpenAI CLIP image encoder

The vision encoder is OpenAI CLIP `ViT-L/14`. Only its visual tower is retained. It produces a 768-dimensional image representation compatible with M-CLIP's projected text dimension.

OpenAI CLIP loaded its CUDA convolution weights in `float16`. The image-encoding wrapper therefore casts image tensors to the exact device and dtype of `vision_model.conv1.weight`, then converts the resulting embeddings to `float32` for prompt attention.

A probe produced:

```text
Vision weight dtype: torch.float16
Image output shape:  (1, 768)
Image output dtype:  torch.float32
Image values finite: True
```

The vision model was moved back to CPU during text-decoder training because images are not used in that stage. Actual dataset images have not yet been evaluated through the completed decoder.

## 8. Report decoder

The paper only calls the decoder transformer-based; it does not disclose its internal design. The implemented decoder is therefore an explicit architecture assumption.

### 8.1 Architecture

| Component | Configuration |
|---|---|
| Token embedding | Vocabulary size 368, hidden size 768 |
| Position embedding | Learned, 160 positions |
| Transformer type | PyTorch Transformer decoder |
| Decoder layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 3072 |
| Activation | GELU |
| Dropout | 0.1 |
| Conditioning memory | One 768-D embedding token |
| Output projection | 768 to 368 vocabulary logits |
| Generation | Greedy autoregressive decoding |
| Stop condition | `<eos>` or 160-token limit |

The total trainable parameter count is 57,399,152.

### 8.2 Conditioning

During text autoencoding, a normalized M-CLIP report vector is reshaped from `[B,768]` to a single cross-attention memory token `[B,1,768]`. The causal decoder reconstructs the report using that memory and the preceding ground-truth tokens.

### 8.3 Teacher forcing and loss

An encoded report has the form:

```text
<bos> token_1 token_2 ... token_n <eos>
```

The decoder input is every token except the final token, and the prediction target is every token except the initial token. This one-position shift implements teacher forcing.

Training minimizes token-level cross-entropy. Padding positions are ignored, valid-token losses are summed, and the sum is divided by the number of non-padding targets. Gradients are clipped to a maximum norm of 5.0.

A real six-report sanity batch produced:

```text
Condition shape:  (6, 768)
Token shape:      (6, 108)
Initial loss:     6.2932
Finite gradients: True
GPU allocation:  approximately 0.8 GiB
```

## 9. Optimization and model selection

The text-autoencoder optimization configuration is:

| Setting | Value |
|---|---:|
| Batch size | 6 |
| Optimizer | SGD |
| Learning rate | 0.001 |
| Momentum | 0.99 |
| Gradient clipping | 5.0 |
| Maximum epochs | 200, implementation cap |
| Early-stopping metric | Validation BLEU-4 |
| Early-stopping patience | 50 consecutive non-improving epochs |

The paper states an SGD learning rate of 0.001, momentum of 0.99, batch size of 6, dropout, and 50-epoch BLEU early stopping. It does not clearly state which BLEU order or the maximum number of epochs. BLEU-4 and a 200-epoch maximum are documented assumptions.

Validation uses greedy generation on all 71 validation reports. Reports are tokenized with the COCO PTB tokenizer and scored using `pycocoevalcap` BLEU-4. The test set is not used for checkpoint selection.

## 10. Text-autoencoder training results

The externally preserved run contains 90 completed epochs.

| Epoch | Training loss | Validation BLEU-4 | Interpretation |
|---:|---:|---:|---|
| 1 | 3.0513 | 45.5640 | Initial learned reconstruction |
| 2 | 1.3539 | 46.9752 | Rapid early improvement |
| 8 | approximately 0.67 | 52.3389 | First strong checkpoint |
| 30 | 0.1926 | 49.0544 | Lower training loss but weaker validation |
| 35 | approximately 0.16 | 55.0886 | New best checkpoint |
| 45 | 0.1329 | 39.3920 | Large validation fluctuation |
| 60 | 0.1140 | 46.3690 | Continued overfitting trend |
| 75 | 0.1057 | 54.9767 | Close to the previous best |
| 85 | approximately 0.10 | **56.4442** | Best history entry; matching weights missing |
| 90 | 0.1005 | 54.2114 | Latest externally preserved epoch |

At epoch 90:

```text
Best epoch:             85
Best validation BLEU-4: 56.4442
Patience state:          5/50
```

The continually decreasing training loss and volatile validation BLEU show that the decoder increasingly memorizes training reports while small parameter changes can substantially change greedy validation generations. The best validation checkpoint, rather than the latest checkpoint, must be used for downstream evaluation.

The value 56.4442 is not comparable to the paper's 0%-paired test BLEU-4. It is a validation score for reconstructing reports from their own text embeddings. The paper's final result is produced from image embeddings on a held-out test set.

## 11. Checkpoint design

### 11.1 Latest checkpoint

`text_decoder_last.pt` is approximately 438 MB and contains:

- latest completed epoch;
- decoder parameters;
- SGD optimizer state and momentum;
- best validation BLEU-4 observed so far;
- number of epochs without improvement; and
- complete training history.

It is used only to resume training.

### 11.2 Best checkpoint

The portable `text_decoder_best.pt` is approximately 219 MB and is intended to contain the validation-selected decoder with its epoch and metadata. Direct Gate 1 inspection instead found epoch8 and BLEU4 52.33887482243904, not history-selected epoch85. The remote file has the same SHA-256 as this stale local file. It has no optimizer state. Do not use it as the selected epoch85 decoder.

### 11.3 Portable backup

The portable backup also contains manifests, vocabulary, cached text embeddings, prompt embeddings, preprocessing summaries, and training history. The newest verified local backup is `(5)`, containing resumable epoch90 and a stale epoch8 best file. Its history records epoch85 as the best; the corresponding weights are absent from the examined backups. Exact hashes and bounded search coverage are in the Gate 1 recovery record.

The backup is attached to Kaggle read-only at:

```text
/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip
```

Historically files were copied into `/kaggle/working/zeromrg_covctr_preprocessed`. New Gate 1 notebooks read dependencies from the selected immutable input and use a unique output directory for diagnostics and any separately approved future continuation.

## 12. Session failure and recovery

Historical logs show a continuation through epoch120 and a Kaggle platform error with status code70. No matching later persistent checkpoint was found in the bounded accessible evidence; weights cannot be recovered from those logs. The canonical latest recovery point is epoch90.

The original checkpoint did not preserve RNG states. Consequently, restarted epoch91 cannot reproduce the lost stochastic trajectory bit for bit. New continuation code records a seeded legacy restart, explicit per-epoch data-loader seed and future Python/Torch CPU/CUDA RNG states. This improves auditability but does not guarantee bitwise replay across hardware, kernels or package versions. Exact historical encoder revisions are also unverified.

This recovery change does not alter the model architecture, dataset, objective, optimizer, batch size, or validation rule, but it must be reported as a reproducibility correction.

Default continuation currently refuses the epoch85-versus-file8 mismatch. After recovery and explicit execution approval, at most five epochs per session chunk and verified external backups are proposed to reduce future loss:

```text
90 -> 95 -> backup
95 -> 100 -> backup
100 -> 105 -> backup
```

## 13. Leakage controls

The implementation currently enforces the following controls:

- validation and test reports do not enter vocabulary construction;
- prompt reports come only from training;
- the 10% paired subset comes only from training;
- split image IDs are mutually disjoint;
- validation BLEU, not test BLEU, selects the best checkpoint;
- test text embeddings are not used as conditions during text-autoencoder training; and
- raw input data remains read-only.

## 14. Work not yet completed

The following stages remain:

### 14.1 Finish text-autoencoder training

First recover matching selected-best weights or obtain explicit approval for the separately documented incomplete-selection recovery policy. Default continuation is blocked today. After scientific and remote execution approval, resume from checkpoint-derived epoch91 and continue until either:

- validation BLEU-4 fails to improve for 50 consecutive epochs; or
- the 200-epoch implementation cap is reached.

Then load the genuinely validated selected-best artifact. Neither available epoch8 nor latest90 may substitute for missing epoch85. A five-epoch chunk is a persistence boundary, not a scientific stopping rule.

### 14.2 Zero-paired image-to-report evaluation

For every held-out test image:

1. Apply OpenAI CLIP's canonical image preprocessing.
2. Encode the image with the frozen ViT-L/14 visual tower.
3. Attend from the image vector to the 500 frozen prompt vectors using scaled dot-product attention.
4. Add the attended prompt vector to the base image vector.
5. L2-normalize the sum.
6. Feed the result to the selected text decoder.
7. Generate greedily until `<eos>` or the maximum length.
8. Save every reference and generated report.
9. Compute held-out BLEU-1 through BLEU-4, CIDEr, and ROUGE-L.

This produces the actual 0%-paired result.

### 14.3 Other experiments excluded

Ten-percent paired, one-hundred-percent paired and IU-Xray experiments are outside the current authorized scope. Existing subset artifacts are historical dependencies, not instructions to run those experiments.

### 14.5 Final result package

The final package should contain:

- preprocessing audit;
- exact split, prompt, and paired-subset manifests;
- vocabulary and length statistics;
- resolved model/training configuration;
- text-autoencoder history and selected checkpoint;
- only the authorized 0%-paired predictions;
- BLEU-1 through BLEU-4, CIDEr, and ROUGE-L metrics;
- hardware and package versions;
- dataset-count discrepancy and implementation assumptions; and
- comparison with the published paper values clearly labeled as reference values.

## 15. Scientific interpretation and limitations

The implementation should be described as a paper-guided COV-CTR reproduction with fixed audited splits and explicit recovery limitations. Exact reproduction is not claimable because:

- the available archive yields 714 pairs instead of the paper's 728 images;
- the paper does not disclose split membership or a random seed;
- the decoder architecture and tokenizer are unspecified;
- prompt normalization is unspecified;
- generation search settings are unspecified;
- exact pretrained checkpoint revisions and package versions are unspecified;
- selected epoch85 weights and original RNG state are not recovered; and
- metric preprocessing/toolkit details are incomplete.

Published values must remain reference values. Only metrics calculated from saved held-out predictions should be reported as results of this implementation.
