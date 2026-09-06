# Phase 1 Paper Analysis: ZeroMRG

## 1. Source and analysis convention

**Paper:** Yuena Jiang and Yanxun Chang, “Auto-encoding clinical language for zero-shot medical report generation,” *Engineering Applications of Artificial Intelligence* 164 (2026), 113244. DOI: 10.1016/j.engappai.2025.113244.

This document analyzes all nine pages of the paper, including Fig. 1, equations, experimental tables, qualitative discussion, limitations, and references. It is implementation-oriented, but it does not choose undocumented implementation details or implement the model.

Every material statement is marked with one of the requested evidence classes:

- **PAPER-STATED** — explicitly stated in prose, an equation, a figure, or a table.
- **INFERRED** — a likely interpretation needed to turn the paper into code, but not explicitly specified.
- **MISSING/AMBIGUOUS** — absent, internally inconsistent, malformed, or not precise enough to reproduce faithfully.

Page and section references below refer to the paper PDF.

## 2. Research problem and claimed contribution

- **PAPER-STATED:** The task maps a medical image \(I\) to a report \(S=\{s_1,\ldots,s_T\}\), where the report contains multiple sentences describing the patient's condition (Sec. 3.1).
- **PAPER-STATED:** ZeroMRG targets zero-shot medical report generation without downstream labeled image-report pairs. It is intended to support multiple image modalities (CT and chest X-ray) and multiple languages (English and Chinese) in one framework (Abstract; Secs. 1 and 3).
- **PAPER-STATED:** The central strategy is to train report generation as text-only auto-encoding, then replace the text embedding at inference with a visual embedding from Multilingual-CLIP enriched by report prompts (Secs. 3.2–3.3; Fig. 1).
- **PAPER-STATED:** The paper also evaluates optional 10%-paired and 100%-paired regimes in addition to the zero-shot 0%-paired regime (Tables 1–4).
- **PAPER-STATED:** The intended low-resource claim concerns absence of *paired* downstream image-report training data, not absence of downstream-domain text. Text reports are required to train the auto-encoder and to form prompts.
- **INFERRED:** “Zero-shot” should therefore be implemented and reported as zero paired-image supervision, while clearly accounting for the unpaired reports used by both the decoder and prompt bank.
- **MISSING/AMBIGUOUS:** The paper repeatedly says “without any labeled data” even though it uses medical reports from the downstream datasets. It does not formally define what data are permissible in the zero-shot setting or provide a leakage-control protocol.

## 3. Complete architecture

### 3.1 Modules

| Component | Role | Evidence classification |
|---|---|---|
| Multilingual text encoder \(E_m\) | Encodes English or Chinese reports into the Multilingual-CLIP shared embedding space. | **PAPER-STATED:** Named in Sec. 3.1 and Fig. 1. |
| Vision encoder \(E_v\) | Encodes CT or X-ray images using Multilingual-CLIP and report prompts. | **PAPER-STATED:** Sec. 3.1, Sec. 3.3, and Fig. 1. |
| Multilingual decoder \(D_m\) | Autoregressively generates/reconstructs English or Chinese report tokens conditioned on an encoder embedding. | **PAPER-STATED:** Named in Secs. 3.1–3.3; conditional token probabilities appear in Eqs. (2) and (7). **INFERRED:** “Autoregressive” follows from conditioning each token on the preceding ground-truth prefix. |
| Prompt bank \(f_p\) | A randomly selected set of medical reports represented as prompt embeddings; supplies domain/report knowledge during image inference. | **PAPER-STATED:** Sec. 3.3 and Eq. (6). |
| Prompt-to-image attention | Computes scaled dot-product attention from image features to prompt embeddings, yielding prompt-based visual features \(f_p(I)\). | **PAPER-STATED:** Eq. (6). |
| L2 normalization | Normalizes text features during auto-encoding and the sum of image and attended-prompt features during inference. | **PAPER-STATED:** Eqs. (3) and (5). |
| Optional supervised path | Fine-tunes an image encoder and decoder on paired image-report data after auto-encoding. | **PAPER-STATED:** Sec. 3.4 and Tables 1–4. |

### 3.2 End-to-end data flow

#### Stage A: text-only auto-encoding

1. **PAPER-STATED:** Input is an English report \(S_{en}\) or Chinese report \(S_{cn}\), with no image required (Sec. 3.2; Fig. 1).
2. **PAPER-STATED:** Multilingual-CLIP extracts a textual feature \(\psi(S)\).
3. **PAPER-STATED:** The feature is L2-normalized:

   \[
   \bar\psi(S)=\frac{\psi(S)}{\lVert\psi(S)\rVert_2}.
   \]

   The paper prints the normalized output with the same symbol \(\psi(S)\); the bar is introduced here only to make the operation readable.
4. **PAPER-STATED:** The multilingual decoder receives the text embedding and reconstructs the original report (Sec. 3.2; Eq. (4); Fig. 1).
5. **PAPER-STATED:** Training minimizes token-level cross-entropy (Eq. (2)).
6. **PAPER-STATED:** Fig. 1 shows a snowflake on the multilingual encoder but not the decoder.
7. **INFERRED:** The Multilingual-CLIP text encoder is frozen and the decoder is trained. This is the only interpretation consistent with Fig. 1 and the claim that the decoder learns report generation from pretrained text embeddings.
8. **INFERRED:** Decoder inputs are shifted report tokens (teacher forcing), because Eq. (2) predicts \(s_t\) conditioned on \(s_{1:t-1}\).

#### Stage B: zero-shot image-to-report inference

1. **PAPER-STATED:** Input is a medical image \(I\), illustrated as COV-CTR CT, IU-Xray, or COV-CT CT (Sec. 3.3; Fig. 1).
2. **PAPER-STATED:** Multilingual-CLIP produces image features \(f_M(I)\).
3. **PAPER-STATED:** A set of \(n\) medical reports is randomly selected as prompts, represented by prompt embeddings \(f_p\).
4. **PAPER-STATED:** Scaled dot-product attention computes prompt-conditioned features:

   \[
   f_p(I)=\operatorname{softmax}\!\left(\frac{f_M(I)f_p^\top}{\sqrt d}\right)f_p.
   \]

5. **PAPER-STATED:** The base image and prompt-conditioned features are added and L2-normalized:

   \[
   \varphi(I)=\frac{f_M(I)+f_p(I)}{\lVert f_M(I)+f_p(I)\rVert_2}.
   \]

6. **PAPER-STATED:** The previously trained \(D_m\) consumes \(\varphi(I)\) and generates the report.
7. **PAPER-STATED:** Generation stops when a special `END` token is emitted or a predefined maximum report/sentence length is reached (Sec. 4.3).
8. **PAPER-STATED:** Multilingual-CLIP, the prompts, and—according to the snowflake in Fig. 1—the decoder are frozen during zero-shot inference.

#### Stage C: optional paired-data supervised training

1. **PAPER-STATED:** Fully supervised training is conducted “on the basis of auto-encoding” (Sec. 3.4).
2. **PAPER-STATED:** The paper says both encoder \(E_v\) and decoder \(D_m\) are trained using labeled image-report pairs.
3. **PAPER-STATED:** The supervised objective is conditional token cross-entropy using \(\varphi(I)\) (Eq. (7)).
4. **PAPER-STATED:** Experiments use 10% and 100% paired-data conditions in addition to 0% (Tables 1–4).
5. **MISSING/AMBIGUOUS:** It is not specified whether supervised training starts from the text-only decoder checkpoint in all experiments, which exact parts of the Multilingual-CLIP encoder are unfrozen, or whether the prompt bank itself is updated.

### 3.3 Architecture inconsistencies that must not be silently repaired

- **MISSING/AMBIGUOUS:** Sec. 3.1 introduces \(E_m\) as the multilingual encoder and \(E_v\) as the vision encoder, but Sec. 3.2 calls the report encoder “\(E_v\)” and Eq. (4) applies \(E_v\) to text. Fig. 1 supports \(E_m\) for text and \(E_v\) for images.
- **MISSING/AMBIGUOUS:** Sec. 3.1 twice calls the proposed model “ZeroNLG,” evidently inherited from the related ZeroNLG work, while the remainder calls it ZeroMRG.
- **MISSING/AMBIGUOUS:** Eq. (1) gives \(i,j=1,2,3,4,\ldots\) without using \(i\) meaningfully and does not define how an output language is selected.
- **MISSING/AMBIGUOUS:** Fig. 1 presents one embedding arrow into a decoder but does not specify whether the embedding is a prefix token, a memory sequence for cross-attention, an initial state, or projected/concatenated at every decoding step.

## 4. Pretrained checkpoint, image encoder, and text encoder

### 4.1 Checkpoint

- **PAPER-STATED:** The exact reported model name is `M-CLIP/XLM-Roberta-Large-Vit-L-14` (Sec. 4.3).
- **PAPER-STATED:** It is described as Multilingual-CLIP, a multilingual variant of CLIP, used to obtain both textual and visual features.
- **INFERRED:** From the checkpoint name, the language side is based on XLM-RoBERTa-Large and the visual side on ViT-L/14.
- **MISSING/AMBIGUOUS:** Repository/registry (for example, Hugging Face), checkpoint revision/commit, file hashes, library version, and loading API are not given.
- **MISSING/AMBIGUOUS:** The paper does not identify a separately pretrained decoder checkpoint; it only calls the decoder a transformer-based mapping network.

### 4.2 Image encoder

- **PAPER-STATED:** Multilingual-CLIP extracts \(f_M(I)\), and prompt attention enriches it before decoding (Sec. 3.3).
- **PAPER-STATED:** The paper assigns \(f_M(I)\in\mathbb{R}^{N_M\times d_M}\), suggesting one or more visual feature vectors, and requires \(d_M=d_p\) for residual addition to prompt features.
- **INFERRED:** The likely visual backbone is ViT-L/14 because this appears in the checkpoint name.
- **MISSING/AMBIGUOUS:** It is not stated whether \(f_M(I)\) is the global projected CLIP embedding, a class token, patch-token features, multiple image/view features, or a batch dimension represented as \(N_M\).
- **MISSING/AMBIGUOUS:** No input resolution, resizing/cropping, grayscale-to-RGB conversion, intensity/windowing policy for CT, interpolation, normalization mean/std, data augmentation, or view aggregation method is reported.
- **MISSING/AMBIGUOUS:** COV-CT is described as multi-image per report, but Eq. (5) takes one image and no fusion mechanism is specified.

### 4.3 Text/language encoder

- **PAPER-STATED:** The text encoder accepts English and Chinese medical reports and maps them to a shared Multilingual-CLIP feature space (Sec. 3.2).
- **PAPER-STATED:** Text embeddings are L2-normalized before decoding (Eq. (3)).
- **PAPER-STATED:** Fig. 1 marks this encoder frozen during text-only training.
- **INFERRED:** The likely text backbone is XLM-RoBERTa-Large because this appears in the checkpoint name.
- **MISSING/AMBIGUOUS:** The tokenizer, maximum input length, truncation/padding policy, pooling strategy, text projection layer, embedding dimension, casing/Unicode handling, and treatment of long reports are unspecified.
- **MISSING/AMBIGUOUS:** It is not stated whether prompt reports use the exact same encoder output as auto-encoding reports, though this is the most natural interpretation.

## 5. Multimodal and multilingual alignment

- **PAPER-STATED:** Alignment is inherited from Multilingual-CLIP’s shared image/text embedding space and augmented by report prompts at inference (Secs. 3.2–3.3).
- **PAPER-STATED:** The prompt operation is attention from image features (queries) to prompt embeddings (keys and values), followed by residual addition and L2 normalization (Eqs. (5)–(6)).
- **PAPER-STATED:** There is no additional contrastive loss in the stated ZeroMRG objective; only cross-entropy reconstruction/generation losses are given.
- **INFERRED:** The decoder is trained to interpret points on the normalized text-embedding manifold. Zero-shot inference assumes normalized image-plus-prompt features land close enough to that manifold for the same decoder to work.
- **INFERRED:** The prompt residual is the main domain-adaptation bridge because the general-purpose CLIP image embedding alone has a medical image/report modality gap; this matches the authors’ ablation interpretation (Sec. 4.5).
- **MISSING/AMBIGUOUS:** No temperature, learned projection, modality discriminator, explicit embedding-distance loss, or calibration procedure is described.
- **MISSING/AMBIGUOUS:** The softmax axis is not stated. Given the printed shapes, normalizing over the \(N_p\) prompt dimension is the dimensionally consistent interpretation.
- **MISSING/AMBIGUOUS:** The equation assumes \(d_M=d_p\), but the paper does not state whether the selected checkpoint already guarantees this or whether a projection is required.
- **MISSING/AMBIGUOUS:** The method for choosing the desired output language is absent. Possibilities such as language-specific prompt banks, language control tokens, or separate decoding calls are not described.

## 6. Auto-encoding mechanism

- **PAPER-STATED:** Auto-encoding uses only medical reports and reconstructs each report from its normalized Multilingual-CLIP text feature (Sec. 3.2).
- **PAPER-STATED:** Both English and Chinese reports are shown in the formulation and Fig. 1.
- **PAPER-STATED:** The stated purpose is to teach one decoder to generate reports from embeddings across languages/modalities in a unified feature space.
- **INFERRED:** Training-set reports from the applicable datasets/languages are likely pooled or interleaved for decoder training; the paper does not describe paired English-Chinese examples as a requirement.
- **MISSING/AMBIGUOUS:** Eq. (2) appears to multiply an English conditional probability by a Chinese conditional expression, but the Chinese factor is missing \(p_\theta\) in the typeset equation. The likely intended loss is the sum of English and Chinese negative log-likelihoods, but that correction is not paper-stated.
- **MISSING/AMBIGUOUS:** It is unclear whether English and Chinese losses are computed jointly per update, alternated by batch, or used only for datasets of the corresponding language.
- **MISSING/AMBIGUOUS:** No corruption/noising is described; this is deterministic embedding-to-text reconstruction, not a denoising auto-encoder.
- **MISSING/AMBIGUOUS:** Decoder vocabulary/tokenizer, BOS/EOS/PAD handling, loss masking, label smoothing, and maximum sequence length are absent.

## 7. Prompt-learning mechanism

- **PAPER-STATED:** Select \(n\) medical reports randomly from the training dataset and use them as prompts (Secs. 3.3 and 4.3).
- **PAPER-STATED:** Prompt counts are 500 for COV-CTR, 290 for COV-CT, and 1000 for IU-Xray.
- **PAPER-STATED:** Prompt embeddings act as keys and values in Eq. (6); image features act as queries.
- **PAPER-STATED:** The attended prompt feature is added with coefficient 1 to the base visual feature; the sum is L2-normalized (Eq. (5)).
- **PAPER-STATED:** The prompts are frozen during inference (Sec. 3.3).
- **INFERRED:** The selected reports are encoded into \(f_p\) using the Multilingual-CLIP text encoder and cached.
- **MISSING/AMBIGUOUS:** Despite the term “prompt learning,” no prompt parameters are optimized in the zero-shot method. The described mechanism is closer to a fixed, randomly sampled report-memory bank with attention.
- **MISSING/AMBIGUOUS:** The paper gives no random seed, sampling with/without replacement, selection criterion, deduplication, language composition, length filtering, or resampling schedule.
- **MISSING/AMBIGUOUS:** It is not stated whether the same prompt set is used for validation/test, whether prompts are sampled once or per run/batch/image, or whether a test report could enter the prompt bank.
- **MISSING/AMBIGUOUS:** There is no explanation for the prompt counts. COV-CT’s 290 prompts consume nearly all of a nominal 80% split of 368 reports (about 294), making exact split rounding, prompt eligibility, and overlap controls especially consequential; none are specified.

## 8. Decoder and report-generation mechanism

- **PAPER-STATED:** The decoder is multilingual and “transformer-based”; Sec. 4.3 calls it a “transformer-based mapping network.”
- **PAPER-STATED:** It consumes either a normalized report embedding during auto-encoding or \(\varphi(I)\) during image generation.
- **PAPER-STATED:** It produces token probabilities conditioned on the preceding token prefix (Eqs. (2) and (7)).
- **PAPER-STATED:** Generation terminates on `END` or at a predefined maximum length.
- **INFERRED:** Training uses causal masking and teacher forcing, and inference begins from an unreported start token.
- **MISSING/AMBIGUOUS:** Number of transformer layers, hidden size, feed-forward size, attention heads, activation, positional encoding, normalization layout, dropout rate/location, vocabulary size, parameter initialization, and weight tying are absent.
- **MISSING/AMBIGUOUS:** “Mapping network” is not defined. It is unclear whether this is a standard Transformer decoder with cross-attention, a GPT-like causal transformer receiving embedding prefixes, or an embedding mapper followed by another language decoder.
- **MISSING/AMBIGUOUS:** The paper does not specify greedy decoding, beam search, sampling, beam width, temperature, top-k/top-p, length penalty, repetition penalty, or postprocessing.
- **MISSING/AMBIGUOUS:** Sentence segmentation and report section formatting are not described.

## 9. Loss functions

### 9.1 Text-only reconstruction loss

- **PAPER-STATED:** Eq. (2) is a token-level cross-entropy/negative log-likelihood summed from \(t=1\) to \(T\), using ground-truth prefixes.
- **INFERRED:** A coherent implementation would compute per-language autoregressive CE and sum or average it over valid tokens and languages.
- **MISSING/AMBIGUOUS:** Eq. (2) is malformed/underspecified for the Chinese term, and reduction across examples/tokens/languages is not stated.

### 9.2 Supervised image-conditioned loss

- **PAPER-STATED:** Eq. (7) is image-conditioned cross-entropy over token predictions given \(\varphi(I)\) and the preceding ground-truth tokens.
- **MISSING/AMBIGUOUS:** Eq. (7) has the same malformed Chinese factor and does not define reduction or weighting.

### 9.3 Losses not present

- **PAPER-STATED:** No new contrastive, alignment, retrieval, reinforcement-learning, clinical-label, coverage, or auxiliary classification loss is specified for ZeroMRG.
- **MISSING/AMBIGUOUS:** Sec. 4.3 says the “entire framework” is trained end-to-end under cross-entropy, which conflicts with the explicitly frozen pretrained encoder in Fig. 1 for text-only training and with frozen inference components.

## 10. Frozen and trainable parameters

| Regime | Frozen | Trainable | Confidence/issues |
|---|---|---|---|
| Text-only auto-encoding | Multilingual encoder / Multilingual-CLIP text encoder. | Multilingual decoder \(D_m\). | **PAPER-STATED:** Snowflake placement in Fig. 1. **INFERRED:** Prose does not explicitly say “frozen” in Sec. 3.2. |
| Zero-shot inference | Multilingual-CLIP, prompt embeddings, and trained decoder. | None. | **PAPER-STATED:** Sec. 3.3 says Multilingual-CLIP and prompts are frozen; Fig. 1 also marks decoder frozen. |
| 10%/100% paired training | Not reliably specified. | Encoder \(E_v\) and decoder \(D_m\) are said to be trained. | **PAPER-STATED:** Sec. 3.4. **MISSING/AMBIGUOUS:** Whether this means all CLIP weights, only a projection/attention module, and/or prompt embeddings is not stated. |

## 11. Training stages and optimization

### 11.1 Stages

1. **PAPER-STATED:** Randomly split each dataset into train/validation/test with dataset-specific ratios (Sec. 4.1).
2. **PAPER-STATED:** Use medical reports for text-only auto-encoding; image-report pairs are not used in the 0% regime. **INFERRED:** These are likely restricted to the training split, but the paper does not explicitly say so.
3. **PAPER-STATED:** Randomly select training reports to form the prompt bank.
4. **PAPER-STATED:** Evaluate the frozen text-trained decoder with image-plus-prompt embeddings for 0% paired zero-shot generation.
5. **PAPER-STATED:** Optionally train with 10% or 100% of paired data and evaluate the resulting few-/fully-supervised variants.
6. **MISSING/AMBIGUOUS:** Exact stage order for the 10% and 100% models, checkpoint transfer, optimizer reset, and whether auto-encoding continues jointly during supervised training are not specified.

### 11.2 Reported hyperparameters

| Hyperparameter/procedure | Value | Classification |
|---|---:|---|
| Mini-batch size | 6 | **PAPER-STATED** |
| Optimizer | SGD | **PAPER-STATED** |
| Learning rate | \(1\times10^{-3}\) | **PAPER-STATED** |
| Momentum | 0.99 | **PAPER-STATED** |
| Regularization | Dropout | **PAPER-STATED**, but rate/location are **MISSING/AMBIGUOUS** |
| Early stopping | Stop if validation BLEU does not improve for 50 epochs | **PAPER-STATED** |
| Prompt count, COV-CTR | 500 | **PAPER-STATED** |
| Prompt count, COV-CT | 290 | **PAPER-STATED** |
| Prompt count, IU-Xray | 1000 | **PAPER-STATED** |
| Generation termination | `END` or predefined maximum length | **PAPER-STATED**, maximum is **MISSING/AMBIGUOUS** |
| Loss | Cross-entropy | **PAPER-STATED** |

### 11.3 Unreported training details

- **MISSING/AMBIGUOUS:** Maximum epochs and minimum epochs.
- **MISSING/AMBIGUOUS:** Which BLEU variant drives early stopping (BLEU-1, BLEU-4, aggregate, or another definition).
- **MISSING/AMBIGUOUS:** Learning-rate schedule/warmup, weight decay, gradient clipping, gradient accumulation, mixed precision, and checkpoint selection rule beyond early stopping.
- **MISSING/AMBIGUOUS:** Batch construction across datasets/languages/modalities and whether one joint model is trained simultaneously on all datasets or separate runs/checkpoints are used.
- **MISSING/AMBIGUOUS:** Random seeds, number of runs, variance/confidence intervals, hardware, training duration, framework, and dependency versions.

## 12. Datasets, preprocessing, and splits

### 12.1 COV-CTR

- **PAPER-STATED:** 728 images: 349 COVID-19 and 379 non-COVID, sourced from published papers, paired with English and Chinese reports.
- **PAPER-STATED:** Only English reports are used in this study.
- **PAPER-STATED:** Random 8:1:1 train/validation/test split with no overlap.
- **PAPER-STATED:** It is treated as a CT dataset in the narrative/figure.
- **MISSING/AMBIGUOUS:** Patient/study grouping, stratification, exact split counts/indices, report fields, image format, preprocessing, and whether images from the same publication/patient can cross splits are absent.
- **Repository scope note:** COV-CTR is not among the datasets currently in scope under `research/AGENTS.md`; it is included here only because a complete analysis must cover every dataset used in the paper.

### 12.2 COV-CT

- **PAPER-STATED:** 368 Chinese medical reports and 1104 chest CT images from 96 patients at two hospitals, focused on COVID-19 diagnosis.
- **PAPER-STATED:** Random 8:1:1 train/validation/test split.
- **PAPER-STATED:** The paper says each report is accompanied by ten selected CT images capturing distinct findings.
- **MISSING/AMBIGUOUS:** The stated totals are inconsistent: 368 reports × 10 images would be 3680 images, not 1104; 1104/368 equals 3 images per report.
- **MISSING/AMBIGUOUS:** It is not stated whether splitting is by image, report/study, or patient. A random image/report split could leak the same patient across subsets.
- **MISSING/AMBIGUOUS:** The method for encoding/fusing multiple CT images per report is absent.

### 12.3 IU-Xray

- **PAPER-STATED:** 7470 chest X-ray images associated with 3955 radiology reports.
- **PAPER-STATED:** Reports contain impression, findings, tags, comparison, and indication sections.
- **PAPER-STATED:** The target report is the concatenation of impression and findings.
- **PAPER-STATED:** Conventional random 70%/10%/20% train/validation/test split.
- **MISSING/AMBIGUOUS:** Exact split indices/seed, ordering and delimiter for the concatenation, missing-section handling, patient grouping, image-view handling, and image preprocessing are absent.

### 12.4 Preprocessing specified across datasets

- **PAPER-STATED:** IU-Xray target text concatenates impression and findings.
- **PAPER-STATED:** COV-CTR uses only English reports.
- **PAPER-STATED:** Text features and image-plus-prompt features are L2-normalized.
- **MISSING/AMBIGUOUS:** No other image or text preprocessing is reported. In particular, the paper omits image resize/crop/intensity normalization/augmentation, report cleanup, section separators, sentence/token segmentation, Unicode normalization, vocabulary construction, special-token definitions, sequence truncation, and padding.

## 13. Evaluation protocol and metrics

- **PAPER-STATED:** BLEU, CIDEr, and ROUGE-L are described in Sec. 4.2; Sec. 4.4 and Table 3 also use METEOR.
- **PAPER-STATED:** Metrics are computed using the “standard evaluation toolkit” attributed to the Microsoft COCO captions work (Chen et al., 2015).
- **PAPER-STATED:** BLEU-1 through BLEU-4 are reported. CIDEr and ROUGE-L are reported for COV-CTR and COV-CT. METEOR and ROUGE-L are reported for IU-Xray.
- **MISSING/AMBIGUOUS:** Sec. 4.2 omits METEOR even though it is later reported, and Sec. 4.4 says all four metric families are reported although the dataset tables do not all contain both CIDEr and METEOR.
- **MISSING/AMBIGUOUS:** Toolkit version, tokenization (especially Chinese), stemming, lowercase/punctuation rules, smoothing, corpus-vs-sentence aggregation, scaling by 100, and CIDEr configuration are not provided.
- **MISSING/AMBIGUOUS:** There is no clinical efficacy metric, labeler-based metric, factuality metric, or clinician evaluation. The authors explicitly state that fine-grained anatomical accuracy requires further clinician validation (Conclusion).
- **MISSING/AMBIGUOUS:** The paper does not report statistical uncertainty or significance tests.

## 14. Inference procedure suitable for later implementation

The following is the most literal executable sequence supported by the paper; unresolved choices remain marked rather than filled in.

1. **PAPER-STATED:** Load the trained multilingual decoder from text-only auto-encoding.
2. **PAPER-STATED:** Load `M-CLIP/XLM-Roberta-Large-Vit-L-14` and freeze it.
3. **PAPER-STATED:** Randomly select the dataset-specific number of reports from the training set as prompts.
4. **INFERRED:** Encode and L2-normalize/cache those reports using the checkpoint’s text encoder to obtain \(f_p\). Exact pooling and normalization are **MISSING/AMBIGUOUS**.
5. **PAPER-STATED:** Encode the test image with the Multilingual-CLIP visual encoder to obtain \(f_M(I)\). Image preprocessing and multiple-view aggregation are **MISSING/AMBIGUOUS**.
6. **PAPER-STATED:** Compute prompt attention using Eq. (6), with \(\sqrt d\) scaling.
7. **PAPER-STATED:** Add attended prompt features to the base visual features and L2-normalize using Eq. (5).
8. **PAPER-STATED:** Feed \(\varphi(I)\) to the frozen multilingual decoder.
9. **INFERRED:** Start decoding with a BOS/control token and decode autoregressively. The initial token, language-control method, and search algorithm are **MISSING/AMBIGUOUS**.
10. **PAPER-STATED:** Stop at `END` or the maximum length.
11. **INFERRED:** Remove special/padding tokens and detokenize. Exact postprocessing is **MISSING/AMBIGUOUS**.
12. **PAPER-STATED:** Compare the output with the held-out reference using the dataset’s reported metrics.

## 15. Reported results

The paper prints metric values on a 0–100-like scale; it does not explicitly state scaling. Values below are transcribed as printed.

### 15.1 COV-CTR English report generation (Table 1)

| Method | Paired ratio | B-1 | B-2 | B-3 | B-4 | CIDEr | R-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoAtt | 100% | 70.9 | 64.5 | 60.8 | 55.2 | 67.2 | 74.8 |
| NIC | 100% | 69.7 | 62.1 | 56.8 | 51.5 | 65.9 | 72.3 |
| AdaAtt | 100% | 67.6 | 63.3 | 59.6 | 51.4 | 68.2 | 72.6 |
| Vision-BERT | 100% | 71.0 | 65.3 | 60.6 | 55.8 | 68.4 | 74.7 |
| Vision-GPT | 100% | 70.8 | 64.5 | 60.0 | 54.9 | 68.0 | 74.6 |
| R2Gen | 100% | 71.6 | 63.8 | 57.6 | 52.4 | 152.8 | 67.7 |
| ASGK | 100% | 71.2 | 65.9 | 61.1 | 57.0 | 68.4 | 74.6 |
| ZeroMRG | 0% | 65.9 | 58.5 | 54.8 | 50.7 | 63.5 | 67.4 |
| ZeroMRG | 10% | 68.8 | 62.1 | 59.5 | 53.6 | 66.5 | 72.3 |
| ZeroMRG | 100% | 75.1 | 68.6 | 64.4 | 61.2 | 72.3 | 77.5 |

- **PAPER-STATED:** These are the reported Table 1 values.
- **MISSING/AMBIGUOUS:** R2Gen CIDEr (152.8) is on a visibly different numerical range from all surrounding CIDEr values, so direct comparison may reflect inconsistent scaling or a table error.

### 15.2 COV-CT Chinese report generation (Table 2)

| Method | Paired ratio | B-1 | B-2 | B-3 | B-4 | CIDEr | R-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoAtt | 100% | 60.8 | 53.5 | 49.4 | 46.8 | 25.5 | 57.4 |
| Vision-BERT | 100% | 57.9 | 51.4 | 47.5 | 44.9 | 29.6 | 57.8 |
| DenseNet+GPT-2 | 100% | 55.3 | 47.6 | 42.3 | 39.0 | 23.1 | 51.4 |
| VGG+GPT-2 | 100% | 55.8 | 48.0 | 42.6 | 39.2 | 31.5 | 51.3 |
| ZeroMRG | 0% | 58.8 | 51.4 | 46.2 | 41.9 | 42.8 | 50.7 |
| ZeroMRG | 10% | 62.9 | 54.8 | 50.5 | 47.6 | 46.7 | 55.1 |
| ZeroMRG | 100% | 69.4 | 61.2 | 56.6 | 53.5 | 57.3 | 60.9 |

- **PAPER-STATED:** These are the reported Table 2 values.

### 15.3 IU-Xray English report generation (Table 3)

| Method | Paired ratio | B-1 | B-2 | B-3 | B-4 | METEOR | R-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| NIC | 100% | 21.6 | 12.4 | 8.7 | 6.6 | — | 30.6 |
| AdaAtt | 100% | 22.0 | 12.7 | 8.9 | 6.8 | — | 30.8 |
| Att2in | 100% | 22.4 | 12.9 | 8.9 | 6.8 | — | 30.8 |
| Transformer | 100% | 39.6 | 25.4 | 17.9 | 13.5 | 16.4 | 34.2 |
| M² Transformer | 100% | 43.7 | 29.0 | 20.5 | 15.2 | 17.6 | 35.3 |
| R2Gen | 100% | 47.0 | 30.4 | 21.9 | 16.5 | 18.7 | 37.1 |
| R2GenCMN | 100% | 47.5 | 30.9 | 22.2 | 17.0 | 19.1 | 37.5 |
| PPKED | 100% | 48.3 | 31.5 | 22.4 | 16.8 | 19.0 | 37.6 |
| ZeroMRG | 0% | 40.8 | 25.6 | 16.5 | 13.3 | 14.7 | 33.8 |
| ZeroMRG | 10% | 43.9 | 29.8 | 19.5 | 15.4 | 17.6 | 35.5 |
| ZeroMRG | 100% | 50.5 | 35.3 | 26.2 | 18.6 | 21.4 | 40.8 |

- **PAPER-STATED:** These are the reported Table 3 values.

### 15.4 Claims based on main results

- **PAPER-STATED:** The authors claim the 0% paired version achieves over 90% of supervised baseline performance, 10% paired data approaches or surpasses fully supervised methods, and 100% ZeroMRG improves over baselines by approximately 4% (COV-CTR), 9% (COV-CT), and 3% (IU-Xray).
- **MISSING/AMBIGUOUS:** The calculation used for “over 90%” and the approximate improvement percentages is not defined (metric, baseline, averaging, or relative/absolute change).

## 16. Ablation settings and findings

### 16.1 COV-CT component ablation (Table 4)

| Paired ratio | M-CLIP | Prompts | B-1 | B-2 | B-3 | B-4 | CIDEr | R-L |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 0% | ✓ | — | 40.2 | 35.9 | 31.9 | 27.6 | 20.1 | 28.4 |
| 0% | — | ✓ | 49.4 | 45.0 | 40.7 | 36.6 | 38.1 | 44.5 |
| 0% | ✓ | ✓ | 58.8 | 51.4 | 46.2 | 41.9 | 42.8 | 50.7 |
| 10% | ✓ | — | 48.8 | 43.5 | 38.2 | 34.6 | 33.0 | 41.1 |
| 10% | — | ✓ | 58.4 | 51.3 | 47.4 | 43.4 | 41.8 | 52.2 |
| 10% | ✓ | ✓ | 62.9 | 54.8 | 50.5 | 47.6 | 46.7 | 55.1 |
| 100% | ✓ | — | 53.8 | 48.0 | 45.6 | 42.8 | 49.6 | 52.4 |
| 100% | — | ✓ | 59.5 | 55.3 | 53.2 | 49.5 | 53.4 | 56.6 |
| 100% | ✓ | ✓ | 69.4 | 61.2 | 56.6 | 53.5 | 57.3 | 60.9 |

- **PAPER-STATED:** Both components improve results, prompts alone outperform M-CLIP alone, and their combination performs best in every reported regime/metric.
- **PAPER-STATED:** The authors attribute this to fixed prompts carrying expert-report knowledge and bridging the general CLIP image/report modality gap.
- **PAPER-STATED:** They report a 112.9% zero-shot CIDEr improvement and 51.8% zero-shot BLEU-4 improvement of the combined method over M-CLIP-only, with the BLEU-4 improvement declining to 25.0% at 100% paired data (Sec. 4.5; Fig. 2).
- **MISSING/AMBIGUOUS:** “Prompts only” is not operationally defined. Eq. (6) requires \(f_M(I)\) to query prompts, so it is unclear what is removed when the M-CLIP column is off.
- **MISSING/AMBIGUOUS:** There are no ablations for prompt-bank size, report selection, frozen vs. trainable encoders, normalization, attention scaling, decoder architecture, language mixing, or multi-image fusion.

## 17. Qualitative analysis and limitations

- **PAPER-STATED:** Fig. 3 presents Chinese and English generations for CT/X-ray examples; the authors highlight findings such as shadows, nodules, calcification, and granuloma (Sec. 4.6).
- **PAPER-STATED:** Stated limitations are dependence on multilingual embedding quality, uncertain coverage of low-resource languages, need for clinician validation of fine-grained anatomical accuracy/rare diseases, hospital-system integration, and privacy compliance (Conclusion).
- **MISSING/AMBIGUOUS:** No blinded clinician assessment, error taxonomy, hallucination analysis, or patient-level safety evaluation is reported.

## 18. Reproduction blockers and priority ambiguities

These items materially prevent an exact reproduction and should be resolved in a later reproduction plan without being misrepresented as paper-stated:

1. **MISSING/AMBIGUOUS:** Exact decoder architecture and conditioning interface.
2. **MISSING/AMBIGUOUS:** Decoder tokenizer/vocabulary and language-control mechanism.
3. **MISSING/AMBIGUOUS:** Meaning and extraction level of \(f_M(I)\), \(\psi(S)\), \(N_M\), and \(N_p\).
4. **MISSING/AMBIGUOUS:** Exact checkpoint source/revision and software versions.
5. **MISSING/AMBIGUOUS:** Image preprocessing, especially CT handling and grayscale/RGB conversion.
6. **MISSING/AMBIGUOUS:** Multi-image/multi-view fusion for COV-CT and IU-Xray.
7. **MISSING/AMBIGUOUS:** Correct form and aggregation of bilingual Eq. (2) and Eq. (7).
8. **MISSING/AMBIGUOUS:** Trainable scope in 10%/100% paired training.
9. **MISSING/AMBIGUOUS:** Prompt encoding, sampling, lifetime, language, seed, and leakage prevention.
10. **MISSING/AMBIGUOUS:** Split indices, seeds, grouping level, and split leakage controls.
11. **MISSING/AMBIGUOUS:** Sequence length, decoder hyperparameters, dropout, epochs, and decoding search.
12. **MISSING/AMBIGUOUS:** Chinese tokenization and metric configuration.
13. **MISSING/AMBIGUOUS:** Whether results come from a single jointly multilingual/multimodal checkpoint or dataset-specific checkpoints.
14. **MISSING/AMBIGUOUS:** The exact semantics of “prompts only” in the component ablation.

## 19. Component-by-component implementation checklist

No item below is implemented in Phase 1. The checklist records everything a faithful codebase will need.

### Configuration and reproducibility

- [ ] Encode paper-stated dataset-specific split ratios and prompt counts. **PAPER-STATED**
- [ ] Record checkpoint identifier `M-CLIP/XLM-Roberta-Large-Vit-L-14`, plus a pinned source/revision/hash after verification. Identifier is **PAPER-STATED**; pinning details are **MISSING/AMBIGUOUS**.
- [ ] Add configuration for every undocumented choice (seed, decoder layout, preprocessing, lengths, decoding, metric tokenization) and label these as implementation assumptions. **MISSING/AMBIGUOUS**
- [ ] Persist run configuration, split manifests, prompt IDs, checkpoints, and random seeds. **INFERRED** as necessary for reproducibility.

### Dataset and split layer

- [ ] Implement COV-CT report/image parsing and stable report-to-multiple-image mapping. Dataset use is **PAPER-STATED**; exact fusion is **MISSING/AMBIGUOUS**.
- [ ] Implement IU-Xray parsing and `impression + findings` target construction. **PAPER-STATED**
- [ ] Create train/validation/test manifests at 8:1:1 for COV-CT and 70:10:20 for IU-Xray. **PAPER-STATED**; grouping/seed are **MISSING/AMBIGUOUS**.
- [ ] Prevent patient/study/report leakage across splits and document any departure from literal random splitting. **INFERRED** safety/reproducibility requirement; paper protocol is **MISSING/AMBIGUOUS**.
- [ ] Validate counts, missing reports/sections, duplicates, number of views, and the COV-CT 1104-vs.-ten-images contradiction before selecting a loader policy. **MISSING/AMBIGUOUS**
- [ ] Construct deterministic 0%, 10%, and 100% paired-data subsets for reproduction experiments. Ratios are **PAPER-STATED**; selection procedure is **MISSING/AMBIGUOUS**.

### Image preprocessing and batching

- [ ] Implement checkpoint-compatible resize/crop, channel conversion, scaling, and normalization. **MISSING/AMBIGUOUS**
- [ ] Decide and document CT intensity/window handling based on the actual stored dataset format. **MISSING/AMBIGUOUS**
- [ ] Implement multi-image/view collation, masks, and a paper-faithful fusion policy once clarified. **MISSING/AMBIGUOUS**
- [ ] Produce image tensors and identifiers without modifying raw data. **INFERRED** engineering requirement.

### Text preprocessing and tokenization

- [ ] Implement report cleanup while preserving clinical meaning and section boundaries. **MISSING/AMBIGUOUS**
- [ ] Load the Multilingual-CLIP/XLM-R text tokenizer for encoder inputs. Backbone is **INFERRED** from the checkpoint name; exact tokenizer is **MISSING/AMBIGUOUS**.
- [ ] Choose/implement the decoder tokenizer or vocabulary for both English and Chinese. **MISSING/AMBIGUOUS**
- [ ] Define BOS, `END`/EOS, PAD, unknown-token, truncation, padding, and maximum-length behavior. Only `END`/maximum stopping is **PAPER-STATED**; exact definitions are **MISSING/AMBIGUOUS**.
- [ ] Implement causal input/target shifting and valid-token loss masks. **INFERRED** from Eqs. (2) and (7).
- [ ] Define an explicit output-language control mechanism if one decoder serves both languages. **MISSING/AMBIGUOUS**

### Pretrained multilingual encoder

- [ ] Load and validate the exact Multilingual-CLIP checkpoint. **PAPER-STATED** identifier.
- [ ] Expose frozen text embedding extraction \(\psi(S)\). **PAPER-STATED**
- [ ] Expose frozen image embedding extraction \(f_M(I)\). **PAPER-STATED**
- [ ] Determine and test global-vs.-token embedding extraction, pooling, projection, and dimensionality. **MISSING/AMBIGUOUS**
- [ ] Implement numerically safe L2 normalization for text and enriched visual embeddings. **PAPER-STATED**

### Prompt bank and alignment

- [ ] Sample and persist 290 COV-CT or 1000 IU-Xray training reports as prompts. **PAPER-STATED** counts; sampling details are **MISSING/AMBIGUOUS**.
- [ ] Encode/cache prompt embeddings and record prompt report IDs. Encoding route is **INFERRED**; prompt embeddings are **PAPER-STATED**.
- [ ] Implement scaled dot-product image-to-prompt attention from Eq. (6), with tests for shapes and softmax axis. Equation is **PAPER-STATED**; axis is **MISSING/AMBIGUOUS**.
- [ ] Implement residual addition and L2 normalization from Eq. (5). **PAPER-STATED**
- [ ] Enforce/check \(d_M=d_p\), adding no projection unless later documented as an explicit necessary assumption. Equality is **PAPER-STATED**; how achieved is **MISSING/AMBIGUOUS**.
- [ ] Freeze prompt embeddings in the zero-shot inference path. **PAPER-STATED**

### Multilingual report decoder

- [ ] Implement the transformer-based multilingual decoder/mapping network. **PAPER-STATED** at a high level; exact architecture is **MISSING/AMBIGUOUS**.
- [ ] Implement a clearly documented encoder-embedding conditioning interface. **MISSING/AMBIGUOUS**
- [ ] Implement causal self-attention, positional information, output projection, and autoregressive token probabilities. **INFERRED** from the stated conditional likelihood.
- [ ] Select/configure layers, heads, hidden/FFN dimensions, activations, normalization, dropout, initialization, and weight tying. **MISSING/AMBIGUOUS**
- [ ] Support English and Chinese output with one decoder, as claimed. **PAPER-STATED**; control mechanism is **MISSING/AMBIGUOUS**.

### Losses

- [ ] Implement token-level reconstruction cross-entropy for text-only training. **PAPER-STATED**
- [ ] Implement image-conditioned token cross-entropy for paired training. **PAPER-STATED**
- [ ] Resolve/document English-Chinese loss combination, token reduction, and padding exclusion. **MISSING/AMBIGUOUS**
- [ ] Do not add an alignment/contrastive/clinical auxiliary loss as if it were in the paper. Its absence is **PAPER-STATED** by the provided objective.

### Training stages

- [ ] Stage A: freeze Multilingual-CLIP text encoder and train the decoder by report reconstruction. **INFERRED** from Fig. 1 with strong visual evidence.
- [ ] Use batch size 6, SGD, learning rate \(10^{-3}\), and momentum 0.99. **PAPER-STATED**
- [ ] Add dropout and validation-BLEU early stopping with patience 50. **PAPER-STATED**; rate and BLEU variant are **MISSING/AMBIGUOUS**.
- [ ] Save/reload the best auto-encoding checkpoint for zero-shot inference. **INFERRED**
- [ ] Stage B: run frozen 0%-paired inference with Multilingual-CLIP + prompts + decoder. **PAPER-STATED**
- [ ] Stage C: optionally initialize from auto-encoding and train 10%/100% paired variants. Existence is **PAPER-STATED**; initialization and parameter scope are **MISSING/AMBIGUOUS**.
- [ ] Decide/document optimizer state reset, unfreezing scope, joint-vs.-sequential objectives, maximum epochs, and LR schedule. **MISSING/AMBIGUOUS**

### Report generation

- [ ] Implement autoregressive generation conditioned on \(\varphi(I)\). **PAPER-STATED**
- [ ] Stop on `END` or configured maximum length. **PAPER-STATED**
- [ ] Select and document greedy/beam/sampling strategy and its hyperparameters. **MISSING/AMBIGUOUS**
- [ ] Detokenize and reconstruct readable report sections without altering clinical content. **MISSING/AMBIGUOUS**
- [ ] Support batch inference and multi-image studies. Batch behavior is **INFERRED**; multi-image handling is **MISSING/AMBIGUOUS**.

### Evaluation and ablations

- [ ] Implement BLEU-1/2/3/4 and ROUGE-L for both in-scope datasets. **PAPER-STATED**
- [ ] Implement CIDEr for COV-CT and METEOR for IU-Xray. **PAPER-STATED**
- [ ] Pin a COCO-caption-compatible toolkit and explicitly define English/Chinese tokenization and score scaling. Toolkit family is **PAPER-STATED**; exact configuration is **MISSING/AMBIGUOUS**.
- [ ] Evaluate 0%, 10%, and 100% paired regimes. **PAPER-STATED**
- [ ] Reproduce M-CLIP-only, prompts-only, and combined ablations where operationally definable. **PAPER-STATED**; prompts-only semantics are **MISSING/AMBIGUOUS**.
- [ ] Store predictions, references, prompt IDs, per-run metrics, and aggregate results without inserting paper numbers as experimental outputs. **INFERRED** reproducibility/integrity requirement.
- [ ] Add sanity tests for finite loss, embedding norms/shapes, frozen gradients, prompt attention weights, checkpoint round-trips, termination, and deterministic evaluation. **INFERRED** implementation requirement.

## 20. Phase 1 conclusion

The paper defines a clear high-level idea: train a multilingual transformer decoder to invert frozen Multilingual-CLIP report embeddings, then reuse that decoder on a normalized sum of image embeddings and attention-retrieved report-prompt embeddings. Its equations specify the prompt attention and residual normalization, and its experiments specify the checkpoint name, optimizer basics, prompt counts, split ratios, and evaluation families. However, exact reproduction is blocked by missing decoder details, preprocessing/tokenization, language control, multi-image handling, split manifests, prompt sampling protocol, embedding extraction level, and several inconsistent equations/names. These gaps must remain explicit inputs to the later reproduction plan; they must not be presented as facts from the paper.
