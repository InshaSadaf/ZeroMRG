# Kaggle setup for ZeroMRG

This repository is developed locally and is intended to execute primarily in a Kaggle Notebook with GPU acceleration. The current implementation scope provides environment checks, read-only dataset validation, and deterministic dataset/text artifacts. It does not implement or train the ZeroMRG model or download checkpoints.

## 1. Create the notebook and enable a GPU

Create or open a Kaggle Notebook, open **Notebook options**, and select an available GPU accelerator. Kaggle may assign different accelerator models over time; the code detects all visible GPUs and selects one configured primary device. Distributed training is not enabled.

Make the repository available in the notebook, for example by uploading a versioned source archive or cloning your Git repository into `/kaggle/working`. Set the notebook working directory to the repository root before running commands.

## 2. Attach the datasets

Use **Add Input** in the Kaggle notebook to attach the COV-CTR and IU-Xray datasets. Kaggle mounts attached inputs read-only below:

```text
/kaggle/input/<actual-cov-ctr-dataset-name>/
/kaggle/input/<actual-iu-xray-dataset-name>/
```

No dataset slug is assumed by this repository. After attaching the datasets, inspect the names shown under `/kaggle/input` and supply those actual paths as configuration overrides. The bootstrap lists the attached input directories for this purpose.

`/kaggle/input` is immutable. ZeroMRG must never preprocess in place, rename raw files, or write into an attached dataset. Generated manifests, processed data, logs, checkpoints, results, and other artifacts belong under `/kaggle/working/zeromrg`.

## 3. Verify dependencies

Kaggle images commonly include PyTorch, NumPy, pandas, Pillow, and other scientific packages. Check first so an already compatible CUDA/PyTorch installation is not needlessly replaced:

```bash
python scripts/check_environment.py --config configs/base.yaml --config configs/kaggle.yaml
```

Install only packages reported missing, using `requirements.txt` as the dependency specification. Source code and bootstrap scripts never install packages. The later IU-Xray METEOR evaluation also requires a Java runtime, which is not a Python package.

## 4. Configure actual input paths

Paths can be supplied without editing Python source. For a one-time notebook run:

```bash
python scripts/kaggle_bootstrap.py \
  --cov-ctr-path /kaggle/input/<actual-cov-ctr-dataset-name> \
  --iu-xray-path /kaggle/input/<actual-iu-xray-dataset-name>
```

Angle-bracket placeholders must be replaced with the directories actually listed in the notebook. An equivalent composed environment/dataset check is:

```bash
python scripts/check_environment.py \
  --config configs/cov_ctr.yaml \
  --config configs/kaggle.yaml \
  --override paths.datasets.cov_ctr=/kaggle/input/<actual-cov-ctr-dataset-name>
```

Run IU-Xray independently by replacing `configs/cov_ctr.yaml` with `configs/iu_xray.yaml` and overriding `paths.datasets.iu_xray`. This keeps the two experiments separate as required by the reproduction plan.

For a persistent notebook-specific setup, copy `configs/kaggle.yaml` to a small private overlay and fill in the two dataset paths there. Do not commit private or machine-specific paths if they are not portable.

## 5. What the bootstrap does

`scripts/kaggle_bootstrap.py` performs only safe setup checks. It:

- verifies the source/config/script directories;
- loads and hashes the resolved Kaggle configuration;
- lists immediate children of `/kaggle/input` without changing them;
- validates a configured dataset path only when one is provided;
- creates and write-tests the configured output directories below `/kaggle/working`;
- reports the single selected CPU/CUDA device and visible GPU count; and
- reports missing Python dependencies.

It does not extract or preprocess datasets, download model weights, or start training.

## 6. Persist later outputs

Kaggle working storage is writable but tied to the notebook session/version. Later stages will write to:

```text
/kaggle/working/zeromrg/data/processed/
/kaggle/working/zeromrg/checkpoints/
/kaggle/working/zeromrg/results/
/kaggle/working/zeromrg/logs/
/kaggle/working/zeromrg/artifacts/
```

After a future run finishes, use **Save Version** with outputs enabled. For important checkpoints and result bundles, create a private Kaggle Dataset from the notebook outputs or download them before the session is discarded. Never copy generated files back into `/kaggle/input`.

## 7. Run read-only dataset validation

After configuring both real input paths, build audit and valid-sample manifests with:

```bash
python scripts/validate_data.py --dataset all \
  --cov-ctr-path /kaggle/input/<actual-cov-ctr-dataset-name> \
  --iu-xray-path /kaggle/input/<actual-iu-xray-dataset-name>
```

The validator streams image files sequentially and writes only beneath `/kaggle/working/zeromrg/data/processed`. It does not extract into or otherwise modify `/kaggle/input`. Run `--dataset cov_ctr` or `--dataset iu_xray` to validate one source independently.

## 8. Prepare deterministic text artifacts

After validation succeeds, create or safely reuse deterministic splits, normalized reports, decoder vocabularies, paired-10 IDs, and prompt candidate IDs:

```bash
python scripts/prepare_text_data.py --dataset all
```

This phase reads only the validated manifests. It does not copy or preprocess images, create prompt embeddings, load pretrained models, or train anything. Existing split artifacts are reused only after their hashes and complete memberships pass compatibility checks.

## 9. Local development

Local diagnostics use repository-relative paths and contain no drive-letter assumptions:

```bash
python scripts/check_environment.py --config configs/base.yaml --config configs/local.yaml
python -m unittest discover -s tests -v
```

Use dotted-path overrides when local archives live elsewhere, for example `--override paths.datasets.iu_xray=relative/or/absolute/path`. A normal development machine should be reported as `LOCAL`; unavailable Kaggle mount paths are not an error during local checks.

## 10. Resource-constrained IU-Xray-500 profile

The full COV-CTR population remains unchanged and is identified as `FULL DATASET REPRODUCTION`: 714 usable samples, a 571/71/72 split, 57 paired-training IDs, and 500 prompt IDs. The canonical full-IU configuration and artifacts also remain available and unchanged.

For limited Kaggle storage, memory, or runtime, build the explicitly labelled `RESOURCE-CONSTRAINED` IU-Xray profile locally:

```bash
python scripts/build_iu_xray_kaggle_subset.py
```

Use `--source <path>` if the original IU-Xray ZIP/folder is not at the path in `configs/local.yaml`. The command deterministically selects 500 validated study UIDs, retains every image view for those UIDs, creates a 350/50/100 split, selects 35 paired-training IDs and 250 prompt IDs, and builds a vocabulary from the 350 training reports only. It writes selection provenance to `data/processed/iu_xray/subset_500_ids.json` and profile-specific derived artifacts to `data/processed/iu_xray/kaggle_500/`.

The portable upload is created as both `artifacts/IU-Xray-500/` and `artifacts/IU-Xray-500.zip`. The package retains the compatible IU layout (`indiana_reports.csv`, `indiana_projections.csv`, and `images/images_normalized/`), contains only selected metadata rows and images, preserves image bytes, and includes its provenance and validation summary. Upload the ZIP as a private Kaggle Dataset, attach it to the notebook, and point `paths.datasets.iu_xray` at its actual `/kaggle/input/...` mount. No Kaggle dataset slug is assumed.

Runs using this profile must load `configs/iu_xray_kaggle_500.yaml` and keep checkpoints/results under an equivalent `iu_xray/kaggle_500` namespace. They must be reported as resource-constrained subset results, never as results on the paper's full IU-Xray population.

After validating the attached reduced package, prepare its profile-specific text artifacts with:

```bash
python scripts/prepare_text_data.py --dataset iu_xray --profile kaggle_500 \
  --override paths.datasets.iu_xray=/kaggle/input/<iu-xray-500-dataset>
```

The command verifies that the 500 validated UIDs and all mapped views exactly match the packaged `subset_500_ids.json` before creating or reusing the 350/50/100 split and other profile artifacts. Do not run reduced IU data through the default `--profile full` path.
