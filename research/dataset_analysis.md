# Dataset Analysis: COV-CTR and IU-Xray

## Scope and inspection method

This report covers dataset analysis only. It does not preprocess the datasets, implement a `Dataset`/`DataLoader`, implement the model, combine the datasets, define a new architecture, or start training.

The only dataset files currently present under `data/` are two ZIP archives:

```text
data/
├── COV-CTR with English reports.zip
└── IU-Xray.zip
```

All inspection was read-only. The audit:

- enumerated every ZIP entry;
- parsed every CSV logical record with a CSV-aware parser rather than counting physical lines;
- read the file signature and image header of every image;
- joined all annotation image keys against all physical image basenames;
- computed missing-key, duplicate-key, field-null, and view-count statistics;
- verified suspected byte-duplicate IU-Xray images by direct byte comparison; and
- visually inspected representative COV-CTR CT images and IU-Xray frontal/lateral radiographs.

The original archives were not extracted into the project, altered, renamed, or overwritten. Counts below describe the actual supplied archives, not documentation-derived expectations.

One naming discrepancy should be explicit: `research/AGENTS.md` describes a future `data/COV_CT/` location, but the user-selected dataset and the actual supplied archive are **COV-CTR**, which is also a distinct dataset in the paper. This report analyzes the supplied COV-CTR archive only.

## 1. COV-CTR Dataset Analysis

### 1.1 Exact archive structure

```text
COV-CTR with English reports.zip
└── COV-CTR/
    ├── .DS_Store
    ├── reports_ZH_EN.csv
    ├── 544 files named with a .png extension
    └── 202 files named with a .jpg extension
```

There are 749 ZIP entries: one directory entry and 748 files. The 748 files consist of 746 image files, one CSV, and one macOS `.DS_Store` artifact. The image directory is flat; there are no train, validation, test, patient, study, or modality subdirectories.

### 1.2 Images

| Property | Observed value |
|---|---:|
| Physical image files | 746 |
| Filename extensions | 544 `.png`; 202 `.jpg` |
| Actual formats by signature | 510 PNG; 236 JPEG |
| Extension/content mismatch | 34 `.png` files actually contain JPEG data |
| Successfully readable image headers | 746/746 |
| Unique width × height pairs | 637 |
| Width range | 102–1671 px |
| Height range | 61–1308 px |
| Most frequent size | 287 × 202 px (16 images) |
| Orientation | 736 landscape; 8 square; 2 portrait |
| Uncompressed file-size range | 9,451–1,594,100 bytes |

Actual pixel encodings are heterogeneous:

- PNG: 315 8-bit RGB, 174 8-bit RGBA, and 21 8-bit grayscale images.
- JPEG: 213 8-bit three-component and 23 8-bit single-component images.

Representative images were visually confirmed to be rendered axial chest CT slices. They are ordinary raster images, not DICOM volumes. Some images visibly contain source-publication artifacts such as colored circles/annotations, cropping, and variable window/contrast. The dataset therefore contains heterogeneous rendered figures rather than a standardized CT acquisition series.

No byte-duplicate image candidates were found using the ZIP size/CRC index.

### 1.3 Annotation/report file

`COV-CTR/reports_ZH_EN.csv` is a UTF-8 CSV with a byte-order mark. It has 1,453 physical lines but only 726 logical data records because every quoted `reports_En` value contains a trailing line feed. A line-oriented parser would therefore misread this file; a standards-compliant CSV parser reads it as 726 records.

The exact columns are:

| Column | Meaning observed from values | Empty values | Unique values after trimming |
|---|---|---:|---:|
| `image_id` | Image filename including extension; serves as the only join key | 0 | 720 |
| `findings` | Chinese narrative radiology findings | 0 | 664 |
| `terminologies` | Chinese abnormality/terminology summary | 0 | 649 |
| `COVID` | Binary `0`/`1` label | 0 | 2 |
| `impression` | Chinese diagnostic impression | 0 | 201 |
| `reports_En` | English free-text report, apparently corresponding to the findings narrative | 0 | 709 after trimming |

Field characteristics:

- `findings`, `terminologies`, and `impression` are Chinese.
- `reports_En` is English and is the paper-compatible target for COV-CTR, since the paper says it uses only English reports for this dataset.
- There is no separate English `findings` column and no English `impression` column.
- Every `reports_En` value has exactly one trailing newline; all 726 therefore require whitespace normalization before tokenization.
- The Chinese fields have no blank values. English reports have no blank values after trimming.
- The `COVID` column is a classification label, not report text. Record-level counts are 335 positive and 391 negative. Counts over unique `image_id` values are 331 positive and 389 negative.
- Chinese narrative length ranges are 36–127 characters for `findings`, 6–87 for `terminologies`, and 4–34 for `impression`.
- Trimmed English reports range from 97–666 characters and 17–116 whitespace-delimited tokens; the median is 401 characters / 68 whitespace-delimited tokens.

### 1.4 Identifiers

- **Image ID:** `image_id` is the complete basename, such as `20.jpg`, `83%2.jpg`, or `2020.02.17.20024018-p17-61%1.png`. Naming is heterogeneous: simple numeric names, numeric names with percent-index suffixes, and source-publication-derived names all occur.
- **Report ID:** no independent report identifier exists.
- **Study/patient ID:** no explicit study or patient identifier exists.
- Percent-indexed and source-derived filenames suggest related image groups, but filename parsing alone cannot establish a reliable study/report identity.

### 1.5 Image-to-report mapping

The intended mapping is a direct equality join:

```text
reports_ZH_EN.csv:image_id == COV-CTR/<image basename>
```

Observed mapping results:

- All 720 unique referenced image IDs have a physical image file.
- No annotation references a missing image.
- 26 physical images are not referenced by any CSV record and therefore have no usable report/label mapping.
- Six image IDs appear twice in the CSV. In every case the two full records differ; these are not harmless exact duplicate rows.
- There are no exact duplicate complete CSV rows.
- There are 714 image IDs with exactly one annotation and 6 image IDs with two conflicting/variant annotations.
- Twelve distinct English report strings are associated with more than one unique image ID, covering 29 images; the largest identical-text group covers four images. Without a report/study ID, identical text does not prove that these images belong to one report rather than separate reports with generic identical wording.

The six duplicate image keys are:

| Duplicate `image_id` | Nature of conflict |
|---|---|
| `2020.02.13.20022673-p13-77%2.png` | Same Chinese findings/impression and label; punctuation/terminology and English report variants differ |
| `2020.02.17.20024018-p17-61%1.png` | Same Chinese fields and label; English report variants differ |
| `2020.02.17.20024018-p17-61%3.png` | Same Chinese fields and label; English report variants differ |
| `2020.02.26.20026989-p34-114_1%0.png` | Findings, terminology, impression, and English report differ; label agrees |
| `20.jpg` | Findings, terminology, impression, and English report differ; label agrees |
| `83%2.jpg` | Findings, terminology, and English report differ; impression and label agree |

The 26 unreferenced image basenames are:

```text
18%3.jpg
2020.01.24.919183-p27-134.png
2020.02.22.20024927-p19-68%0.png
2020.02.22.20024927-p19-68%1.png
2020.02.22.20024927-p19-68%2.png
2020.02.24.20027201-p19-670.png
2020.02.26.20026989-p34-114_2%0.png
2020.03.03.20030775-p12-93%0.png
2020.03.04.20031047-p14-87%0.png
2020.03.04.20031047-p14-87%1.png
2020.03.07.20031393-p7-50%0.png
2020.03.07.20031393-p7-50%1.png
2020.03.07.20031393-p7-50%2.png
2020.03.07.20031393-p7-50%3.png
2020.03.08.20031658-p15-103.png
2020.03.08.20031658-p15-104.png
2020.03.08.20031658-p15-105.png
2020.03.08.20031658-p15-106.png
25%0.jpg
26%2.jpg
43%3.jpg
5%7.jpg
6%2.jpg
83%3.jpg
89%3.jpg
Talaromyces-marneffei-infection-relapse-presenting-as-ost_2020_International-p1-12%3.png
```

### 1.6 Existing splits and usable count

There are no existing split files, split columns, or split directories. The paper calls for a random 8:1:1 split but supplies neither indices nor a random seed.

Usability depends on how annotation conflicts are handled:

- **714 clean, unambiguous image-report pairs** are immediately identifiable (one physical image, one annotation record).
- **720 unique referenced images** could be used only after an explicit, documented adjudication/policy for the six duplicate keys.
- Treating all 726 CSV rows as independent samples would repeat six images with differing text and risks image leakage across splits; this is not a safe implicit choice.
- The 26 unreferenced physical images are not usable for report generation without external annotations, which are out of scope.

The paper reports 728 COV-CTR images split as 349 COVID / 379 non-COVID. None of the actual archive count views match that claim: 746 physical images, 726 CSV records, or 720 uniquely referenced images; record labels are 335/391 and unique-image labels are 331/389.

## 2. IU-Xray Dataset Analysis

### 2.1 Exact archive structure

```text
IU-Xray.zip
├── indiana_projections.csv
├── indiana_reports.csv
└── images/
    └── images_normalized/
        └── 7,470 files named *.dcm.png
```

The archive contains 7,472 files: 7,470 images and two root-level CSV files. There are no split files or split-specific directories.

### 2.2 Images

| Property | Observed value |
|---|---:|
| Physical image files | 7,470 |
| Filename extension | `.dcm.png` for every image |
| Actual format by signature | PNG for all 7,470 images; these are not DICOM files |
| Pixel encoding | 8-bit grayscale for all 7,470 images |
| Successfully readable image headers | 7,470/7,470 |
| Unique width × height pairs | 774 |
| Width range | 1529–2891 px |
| Height range | 1760–3001 px |
| Orientation | 3,396 portrait; 2,610 landscape; 1,464 square |
| Most frequent sizes | 2048×2496 (1,927), 2048×2048 (1,453), 2496×2048 (1,249) |
| Uncompressed file-size range | 10,464–4,765,500 bytes |

Representative images and their projection metadata were visually confirmed as chest radiographs with frontal and lateral views.

Three image pairs are exactly byte-identical despite having distinct filenames:

```text
1015_IM-0001-1001.dcm.png == 1015_IM-0013-1001.dcm.png
1015_IM-0001-2001.dcm.png == 1015_IM-0013-2001.dcm.png
3245_IM-1537-1001.dcm.png == 3245_IM-1537-4001.dcm.png
```

The first two pairs share report UID 1015 but use different embedded `IM` identifiers. The third pair shares report UID and embedded study code but has different view/image codes. These duplicates must be flagged; filenames alone do not make their pixel content distinct.

### 2.3 Projection/image metadata

`indiana_projections.csv` is UTF-8 CSV with these columns:

| Column | Meaning | Missing | Unique |
|---|---|---:|---:|
| `uid` | Report/study join key | 0 | 3,851 |
| `filename` | Image basename | 0 | 7,466 |
| `projection` | View category | 0 | 2 |

There are 7,466 projection records with no exact duplicate rows and no repeated filename keys. Projection counts are:

- Frontal: 3,818
- Lateral: 3,648

All 7,466 filenames referenced by the projection CSV exist in the archive. Four additional physical images have no projection record:

| Unlisted physical image | Filename-derived UID | Listed images already associated with that UID |
|---|---:|---|
| `2084_IM-0715-1001-0002.dcm.png` | 2084 | One listed frontal and one listed lateral |
| `2084_IM-0715-2001-0001.dcm.png` | 2084 | One listed frontal and one listed lateral |
| `2560_IM-1064-4001.dcm.png` | 2560 | Two listed frontal and one listed lateral |
| `3809_IM-1919-1003002.dcm.png` | 3809 | One listed frontal and one listed lateral |

The four filenames appear to identify existing UIDs, but their projection and intended inclusion cannot be assumed because the authoritative mapping CSV omits them.

### 2.4 Report metadata and fields

`indiana_reports.csv` is UTF-8 CSV with 3,851 logical records and eight columns:

| Column | Observed role | Blank values | Distinct trimmed values |
|---|---|---:|---:|
| `uid` | Unique report/study ID and projection-table join key | 0 | 3,851 |
| `MeSH` | Semicolon-delimited detailed terminology/index terms; `normal` is common | 0 | 1,900 |
| `Problems` | Semicolon-delimited higher-level problem labels | 0 | 1,432 |
| `image` | Free-text exam/view description, not an image filename | 0 | 627 |
| `indication` | Clinical indication | 86 | 2,455 |
| `comparison` | Comparison text | 599 | 395 |
| `findings` | Findings narrative | 514 | 2,554 including blank |
| `impression` | Impression narrative | 31 | 1,771 including blank |

Additional text observations:

- Reports and metadata text are English.
- `MeSH` and `Problems` provide terminology/label fields, but the ZeroMRG paper does not use them as required generation inputs.
- De-identification placeholder `XXXX` is widespread: it appears in 2,031 `image`, 2,349 `indication`, 1,597 `comparison`, 1,426 `findings`, and 644 `impression` values.
- The comparison field also contains 1,615 values equivalent after trimming/punctuation normalization to `None`, `None.`, or `None available`; these are present strings, not CSV nulls.
- Nonblank findings range from 33–1,054 characters (7–169 whitespace tokens), median 207 characters / 29 tokens.
- Nonblank impressions range from 5–887 characters (1–130 whitespace tokens), median 40 characters / 5 tokens.

Target-section availability is:

| Findings | Impression | Reports/studies |
|---|---|---:|
| Present | Present | 3,331 |
| Present | Blank | 6 |
| Blank | Present | 489 |
| Blank | Blank | 25 |

Thus, 3,331 reports strictly support the paper’s `impression + findings` target with both sections present. A relaxed “use whichever target section exists” policy would yield 3,826 reports, but that policy is not specified by the paper and must not be adopted silently.

There are no exact duplicate complete report rows and no duplicate `uid` values. When `uid` is excluded, 43 groups (117 records total, largest group 24) have identical values in every other report column. Some may be legitimate repeated normal-report templates rather than erroneous duplicates, so they require provenance-aware review rather than automatic deletion.

### 2.5 Identifiers and image-to-report mapping

- **Report/study ID:** `uid`, stored as a numeric string. Values span 1–3999 with 148 unused integers; the IDs are not a contiguous row index.
- **Image ID:** the complete filename, generally beginning with `<uid>_IM-...` and ending in `.dcm.png`. The suffix has several variable forms, so a single rigid filename regex does not cover all images.
- **Authoritative join:** `indiana_reports.uid == indiana_projections.uid`, then `indiana_projections.filename` identifies each image under `images/images_normalized/`.
- **Set integrity:** the report UID set and projection UID set are identical. Every one of the 3,851 reports maps to at least one listed image, every projection record maps to a report, and every listed projection filename exists.
- The `image` column in `indiana_reports.csv` is a textual exam description and must not be mistaken for the filename key.

Images per report/study are:

| Listed images per `uid` | Studies/reports | Listed images represented |
|---:|---:|---:|
| 1 | 446 | 446 |
| 2 | 3,210 | 6,420 |
| 3 | 181 | 543 |
| 4 | 13 | 52 |
| 5 | 1 | 5 |
| **Total** | **3,851** | **7,466** |

The common view composition is one frontal plus one lateral image (3,194 studies). Other studies include frontal-only, lateral-only, repeated frontal/lateral views, or three to five images.

This one-to-many study structure explains why image and report counts differ: a single radiology report describes a study/exam, while the study can contain multiple projection images. The physical count is four higher than the projection-table count because four files are not listed in the mapping CSV.

### 2.6 Existing splits and usable count

No split column/file/folder exists. The paper requires a random 70%/10%/20% split but gives no indices or seed. Splitting must be performed at `uid` (report/study) level so views from one report cannot cross subsets.

Approximate usable counts are:

- **Strict paper target:** 3,331 study/report samples with both findings and impression, covering 6,457 listed images.
- **Relaxed at-least-one-section target:** 3,826 study/report samples covering 7,426 listed images. This is not paper-stated and is presented only as an upper bound.
- **All report records:** 3,851, but 25 have neither findings nor impression and cannot form the paper’s target report.
- The four images absent from `indiana_projections.csv` should not enter the usable count without a documented mapping decision.

The paper states 3,955 IU-Xray reports and 7,470 images. The supplied archive has the stated physical image count but only 3,851 report rows and 3,851 mapped UIDs; 7,466 images are actually represented by the projection metadata.

## 3. Dataset Statistics

### 3.1 Consolidated counts

| Statistic | COV-CTR | IU-Xray |
|---|---:|---:|
| Physical image files | 746 | 7,470 |
| Metadata-listed images | 720 unique (`726` rows) | 7,466 |
| Annotation/report records | 726 | 3,851 |
| Independent report ID available | No | Yes (`uid`) |
| Unique target report text | 709 trimmed English strings | 2,554 findings values / 1,771 impression values including blanks |
| Reports with complete required target | 714 unambiguous pairs; up to 720 after conflict resolution | 3,331 with both findings and impression |
| Images with no metadata mapping | 26 | 4 |
| Metadata image references with no file | 0 | 0 |
| Exact duplicate metadata rows | 0 | 0 |
| Conflicting duplicate mapping keys | 6 image IDs | 0 UIDs / 0 projection filenames |
| Exact byte-duplicate image pairs | 0 detected | 3 pairs |
| Existing split | None | None |
| Report language used by paper | English | English |
| Modality | Rendered axial chest CT | Chest radiograph (frontal/lateral) |

### 3.2 Paper count comparison

| Dataset | Paper | Supplied archive |
|---|---|---|
| COV-CTR | 728 images: 349 COVID, 379 non-COVID | 746 physical images; 726 rows; 720 unique referenced images; unique labels 331 COVID, 389 non-COVID |
| IU-Xray | 7,470 images and 3,955 reports | 7,470 physical images; 7,466 mapped images; 3,851 reports/UIDs |

These differences mean exact reproduction of the paper’s sample population and reported random splits is not possible from the supplied archives alone.

## 4. Image-Report Mapping

### 4.1 COV-CTR mapping model

The only supported unit is currently an annotation row linked by filename to one image. There is no stable report/study/patient key. The safe conceptual relation is:

```text
annotation row ──image_id/basename──> one raster CT image
```

This relation is not strictly one-to-one because six basenames have two differing annotation rows. Repeated English strings across different image IDs cannot safely be converted into one-report/multiple-image groups: identical clinical templates may occur in unrelated cases. Filename stems likewise cannot be treated as study IDs without additional provenance.

### 4.2 IU-Xray mapping model

IU-Xray has an explicit normalized one-to-many relation:

```text
indiana_reports row (uid)
          │
          ├── indiana_projections row (uid, filename, Frontal)
          ├── indiana_projections row (uid, filename, Lateral)
          └── possibly additional projection rows
                         │
                         └── images/images_normalized/<filename>
```

The report and study are represented by the same `uid`; image views have distinct filenames. A loader must split at UID level and preserve all view metadata even if a later reproduction decision selects one view.

## 5. Paper vs Dataset Compatibility

The paper’s core data expectation is a medical image \(I\) and report \(S\). Text-only auto-encoding needs training reports; prompts are randomly drawn from training reports (500 for COV-CTR and 1,000 for IU-Xray); zero-shot inference needs images; paired 10%/100% variants additionally need image-report pairs. The paper depicts/equates \(I\) as a single image and does not specify multi-view fusion. It uses English COV-CTR reports, concatenated IU-Xray impression and findings, random 8:1:1 COV-CTR splits, and random 70:10:20 IU-Xray splits. Image preprocessing is not specified beyond the later embedding normalization.

| Requirement from Paper | COV-CTR | IU-Xray | Compatible? | Adaptation Required |
|---|---|---|---|---|
| Imaging modality | Rendered axial chest CT images, matching the paper’s COV-CTR modality | Chest radiographs with frontal/lateral projections, matching the paper | Yes | Dataset-specific raster decoding and channel normalization; no modality conversion |
| Image input structure | Row-level mapping suggests one raster image per annotation, but related-slice/study grouping is unavailable | 1–5 images per report/study; most have frontal+lateral | Partial | Preserve image lists; the paper’s missing multi-view handling must be resolved explicitly later rather than silently flattening views |
| Image-report pairing | Direct filename join, but six keys have conflicting annotations and 26 images lack reports | Explicit UID join; all report UIDs map to images | COV-CTR partial; IU-Xray yes | COV-CTR conflict policy required; IU-Xray join reports→projections→files |
| Report structure | English free-text `reports_En`; separate Chinese findings/impression exist but no English section split | Structured findings and impression plus auxiliary sections | Yes with preprocessing | Strip COV-CTR trailing LF; concatenate IU fields exactly as the paper requires, with an explicit missing-section policy |
| Language | English target available; Chinese auxiliary fields also present | English | Yes | Use English targets independently for each dataset; do not combine datasets in this phase |
| Number of images per study | Cannot be determined because no study ID exists | Explicitly 1–5; 3,405/3,851 UIDs have more than one image | Partial | COV grouping cannot be recovered safely; IU variable-view collation and a documented paper-faithful input policy are required |
| Image formats | Mixed PNG/JPEG signatures, mixed grayscale/RGB/RGBA, many dimensions; 34 mislabeled extensions | Uniform 8-bit grayscale PNG, many dimensions | Partial | Decode by content, convert to checkpoint-compatible RGB, and apply the future verified Multilingual-CLIP processor |
| Image preprocessing requirements | Paper supplies no size/window/channel recipe; source images are already rendered and heterogeneous | Paper supplies no size/channel/view recipe; normalized raster PNGs are already provided | Ambiguous in paper | Define only checkpoint-required preprocessing later and document it; CT DICOM windowing is impossible because raw DICOM is absent |
| Metadata requirements | Core filename/report/label fields exist; no report/study/patient ID | Core report UID, image filename, and projection exist | Core needs met | Retain auxiliary metadata; do not make COVID, MeSH, or Problems mandatory model inputs because the paper does not |
| Text fields required by methodology | English report is present for every CSV row | Impression+findings both present in 3,331 reports; at least one present in 3,826 | COV yes after conflict cleanup; IU partial | Explicitly handle IU missing sections; do not invent missing report text |
| Prompt-bank availability | At least ~571 clean reports would nominally fall in an 80% split, enough for 500 prompts, but exact split/conflict policy is absent | A nominal 70% strict-target training set exceeds 1,000 reports | Numerically yes | Sample only from persisted training IDs; fix seed and prevent validation/test report leakage |
| Train/validation/test requirements | No existing split; paper says random 8:1:1 | No existing split; paper says random 70:10:20 | Not directly ready | Generate and persist deterministic manifests at image-key level for COV-CTR and UID level for IU-Xray; paper seed/indices remain unavailable |
| Count/version consistency | Actual counts and class balance do not match paper | Actual report and mapped-image counts do not match paper | No for exact experimental reproduction | Report results as applying to the supplied archive version; do not claim exact paper split reproduction |

## 6. Required Preprocessing

No preprocessing in this section has been executed. These are the required data preparation operations that must precede model input.

### 6.1 COV-CTR

#### A. Image preprocessing

- Decode from file signature rather than extension because 34 `.png` names contain JPEG data.
- Convert grayscale, RGB, and RGBA inputs to a consistent checkpoint-compatible RGB representation; alpha handling must be deterministic.
- Apply the exact resize/crop/interpolation and pixel normalization required by the verified Multilingual-CLIP image processor. The paper itself does not state these values.
- Preserve aspect-ratio/cropping decisions in configuration because dimensions vary widely.
- Do not apply CT DICOM windowing: the archive contains only 8-bit rendered raster images, not raw CT/DICOM data.
- Preserve source annotations/circles unless a later documented reproduction decision says otherwise; silently inpainting or removing them would alter the dataset.

#### B. Text/report preprocessing

- Parse `reports_ZH_EN.csv` with UTF-8-BOM and proper quoted multiline CSV handling.
- Strip the single trailing LF from every `reports_En` value while preserving internal clinical punctuation/content.
- Use `reports_En` as the English target required by the paper. Do not silently append the Chinese impression or translate other fields.
- Retain raw and processed forms separately for auditability.
- Defer tokenizer-specific truncation, special tokens, and maximum length because the paper does not specify them.

#### C. Image-report pairing

- Join `image_id` to the exact image basename.
- Quarantine the 26 unreferenced images from report-generation samples.
- Flag the six duplicate image IDs and require an explicit resolution before splitting; never let duplicate rows for the same pixels cross splits.
- Do not infer multi-image studies from percent suffixes, stems, or identical report strings without supporting metadata.

#### D. Metadata processing

- Preserve `findings`, `terminologies`, `COVID`, `impression`, and `reports_En` even though only the English report is required as target by the paper.
- Parse `COVID` as a binary auxiliary label and validate consistency; do not feed it to ZeroMRG as an input unless the methodology later explicitly calls for it.
- Create stable internal annotation-row IDs because the archive has no report ID, while retaining the original `image_id` unchanged.
- Record image format signature, dimensions, channels, mapping status, and duplicate-conflict status in a processed manifest.

#### E. Dataset splitting

- Split only after resolving duplicate image keys.
- Follow the paper’s 8:1:1 ratio and persist exact IDs and a reproducible seed.
- Keep all records for the same physical image in one subset.
- Patient/study-group leakage cannot be fully controlled because those IDs are absent; this limitation must be reported.
- Draw the 500 prompts only from the final training-report pool.

### 6.2 IU-Xray

#### A. Image preprocessing

- Decode the `.dcm.png` files as PNG, not DICOM.
- Convert 8-bit grayscale images to the checkpoint-compatible RGB channel layout.
- Apply the exact verified Multilingual-CLIP resize/crop/interpolation and normalization later; retain original size metadata.
- Preserve projection labels and all views. Do not silently flatten a multi-view study into independent report duplicates or select only frontal views.
- Flag the three byte-identical image pairs and exclude the four metadata-unlisted images unless an explicit inclusion/mapping policy is documented.

#### B. Text/report preprocessing

- Parse `indiana_reports.csv` as UTF-8 CSV.
- Construct the paper target as `impression + findings`, following the order in the paper’s phrase “concatenation of impression and findings.” The separator and missing-section policy are not stated and must be explicit in the later plan.
- Never synthesize missing findings/impressions. Preserve nullness and create availability flags.
- Preserve `XXXX` de-identification tokens; replacing them with invented content is prohibited.
- Retain raw sections and a separately processed target.
- Defer tokenization/truncation/special-token decisions because the decoder specification is absent from the paper.

#### C. Image-report pairing

- Join report to projection rows by `uid`, then projection filename to its exact physical basename.
- Treat one UID as one study/report sample with a variable-length list of 1–5 images and parallel projection labels.
- Preserve image order deterministically, but do not claim an archive-defined clinical ordering beyond the projection records.
- Keep the four unlisted files outside model samples until their intended projection/mapping is established.
- Flag byte-identical image pairs in the manifest rather than deleting raw files.

#### D. Metadata processing

- Parse semicolon-delimited `MeSH` and `Problems` into lists while preserving the original strings.
- Retain `image` exam description, `indication`, `comparison`, findings, impression, and projection labels.
- Distinguish blank CSV values from textual sentinels such as `None.` and `None available`.
- Record section-availability, de-identification-placeholder, view-count, and duplicate-image flags.
- Do not promote MeSH/Problems to mandatory model inputs; the paper’s ZeroMRG pathway requires report text and images, not labels.

#### E. Dataset splitting

- Split at `uid` level so no views belonging to one report cross train/validation/test.
- Follow the paper’s 70:10:20 ratio and persist exact UID manifests and a reproducible seed.
- Decide and document whether strict complete-target reports only or a missing-section policy defines the split population; the paper does not say.
- Draw the 1,000 prompts exclusively from the finalized training-report set.
- Patient-level grouping cannot be guaranteed because the files expose a study/report UID but no confirmed patient identifier.

## 7. Dataset Loader Requirements

No loader is implemented here. The eventual loader should preserve dataset-native structure and expose enough metadata to audit every sample.

### 7.1 Common conceptual outputs

For image-conditioned or evaluation samples, a common conceptual record should contain:

```text
images                 one tensor or a variable-length list of tensors
image_ids              list of original basenames
view_labels            list aligned with images; unknown where unavailable
study_id               stable study/report grouping key, or explicit null
report_id              original report key, or stable internal row ID
raw_report             exact source target text/sections
processed_report       deterministic model-facing target text
findings               source findings field, if available
impression             source impression field, if available
language               "en"
modality               "CT" or "X-ray"
auxiliary_metadata     labels/terminology/indication/comparison as applicable
validity_flags         missing-section, conflict, unlisted, and duplicate flags
```

Text-only auto-encoding/prompt-bank access should be able to return report records without loading image pixels. Image inference should be able to load image records while still retaining the reference only for held-out evaluation. These are access modes over the same manifests, not combined datasets.

Variable image counts require a later collate policy that can return image masks and aligned view metadata. The dataset layer should preserve all views; how the paper’s singular image encoder consumes them is a separate unresolved methodology decision, not something the loader should hide.

### 7.2 COV-CTR conceptual sample

| Field | Required value |
|---|---|
| `images` | One decoded CT raster for a resolved annotation sample |
| `image_ids` | One original `image_id`/basename |
| `view_labels` | Unknown/null; no view metadata exists |
| `study_id` | Unknown/null; must not be guessed from filename |
| `report_id` | Stable internal annotation-row ID plus original row position |
| `raw_report` | Raw `reports_En` including provenance; processed value stored separately |
| `processed_report` | Trimmed English report |
| `findings` | Chinese `findings` retained as metadata, not substituted for English target |
| `impression` | Chinese `impression` retained as metadata |
| `terminology` | Chinese `terminologies` |
| `labels` | `COVID` binary label |
| Other | Image signature/shape/channel metadata and duplicate-conflict status |

### 7.3 IU-Xray conceptual sample

| Field | Required value |
|---|---|
| `images` | List of 1–5 decoded X-ray tensors for one `uid` |
| `image_ids` | Projection-table filenames aligned with the image list |
| `view_labels` | `Frontal`/`Lateral` aligned with image IDs |
| `study_id` | `uid` |
| `report_id` | `uid` |
| `raw_report` | Raw findings and impression retained separately |
| `processed_report` | Explicitly constructed impression+findings target |
| `findings` | Original `findings`, nullable |
| `impression` | Original `impression`, nullable |
| `terminology` | Original and parsed `MeSH` |
| `labels` | Original and parsed `Problems` |
| Other | `image` description, `indication`, `comparison`, section flags, duplicate flags |

## 8. Problems / Missing Information

Severity denotes impact on a faithful implementation using the supplied data.

### CRITICAL

1. **COV-CTR paper/archive population mismatch.** The paper’s 728 images and 349/379 class counts match none of the physical, row-level, or unique-key counts in the supplied archive. Exact paper splits and results cannot be reproduced from this version.
2. **COV-CTR conflicting image keys.** Six physical images each have two non-identical annotation records. Using all rows creates repeated pixels with differing targets and allows leakage; choosing one target requires an explicit, auditable rule absent from the data/paper.
3. **COV-CTR lacks report, study, and patient IDs.** It is impossible to establish whether filename-related images belong to the same case/report or to guarantee study/patient-disjoint splits. This prevents confident reproduction of any hidden grouping used by the authors.
4. **IU-Xray paper/archive report mismatch.** The paper states 3,955 reports, while the supplied report and projection tables expose 3,851 UIDs. Exact paper split membership and sample population are unavailable.
5. **IU-Xray multi-image input is unspecified by the paper.** 3,405 of 3,851 reports have multiple images, but the architecture is written for singular \(I\) and gives no view fusion/selection method. Silently duplicating reports per view or selecting one view would change the effective methodology.
6. **No existing splits or reproducibility seed.** Both paper evaluations use random splits, but neither archive supplies split manifests and the paper supplies no seed. Reported results cannot be tied to the same test population.

### MODERATE

1. **COV-CTR has 26 unannotated images.** They cannot be used for paired generation/evaluation or text prompting without external information.
2. **COV-CTR image representation is highly heterogeneous.** Mixed true formats, mislabeled extensions, color modes, resolutions, windows, crops, and publication annotations require deterministic normalization and may affect pretrained encoder behavior.
3. **IU-Xray missing target sections.** Only 3,331 reports contain both findings and impression; 520 lack at least one, including 25 lacking both. The paper does not give an exclusion or concatenation policy for these cases.
4. **IU-Xray has four physical images absent from projection metadata.** Filename-derived UIDs exist, but projection/inclusion cannot be inferred safely.
5. **IU-Xray contains three byte-identical image pairs under different names.** Without flags, repeated pixels could distort per-image handling or be mistaken for distinct evidence.
6. **Paper image preprocessing is absent.** Neither dataset can be transformed faithfully into the Multilingual-CLIP encoder without relying on the checkpoint’s external processor specification and documenting that implementation assumption.
7. **Potential split leakage cannot be fully assessed.** IU-Xray has study UIDs but no confirmed patient ID; COV-CTR lacks both. Random splitting may place related cases across subsets.
8. **COV-CTR naming/scope inconsistency in repository instructions.** `AGENTS.md` names COV-CT paths, while the current task and actual data supply COV-CTR. Downstream paths/configuration must not conflate these distinct datasets.

### MINOR

1. **COV-CTR English reports contain a trailing LF in every record.** This is deterministic to strip but breaks naive physical-line parsing and changes raw string equality if ignored.
2. **COV-CTR includes `.DS_Store`.** It must be ignored during image enumeration.
3. **IU-Xray `.dcm.png` names are potentially misleading.** The files are PNG rasters, not DICOM; decoders must use content/terminal extension.
4. **IU-Xray contains repeated full report content across UIDs.** There are 43 identical-content groups excluding UID (117 records). These may be valid template reuse and should be flagged, not automatically deleted.
5. **IU-Xray contains pervasive `XXXX` redaction tokens and textual null sentinels.** They are valid source text artifacts that must be distinguished from missing CSV fields and preserved unless a documented preprocessing rule says otherwise.

## 9. Final Compatibility Assessment

### COV-CTR — PARTIALLY COMPATIBLE

COV-CTR supplies the required English medical report text and rendered CT images, and at least 714 clean one-image/one-report pairs are identifiable. It also contains enough reports for the paper’s 500-prompt setting under a nominal 8:1:1 split. However, it is only **PARTIALLY COMPATIBLE** with faithful reproduction because its counts/class balance differ materially from the paper, six image keys have conflicting targets, 26 images lack annotations, and there is no report/study/patient identifier from which to recover grouping or reproduce the authors’ split. The method can be applied only after explicit preprocessing and documented conflict/split assumptions; it cannot be claimed as an exact reproduction of the paper’s COV-CTR population.

### IU-Xray — READY AFTER PREPROCESSING

IU-Xray has a strong explicit UID-based report-to-projection mapping: every report has at least one mapped image and every projection reference resolves. Its English findings/impression structure directly supports the paper’s target construction, and it has ample reports for the 1,000-prompt setting. It is **READY AFTER PREPROCESSING**, provided preprocessing is study-aware, missing sections and four unlisted images are handled explicitly, all views are preserved in the manifest, and no view-fusion choice is hidden in the loader. The supplied version’s 3,851 reports do not match the paper’s 3,955, so exact paper result reproduction remains unavailable even though the dataset is structurally suitable for implementing the methodology.
