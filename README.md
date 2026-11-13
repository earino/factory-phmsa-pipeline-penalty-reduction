---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-bbc08f68dbbd24b5**

[Dataset website](https://earino.github.io/factory-phmsa-pipeline-penalty-reduction/versions/v-bbc08f68dbbd24b5/) · [GitHub release](https://github.com/earino/factory-phmsa-pipeline-penalty-reduction/releases/tag/v-bbc08f68dbbd24b5) · [Hugging Face](https://huggingface.co/datasets/earino/factory-phmsa-pipeline-penalty-reduction/tree/v-bbc08f68dbbd24b5) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: 04b70fa61c8b4b68fb11974080984ef075ff8470c5ee52379579266b5ccfac1a
<!-- /dataset-factory:links -->

# Will PHMSA's proposed pipeline-safety civil penalty be reduced or eliminated within 36 months?

Will a PHMSA Notice of Probable Violation with a proposed civil penalty be closed within 36 months with a lower assessed penalty than proposed?

Derived from U.S. Department of Transportation, Pipeline and Hazardous Materials Safety Administration (PHMSA). This is not its official product.

Prediction time: Notice_Sent_Date, before the operator response, hearing or any order.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6172268907563025,
    "brier": 0.18742755379536652
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.18849675706082494
  },
  "calendar_identity": {
    "auc": 0.6205882352941177,
    "brier": 0.1823742838599128
  }
}
```

## Limitations

- Current-vintage retrospective extract as of 2026-11-03; historical fields may have been updated after case closure, so this does not prove the exact feature set was operationally available to an earlier decision-maker.
- Right-censoring: cases not closed with a reduction within 36 months are coded 0, so later reductions are missed; closed cases with missing Assessed_Penalties are also coded 0 and may hide administrative eliminations recorded as blanks.
- Selection: only NOPVs with positive proposed penalties are eligible; warning letters, notices of amendment, corrective action orders, and compliance-only cases are excluded.
- Cases cluster by operator and sometimes share an inspection; no operator history or lagged outcome features are used, but evaluation should account for operator clustering.
- The data dictionary is dated 2023 and uses some field names that differ from the 2026 raw header; construction must follow the raw header and map semantics explicitly.
- No causal interpretation: inspection routes, regions, and case initiation reflect PHMSA enforcement selection and legal process, not random assignment.
- The U.S. public-domain basis is territorial; 17 U.S.C. § 105 does not establish a worldwide dedication, and reuse outside the United States may depend on local law. No third-party photo or document rights are included.
- Current-vintage extract as of 2026-11-03; historical records may have been revised after case closure, so this is retrospective research rather than a historical forecasting claim.
- Right-censoring: cases not closed with a reduction within 36 months are coded zero, so later reductions are missed.
- Closed cases with missing Assessed_Penalties are coded zero per the frozen task, not imputed.
- Only NOPVs with positive proposed penalties are eligible; warning letters, notices of amendment, corrective action orders and compliance-only cases are excluded.
- Cases cluster by operator and sometimes share an inspection; no operator history or lagged outcome features are used, and no corporate-hierarchy reconciliation is attempted.
- The data dictionary is dated 2023 and uses some field names that differ from the 2026 raw header; construction follows raw header aliases and fails with observed headers if a required field is absent.
- No full-row historical availability ledger exists; feature availability at historical notice dates is assumed from current field definitions and the dictionary.
- No causal interpretation: inspection routes, regions and case initiation reflect PHMSA enforcement selection and legal process, not random assignment.
- The U.S. public-domain basis is territorial; 17 U.S.C. 105 does not establish a worldwide dedication, and reuse outside the United States may depend on local law.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
