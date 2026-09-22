# Datasets

[← DS3dRNA](../README.md) · [Energy tensors](../Energy/README.md) · [Citations](../CITATIONS.md)

DS3dRNA-associated datasets are hosted separately to keep the source repository focused and within practical clone sizes:

[Open the DS3dRNA dataset folder on Google Drive](https://drive.google.com/drive/folders/1EgsNwqr3uatONFZ98olXPeVfSu-aLiUW)

## Folder contents

The shared folder currently contains the following downloadable archives, reports, and index files:

| File | Contents |
| --- | --- |
| `3260_training.zip` | Deduplicated DS3dRNA training structures (3,260 structures). |
| `AlphaFold3_results_All.zip` | Collected AlphaFold 3 structure-validation results used in the accompanying analyses. |
| `CASP17_P20_kissing-multiloop.zip` | CASP17 P20 kissing-multiloop target materials and associated results. |
| `clean_pool_6968.zip` | Cleaned structural candidate pool containing 6,968 representatives. |
| `ssDNA_cif.zip` | Single-stranded DNA structures in CIF/mmCIF format used as the training collection for constructing the DNA energy potential. |
| [`Supplementary_T25_T83_100run_profiles.pdf`](https://drive.google.com/file/d/1a6-9uhurgc0P3cJrapgKecYKLQgzEDsV/view) | Supplementary design-run profiles for T25 and T83: 100 independent runs per target and design mode, with energy profiles, Recovery/MacroF1 trends, and Spearman correlation annotations (324 pages). |
| `T11.zip` | Multi-state benchmark: 11 target clusters comprising 168 conformers in total. |
| `T25.zip` | RNA-only inverse-folding/self-consistency benchmark containing 25 targets. |
| `T83_molecule_report_vs_train.xlsx` | Per-molecule comparison report between the T83 benchmark and the combined training collection. |
| `T83.zip` | Nonredundant benchmark set containing 83 RNA molecules. |
| `Test_pdb_id_all_methods_collection_999.txt` | PDB identifiers in the combined all-method test collection (999 identifiers). |
| `Train_pdb_id_all_methods_collection_6414.txt` | PDB identifiers in the combined all-method training collection (6,414 identifiers). |

The descriptions above follow the current DS3dRNA method manuscript and the folder snapshot documented for this release. Consult metadata packaged inside each download for the definitive target list, provenance, and any archive-specific notes.

`ssDNA_cif.zip` records provenance for the DNA energy tensors distributed under `Energy/DNA/`. It is a training-data archive; the public `DS3dRNA.py` design and ranking interfaces currently accept prepared PDB scaffolds rather than CIF archives.

## Supplementary design-run profiles

[View or download `Supplementary_T25_T83_100run_profiles.pdf`](https://drive.google.com/file/d/1a6-9uhurgc0P3cJrapgKecYKLQgzEDsV/view) from the shared Google Drive folder.

The report covers 25 T25 targets and 83 T83 targets in each of three design modes: **DS3dRNA**, **DS3dRNA_SS**, and **DS3dRNA_auto**. Each target–mode combination uses **100 independent design runs**, giving **324 two-panel pages** in total. PDF bookmarks provide access by dataset, mode, and target.

- **Panel a:** energy versus recorded step for the 100 runs, with a median trend and interquartile band summarizing run-level step-bin means.
- **Panel b:** Recovery and MacroF1 versus energy, summarized using run-level energy-bin means, with median trends and interquartile bands. Annotations give the corresponding energy–Recovery and energy–MacroF1 Spearman coefficients, calculated from all unbinned retained records pooled across the 100 runs. Negative coefficients indicate that lower energy is associated with higher sequence recovery or MacroF1.

Dataset users should record the accessed file names and versions in their methods. Cite both DS3dRNA and TriRNASP as described in [CITATIONS.md](../CITATIONS.md), and retain any dataset-specific provenance or license files included with a download.

The external folder's availability and access permissions are managed independently of this repository.
