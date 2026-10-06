# Quick start

[← DS3dRNA](../README.md) · [Inputs](INPUTS.md) · [Outputs](OUTPUTS.md)

Run all commands from the repository root after activating the DS3dRNA environment.

## 1. Single-state design

With automatic secondary-structure inference for thermodynamic screening:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --batch 10
```

With an explicit dot-bracket constraint:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb \
  --ss Examples/Design/inputs/8VY0.dbn \
  --batch 10
```

Important: in design mode, an explicit `--ss` value (including `--ss auto` and `--ss none`) activates the hard secondary-structure path and fixes `steps=10000`, `tail_steps=4000`, and `topE=10000`. Omitted `--ss` still supplies automatically inferred secondary structure to the local thermodynamic screen, but does not enable the hard base-pair penalty. `--ss none` explicitly supplies an all-unpaired, zero-contact DBN; see [Inputs and constraints](INPUTS.md#secondary-structure) for the complete distinction.

## 2. Reproduce a run

Exact replay requires the **same execution procedure**, not just the same seed number. Preserve the original inputs, constraints, design options, code and energy tensors, and software/hardware environment. Single-seed and multi-seed execution use different paths; a seed taken from a batch is not a standalone reproduction recipe. Matching the replay procedure is necessary; it does not guarantee bitwise identity across different hardware or software versions.

### Replay a batch-generated result

For a run originally generated with `--batch`, use its saved seed archive from `<target>_output/seed_batches/seed_batch_*.txt`. The console prints the exact path as `[INFO] Saved seed batch: ...`.

```bash
# Original run (example):
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --batch 10

# Replay: replace the placeholder with the archive saved by that run.
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb \
  --seed_batch "Examples/Design/inputs/8VY0_output/seed_batches/seed_batch_<timestamp>.txt"
```

The archive path above is a placeholder, not a literal shell command: substitute the actual filename before running it. Keep all other options identical to the original command.

- Keep the **complete original seed list in exactly the original order**. Do not sort, shuffle, deduplicate, add, or remove seeds.
- To reproduce one result from a multi-seed batch, replay its entire original archive and then inspect that seed's result. Extracting it and running `--seed xxx` does **not** reproduce the original batch execution.
- Preserve batch boundaries. A `--batch` request can produce several archives (for example, after OOM-driven chunk reduction); replay each original archive separately, in recorded round order. Do not concatenate archives or split one archive into smaller batches.
- Repeating `--batch 10` generates new seeds; it does not replay the previous run. The summary CSV's seed column is not a replacement for the original ordered archive.

### Replay a standalone single-seed result

If the original result was produced as a standalone single-seed run, reproduce it with the same standalone `--seed` command and all the original options:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --seed 363022884
```

Do not add that seed to a multi-seed batch and expect the standalone result to be reproduced. A saved archive containing only one seed remains a one-seed execution; the relevant distinction is the original execution's seed count and grouping, not merely the filename or flag.

These rules also apply to multi-state design: retain `-m`, the original conformations and their ordering, and the original `--ss`, `--frz`, `--mol`, sampling and other settings. In particular, do not interchange omitted `--ss`, `--ss auto`, and `--ss none`.

`--seed` and `--seed_batch` are mutually exclusive. When either is supplied, `--batch` is ignored. Preserve original outputs and replay using a separate copy of the inputs with the same contents and layout, because design outputs are written next to the input target.

## 3. Batch design

Pass a directory containing PDB targets:

```bash
python DS3dRNA.py path/to/pdb_directory --batch 10
```

For per-target DBN or frozen files, pass a directory through `--ss` or `--frz`. Files are matched by PDB basename; see [INPUTS.md](INPUTS.md).

## 4. Multi-state design

All PDB files in the multi-state folder must describe residue-matched conformations of the same molecule.

```bash
python DS3dRNA.py -m Examples/MultiState_Design/inputs \
  --ss Examples/MultiState_Design/inputs/ss.dbn \
  --frz Examples/MultiState_Design/inputs/Frz.fasta \
  --batch 10
```

## 5. Sequence ranking with DS3dRank

Single scaffold:

```bash
python DS3dRNA.py -rank \
  --str 'Examples/Seq_Rank/R1138_7PTL_A:720.pdb' \
  --fa Examples/Seq_Rank/candidates.fasta \
  --rank_out ranked_candidates.csv
```

Directory of independent scaffolds:

```bash
python DS3dRNA.py -rank --batch_rank \
  --str path/to/pdb_directory \
  --fa candidates.fasta \
  --rank_out rank_outputs
```

One sequence set scored jointly against a multi-state directory:

```bash
python DS3dRNA.py -rank --multi_rank \
  --str path/to/multi_state_directory \
  --fa candidates.fasta \
  --rank_out multi_state.rank.csv
```

Rank mode uses the Fine three-body energy only. Thermodynamic reranking is disabled; `--ss` is optional and acts only as a hard constraint filter.

## 6. RNA and DNA modes

RNA is the default. Select the DNA energy tensors and DNA thermodynamic backend with:

```bash
python DS3dRNA.py target.pdb --mol DNA --batch 10
```

DNA mode displays T rather than U and limits paired constraints to AT/TA/CG/GC.

The same selector applies to ranking:

```bash
python DS3dRNA.py -rank \
  --str target_DNA.pdb \
  --fa DNA_candidates.fasta \
  --mol DNA \
  --rank_out DNA_candidates.rank.csv
```

See [Inputs and constraints](INPUTS.md#rna-and-dna-molecule-modes) for the complete RNA/DNA comparison.

## 7. Results

Each target receives its own output directory containing a summary CSV and compressed ensemble/trajectory files. See [OUTPUTS.md](OUTPUTS.md) for field definitions and post-processing utilities.
