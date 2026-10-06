# Outputs

[← DS3dRNA](../README.md) · [Quick start](QUICKSTART.md) · [Inputs](INPUTS.md)

## Design output directory

For a target named `target.pdb`, DS3dRNA creates `target_output/` next to the input structure. A completed single-seed run normally contains:

- `target.csv`: one-row run summary;
- `Ensemble_<seed>.csv.zip`: compressed accepted-sequence ensemble;
- `trajTopE_<K>_<seed>.csv.zip`: compressed unique sequences with the lowest observed Fine energies across the trajectory;
- `fail.log`: created or updated when a run fails.

Multi-run and multi-state jobs add seed-specific or multi-state naming while retaining the same information categories.

## Seed archives for exact replay

Automatic `--batch` design writes `seed_batches/seed_batch_<timestamp>.txt` inside the target output directory, with a numeric suffix if filenames would collide. Each archive records the target, time, round and seed count in comment headers, followed by the seeds in execution order (one integer per line). Multiple completed sampling rounds can produce multiple archives.

Keep these files unchanged alongside the original command, inputs and environment details. Replay each archive with `--seed_batch`, preserving every seed, its order and the original batch boundaries. A seed extracted from a multi-seed archive and passed to `--seed` does not reproduce that batch execution; standalone single-seed results must likewise be replayed as standalone runs. See [Reproduce a run](QUICKSTART.md#2-reproduce-a-run) for commands and requirements.

## Summary fields

The summary CSV reports the designed sequence and diagnostics such as:

- `Designed_seq`: decoded design sequence;
- `E_fine(kBT)`: Fine TriRNASP energy;
- `recovery`: identity to the sequence encoded in the input scaffold, when available;
- `macroF1`: nucleotide-class-balanced agreement to the encoded sequence;
- `PPL`: profile perplexity diagnostic;
- `div_ensemble`: ensemble diversity diagnostic;
- `seed`: random seed used for the run.

Recovery and MacroF1 are diagnostics against input residue identities, not optimization targets and not experimental validation.

## Rank output

DS3dRank writes CSV rows containing at least:

- `sequence`;
- `energy(kBT)`;
- `recovery`;
- `macroF1`.

Rows are ordered by Fine energy, with lower energy ranked first.

## Post-processing

The design examples include two utilities:

- `DS3dRNA_consensus_script.py`: creates count-weighted consensus and minimum-Fine-energy AlphaFold 3 input JSON files;
- `summarize_weighted_by_count_deltaE10000.py`: aggregates result CSV metrics using `count` as a frequency weight and excludes rows more than 10,000 kBT above the within-file Fine-energy minimum.

These scripts are analysis helpers, not part of the sampler. See the [single-state](../Examples/Design/README.md) and [multi-state](../Examples/MultiState_Design/README.md) example guides.
