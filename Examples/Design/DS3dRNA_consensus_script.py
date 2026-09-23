#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert DS3dRNA summary CSV files to AlphaFold 3 input JSON.

Purpose
-------
The conversion pipeline is:

    DS3dRNA summary CSV
    -> remove '&' from sequence columns
    -> remove rows with E_fine - E_fine_min > 10000
    -> count-weighted profile consensus, default max_run=5 per chain
    -> also output the E_fine-minimum sequence as AF3 JSON
    -> split chains only by filename boundaries like _A:13_B:41
    -> write AF3 JSON

This post-processing utility does not include:
    - secondary-structure constraints
    - ss_auto / PDB parser
    - terminal C-G sealing
    - SS-related deltaE filtering
    - C decoder
    - Ensemble_* reading

Example
-------
python DS3dRNA_consensus_script.py \
    -i ./ \
    -o ./JSON_countConsensus_maxRun5 \
    --count \
    --max_run 5 \
    --seed 1 \
    --method_prefix DS3dRNA_CC5
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


AF3_DIALECT = "alphafold3"
AF3_VERSION = 3
DEFAULT_MODEL_SEEDS = [1]
DEFAULT_INPUT_DIR = "."
DEFAULT_OUTPUT_DIR = "./JSON_countConsensus_maxRun5_Emin"

BASE_ORDER = "AUCGN"
VALID_BASES = set(BASE_ORDER)
DNA_TO_RNA = {"T": "U"}

SEQ_CANDIDATES = (
    "Designed_seq",
    "designed_seq",
    "DesignedSeq",
    "designedseq",
    "sequence",
    "Sequence",
    "seq",
)

COUNT_CANDIDATES = (
    "count",
    "Count",
    "n",
    "N",
    "freq",
    "frequency",
)

CONSENSUS_CANDIDATES = (
    "consensus",
    "consensus_sequence",
    "consensussequence",
    "consensus_seguence",
    "consensusseguence",
)

ENERGY_CANDIDATES = (
    "E_fine(kBT)",
    "E_fine_kBT",
    "E_fine",
    "Efine",
    "fine",
)

# Exclude rows for which E_fine - E_fine_min exceeds this threshold.
# Set to None to disable the filter.
DELTA_E_MAX = 10000.0


def sanitize_seq(seq: str) -> str:
    """
    Clean RNA sequence:
      - remove whitespace
      - remove '&' chain separators from CSV sequence columns
      - uppercase
      - T -> U
      - allow only A/U/C/G/N

    Chain splitting is NOT inferred from '&'. Chain boundaries come only from filename.
    """
    seq = re.sub(r"\s+", "", str(seq).upper())
    seq = seq.replace("&", "")
    seq = "".join(DNA_TO_RNA.get(ch, ch) for ch in seq)

    bad = sorted(set(ch for ch in seq if ch not in VALID_BASES))
    if bad:
        raise ValueError(f"Sequence contains invalid characters after removing &: {bad}")

    return seq


def _norm_key(x) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _get_field(row: Dict[str, str], field_map: Dict[str, str], candidates) -> Optional[str]:
    for c in candidates:
        k = _norm_key(c)
        if k in field_map:
            return row.get(field_map[k])
    return None


def parse_chain_spec_from_stem(stem: str) -> List[Tuple[str, int]]:
    """
    Parse cumulative chain-end positions from filename stem.

    Example:
        1br3_seqID_0.80_A:13_B:41
    returns:
        [("A", 13), ("B", 41)]

    Meaning:
        A = seq[0:13]
        B = seq[13:41]
    """
    pairs = re.findall(r"_([A-Za-z0-9]+):(\d+)", stem)
    if not pairs:
        raise ValueError(f"No chain specification found in filename: {stem}")

    out = [(cid, int(end)) for cid, end in pairs]
    prev = 0
    for cid, end in out:
        if end <= prev:
            raise ValueError(f"Non-increasing chain boundary in filename: {stem}")
        prev = end
    return out


def split_sequence_by_chain_ranges(seq: str, chain_specs: List[Tuple[str, int]]) -> OrderedDict:
    if not chain_specs:
        raise ValueError("Empty chain specs")

    expected_len = chain_specs[-1][1]
    if len(seq) != expected_len:
        raise ValueError(
            f"Sequence length ({len(seq)}) != final chain end ({expected_len})"
        )

    out = OrderedDict()
    prev_end = 0
    for chain_id, end_pos in chain_specs:
        out[chain_id] = seq[prev_end:end_pos]
        prev_end = end_pos
    return out


def _parse_float_or_default(value, default: float = 1.0) -> float:
    if value is None or not str(value).strip():
        return default
    return float(str(value).strip())


def _parse_float_or_none(value) -> Optional[float]:
    if value is None or not str(value).strip():
        return None
    try:
        v = float(str(value).strip())
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def choose_base_with_maxrun(profile_i: Dict[str, float], last_base: Optional[str], run_len: int, max_run: int) -> str:
    """
    Choose the highest-weight base while respecting homopolymer max_run.

    Tie-break: A > U > C > G > N.
    If all A/U/C/G/N choices are blocked, fall back to the ordinary highest base.
    """
    allowed = []
    for b in BASE_ORDER:
        if max_run and max_run > 0 and last_base == b and run_len >= max_run:
            continue
        allowed.append(b)

    if not allowed:
        allowed = list(BASE_ORDER)

    return max(
        allowed,
        key=lambda b: (profile_i.get(b, 0.0), -BASE_ORDER.index(b)),
    )


def build_count_consensus(
    seq_weight_pairs: List[Tuple[str, float]],
    max_run: int = 0,
    chain_specs: Optional[List[Tuple[str, int]]] = None,
) -> str:
    """
    Build count-weighted profile consensus.

    If max_run > 0, enforce a simple homopolymer cap during left-to-right decoding.
    Chain specifications use cumulative end positions parsed from the filename.
    Reset the homopolymer count at each chain boundary so independent RNA
    chains do not restrict one another. Without chain_specs, treat the input
    as one chain. This is intentionally simple and deterministic.
    """
    if not seq_weight_pairs:
        raise ValueError("No sequence-weight pairs for consensus")

    expected_len = len(seq_weight_pairs[0][0])
    chain_starts = set()
    if chain_specs is not None:
        if not chain_specs or chain_specs[-1][1] != expected_len:
            raise ValueError("Chain specifications must end at the sequence length")
        previous_end = 0
        for _, end in chain_specs:
            if end <= previous_end:
                raise ValueError("Chain boundaries must be strictly increasing")
            chain_starts.add(previous_end)
            previous_end = end

    profile = [{b: 0.0 for b in BASE_ORDER} for _ in range(expected_len)]

    for seq, weight in seq_weight_pairs:
        if len(seq) != expected_len:
            raise ValueError(f"Inconsistent sequence length: len={len(seq)}, expected len={expected_len}")
        for i, b in enumerate(seq):
            profile[i][b] = profile[i].get(b, 0.0) + float(weight)

    out = []
    last_base = None
    run_len = 0

    for i in range(expected_len):
        if i in chain_starts:
            last_base = None
            run_len = 0
        b = choose_base_with_maxrun(profile[i], last_base, run_len, max_run)
        out.append(b)

        if b == last_base:
            run_len += 1
        else:
            last_base = b
            run_len = 1

    return "".join(out)


def read_seq_weight_pairs(csv_path: Path, use_count: bool) -> Tuple[List[Tuple[str, float]], Dict[str, object]]:
    """
    Read sequence rows from summary CSV and apply the bad-design filter:

        keep row only if E_fine - E_fine_min <= DELTA_E_MAX

    Priority:
      1. Designed_seq / sequence / seq columns
      2. fallback to explicit consensus column as one sequence with weight 1

    This script does not read Ensemble_* files.
    """
    filter_info = {
        "E_fine_min": "nan",
        "E_min_seq": "",
        "E_min_row_index": "nan",
        "rows_before_filter": 0,
        "rows_after_filter": 0,
        "rows_deltaE_filtered": 0,
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {csv_path.name}")

        field_map = {
            _norm_key(x): x
            for x in reader.fieldnames
            if x is not None and str(x).strip()
        }

        has_seq_col = any(_norm_key(c) in field_map for c in SEQ_CANDIDATES)
        has_consensus_col = any(_norm_key(c) in field_map for c in CONSENSUS_CANDIDATES)

        rows = list(reader)
        filter_info["rows_before_filter"] = len(rows)

        if has_seq_col:
            # Find the minimum Fine energy within this CSV.
            energy_values = []
            for row in rows:
                escore = _parse_float_or_none(_get_field(row, field_map, ENERGY_CANDIDATES))
                if escore is not None:
                    energy_values.append(escore)

            e_fine_min = min(energy_values) if energy_values else None
            if e_fine_min is not None:
                filter_info["E_fine_min"] = e_fine_min

                # Preserve the first minimum-energy sequence for the
                # additional E_min AlphaFold 3 JSON output.
                for row_idx, row in enumerate(rows, start=2):
                    escore = _parse_float_or_none(_get_field(row, field_map, ENERGY_CANDIDATES))
                    if escore != e_fine_min:
                        continue
                    seq_val = _get_field(row, field_map, SEQ_CANDIDATES)
                    if seq_val is None or not str(seq_val).strip():
                        continue
                    filter_info["E_min_seq"] = sanitize_seq(seq_val)
                    filter_info["E_min_row_index"] = row_idx
                    break

            seq_weight_pairs = []
            expected_len = None

            for idx, row in enumerate(rows, start=2):
                escore = _parse_float_or_none(_get_field(row, field_map, ENERGY_CANDIDATES))

                # Exclude high-energy rows from the consensus profile. If no
                # valid Fine energy is present, no delta-E filter is applied.
                if DELTA_E_MAX is not None and e_fine_min is not None:
                    if escore is None or (escore - e_fine_min) > DELTA_E_MAX:
                        filter_info["rows_deltaE_filtered"] += 1
                        continue

                seq_val = _get_field(row, field_map, SEQ_CANDIDATES)
                if seq_val is None or not str(seq_val).strip():
                    continue

                seq = sanitize_seq(seq_val)

                if expected_len is None:
                    expected_len = len(seq)
                elif len(seq) != expected_len:
                    raise ValueError(
                        f"Inconsistent sequence length in {csv_path.name}: "
                        f"row {idx} has len={len(seq)}, expected len={expected_len}"
                    )

                weight = 1.0
                if use_count:
                    count_val = _get_field(row, field_map, COUNT_CANDIDATES)
                    try:
                        weight = _parse_float_or_default(count_val, 1.0)
                    except ValueError:
                        raise ValueError(
                            f"Invalid count value in {csv_path.name}: row {idx}, count={count_val!r}"
                        )

                if weight <= 0:
                    continue
                seq_weight_pairs.append((seq, weight))

            filter_info["rows_after_filter"] = len(seq_weight_pairs)

            if not seq_weight_pairs:
                raise ValueError(
                    f"No valid Designed_seq/sequence rows in {csv_path.name} "
                    f"after deltaE filtering"
                )
            return seq_weight_pairs, filter_info

        if has_consensus_col:
            # Consensus-only files are already aggregated and cannot be
            # filtered row-by-row using Fine energy.
            for row in rows:
                val = _get_field(row, field_map, CONSENSUS_CANDIDATES)
                if val is not None and str(val).strip():
                    filter_info["rows_after_filter"] = 1
                    return [(sanitize_seq(val), 1.0)], filter_info

    # Fall back to the key-value CSV format.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = _norm_key(row[0])
            if key in {"consensus_sequence", "consensussequence", "consensus"}:
                if len(row) < 2 or not str(row[1]).strip():
                    raise ValueError(f"Found consensus row but no value in {csv_path.name}")
                filter_info["rows_after_filter"] = 1
                return [(sanitize_seq(row[1]), 1.0)], filter_info

    raise ValueError(f"Could not read sequence rows in {csv_path.name}")

def discover_valid_csv_files(root_dir: Path):
    valid_files = []
    skipped_files = []

    for csv_path in sorted(root_dir.rglob("*.csv")):
        name_lower = csv_path.name.lower()

        if name_lower.startswith("traj") or name_lower.startswith("ensemble"):
            skipped_files.append((csv_path, "excluded by filename pattern"))
            continue

        try:
            parse_chain_spec_from_stem(csv_path.stem)

            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("No CSV header found")

                field_map = {
                    _norm_key(x): x
                    for x in reader.fieldnames
                    if x is not None and str(x).strip()
                }

                if not (
                    any(_norm_key(c) in field_map for c in SEQ_CANDIDATES)
                    or any(_norm_key(c) in field_map for c in CONSENSUS_CANDIDATES)
                ):
                    raise ValueError("No sequence/consensus column found")

        except Exception as e:
            skipped_files.append((csv_path, str(e)))
            continue

        valid_files.append(csv_path)

    return valid_files, skipped_files


def build_af3_json(name: str, chain_to_seq: OrderedDict, model_seeds=None, method_prefix="DS3dRNA"):
    if model_seeds is None:
        model_seeds = DEFAULT_MODEL_SEEDS

    full_name = f"{method_prefix}_{name}" if method_prefix else name

    sequences = []
    for chain_id, seq in chain_to_seq.items():
        sequences.append({
            "rna": {
                "id": chain_id,
                "sequence": seq,
            }
        })

    return {
        "name": full_name,
        "modelSeeds": model_seeds,
        "sequences": sequences,
        "dialect": AF3_DIALECT,
        "version": AF3_VERSION,
    }


def write_one_json(seq: str, csv_path: Path, out_dir: Path, model_seeds, method_prefix: str, suffix: str):
    chain_specs = parse_chain_spec_from_stem(csv_path.stem)
    chain_to_seq = split_sequence_by_chain_ranges(seq, chain_specs)

    json_name_stem = f"{csv_path.stem}_{suffix}"
    af3_data = build_af3_json(
        name=json_name_stem,
        chain_to_seq=chain_to_seq,
        model_seeds=model_seeds,
        method_prefix=method_prefix,
    )

    out_path = out_dir / f"{json_name_stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(af3_data, f, ensure_ascii=False, indent=2)

    return {
        "json": str(out_path),
        "seq_len": len(seq),
        "chain_lens": {k: len(v) for k, v in chain_to_seq.items()},
        "af3_name": af3_data["name"],
        "seq": seq,
    }


def convert_one_csv(
    csv_path: Path,
    consensus_out_dir: Path,
    emin_out_dir: Path,
    *,
    model_seeds,
    method_prefix: str,
    use_count: bool,
    max_run: int,
):
    seq_weight_pairs, filter_info = read_seq_weight_pairs(csv_path, use_count=use_count)

    chain_specs = parse_chain_spec_from_stem(csv_path.stem)
    expected_len = chain_specs[-1][1]
    observed_len = len(seq_weight_pairs[0][0])
    if observed_len != expected_len:
        raise ValueError(
            f"Sequence length ({observed_len}) != final chain end from filename ({expected_len})"
        )

    consensus = build_count_consensus(
        seq_weight_pairs, max_run=max_run, chain_specs=chain_specs
    )

    consensus_info = write_one_json(
        seq=consensus,
        csv_path=csv_path,
        out_dir=consensus_out_dir,
        model_seeds=model_seeds,
        method_prefix=method_prefix,
        suffix="consensus",
    )

    e_min_seq = filter_info.get("E_min_seq") or ""
    e_min_info = {"json": "", "af3_name": "", "seq": "", "seq_len": "nan"}
    if e_min_seq:
        if len(e_min_seq) != expected_len:
            raise ValueError(
                f"E_min sequence length ({len(e_min_seq)}) != final chain end from filename ({expected_len})"
            )
        e_min_info = write_one_json(
            seq=e_min_seq,
            csv_path=csv_path,
            out_dir=emin_out_dir,
            model_seeds=model_seeds,
            method_prefix=method_prefix,
            suffix="E_min",
        )

    return {
        "csv": str(csv_path),
        "consensus_json": consensus_info["json"],
        "E_min_json": e_min_info["json"],
        "consensus_len": consensus_info["seq_len"],
        "E_min_len": e_min_info["seq_len"],
        "chain_lens": consensus_info["chain_lens"],
        "n_rows_used": len(seq_weight_pairs),
        "E_fine_min": filter_info["E_fine_min"],
        "E_min_row_index": filter_info["E_min_row_index"],
        "rows_before_filter": filter_info["rows_before_filter"],
        "rows_after_filter": filter_info["rows_after_filter"],
        "rows_deltaE_filtered": filter_info["rows_deltaE_filtered"],
        "use_count": bool(use_count),
        "max_run": int(max_run),
        "consensus_af3_name": consensus_info["af3_name"],
        "E_min_af3_name": e_min_info["af3_name"],
        "consensus_seq": consensus_info["seq"],
        "E_min_seq": e_min_info["seq"],
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert DS3dRNA summary CSV files to AlphaFold 3 JSON using a "
            "count-weighted consensus, a Fine-energy filter, and an optional "
            "homopolymer run-length limit."
        ),
    )
    parser.add_argument("-i", "--input_dir", default=DEFAULT_INPUT_DIR,
                        help="Root directory to recursively search summary CSV files.")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root directory to write JSON outputs.")
    parser.add_argument("--seed", type=int, nargs="+", default=DEFAULT_MODEL_SEEDS,
                        help="modelSeeds list for AF3 JSON, e.g. --seed 1 2 3")
    parser.add_argument("--method_prefix", type=str, default="DS3dRNA_CC5",
                        help="Prefix added to AF3 JSON 'name'. Use empty string '' to disable.")
    parser.add_argument("--count", action="store_true",
                        help="Use count/freq/frequency column as profile weight. Without this flag, each row has weight 1.")
    parser.add_argument("--max_run", type=int, default=5,
                        help="Maximum allowed homopolymer run length. 0 means no homopolymer limit.")
    parser.add_argument("--show_skipped", action="store_true",
                        help="Print skipped CSV files and reasons.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    root_out_dir = Path(args.output_dir).resolve()

    if not in_dir.exists():
        print(f"[ERROR] input_dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(1)

    consensus_out_dir = root_out_dir / "consensus_json"
    emin_out_dir = root_out_dir / "E_min_json"
    consensus_out_dir.mkdir(parents=True, exist_ok=True)
    emin_out_dir.mkdir(parents=True, exist_ok=True)

    csv_files, skipped_files = discover_valid_csv_files(in_dir)

    print(f"[SCAN] root              = {in_dir}")
    print(f"[SCAN] valid summary csv = {len(csv_files)}")
    print(f"[SCAN] skipped csv files = {len(skipped_files)}")
    print(f"[OPT]  count weighted    = {args.count}")
    print(f"[OPT]  max_run           = {args.max_run if args.max_run else 'none'}")
    print(f"[OPT]  deltaE filter     = E_fine - E_fine_min <= {DELTA_E_MAX}")
    print(f"[OPT]  method_prefix     = {args.method_prefix}")

    if args.show_skipped and skipped_files:
        print("-" * 80)
        print("[SKIPPED]")
        for p, reason in skipped_files:
            try:
                rel_p = p.relative_to(in_dir)
            except ValueError:
                rel_p = p
            print(f"  {rel_p}: {reason}")

    if not csv_files:
        print(
            f"[ERROR] No valid summary CSV files found under: {in_dir}\n"
            f"        Rule: recursive *.csv, excluding traj*.csv and Ensemble*.csv,\n"
            f"        requiring filename chain boundaries like _A:13_B:41,\n"
            f"        and requiring sequence/consensus columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    ok = 0
    fail = 0
    manifest_rows = []

    print("-" * 80)

    for csv_path in csv_files:
        try:
            info = convert_one_csv(
                csv_path=csv_path,
                consensus_out_dir=consensus_out_dir,
                emin_out_dir=emin_out_dir,
                model_seeds=args.seed,
                method_prefix=args.method_prefix,
                use_count=args.count,
                max_run=args.max_run,
            )
            ok += 1
            manifest_rows.append(info)

            try:
                rel_csv = csv_path.relative_to(in_dir)
            except ValueError:
                rel_csv = csv_path

            print(
                f"[OK] {rel_csv}\n"
                f"     consensus -> {Path(info['consensus_json']).name} | "
                f"E_min -> {Path(info['E_min_json']).name if info['E_min_json'] else 'NA'} | "
                f"len={info['consensus_len']} | chains={info['chain_lens']} | "
                f"rows={info['n_rows_used']} | filtered={info['rows_deltaE_filtered']}"
            )

        except Exception as e:
            fail += 1
            try:
                rel_csv = csv_path.relative_to(in_dir)
            except ValueError:
                rel_csv = csv_path
            print(f"[FAIL] {rel_csv}: {e}", file=sys.stderr)

    manifest_path = root_out_dir / "manifest_count_consensus_maxrun.csv"
    if manifest_rows:
        fieldnames = [
            "csv",
            "consensus_json",
            "E_min_json",
            "consensus_len",
            "E_min_len",
            "chain_lens",
            "n_rows_used",
            "E_fine_min",
            "E_min_row_index",
            "rows_before_filter",
            "rows_after_filter",
            "rows_deltaE_filtered",
            "use_count",
            "max_run",
            "consensus_af3_name",
            "E_min_af3_name",
            "consensus_seq",
            "E_min_seq",
        ]
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("-" * 80)
    print(f"[DONE] success={ok} failed={fail}")
    print(f"[OUT] consensus_json = {consensus_out_dir}")
    print(f"[OUT] E_min_json     = {emin_out_dir}")
    if manifest_rows:
        print(f"[OUT] manifest       = {manifest_path}")


if __name__ == "__main__":
    main()
