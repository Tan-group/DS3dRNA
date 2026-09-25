<div align="center">

# DS3dRNA: De novo design of functional RNAs through higher-order interactions

### De novo design of 3D RNAs via higher-order interactions

<p>
  <a href="https://tpformer.com/talks/ds3drna-2026"><img src="https://img.shields.io/badge/2026%20Conference%20Talk-View%20Presentation-0F766E?style=for-the-badge" alt="View the 2026 DS3dRNA conference presentation"></a>
</p>

<p>
  <a href="https://tpformer.com/talks/ds3drna-2026"><strong>De Novo 3D RNA Design Using Higher-Order Interactions</strong></a><br>
  Oral presentation · 14th National Conference on Soft Matter and Biological Physics
</p>

<p>
  <a href="INSTALL.md"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="INSTALL.md"><img src="https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch with CUDA 12.8"></a>
  <a href="docs/INPUTS.md#rna-and-dna-molecule-modes"><img src="https://img.shields.io/badge/Molecules-RNA%20%7C%20DNA-16A085" alt="RNA and DNA"></a>
  <a href="INSTALL.md"><img src="https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black" alt="Linux"></a>
</p>

<p>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/Release-v1.0-4C78A8" alt="Release v1.0"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Academic%20%26%20Non--commercial-8E44AD" alt="Academic and non-commercial license"></a>
  <a href="CITATIONS.md"><img src="https://img.shields.io/badge/Cite-DS3dRNA%20%2B%20TriRNASP-2E86C1" alt="Cite DS3dRNA and TriRNASP"></a>
</p>

<p>
  <a href="docs/QUICKSTART.md">Quick start</a> ·
  <a href="docs/METHOD.md">Method</a> ·
  <a href="Examples/README.md">Examples</a> ·
  <a href="Datasets/README.md">Datasets</a> ·
  <a href="CITATIONS.md">Citations</a>
</p>

<img src="Pic/Figure1-1.png?v=a0f9b9e" alt="DS3dRNA workflow" width="920">

<p><strong>Fixed-backbone sequence design and high-throughput ranking for single-state and multi-state 3D RNA/DNA scaffolds.</strong></p>

</div>

DS3dRNA searches nucleotide sequence space for sequences compatible with one or more fixed three-dimensional nucleic-acid scaffolds. It combines coarse-grained three-body TriRNASP energies, local nearest-neighbor thermodynamic screening, optional secondary-structure and frozen-site constraints, and Monte Carlo sampling. The same public entry point provides **DS3dRank** for scoring externally generated FASTA candidates.

## At a glance

| 🧬 Design | 🔗 Constrain | ⚡ Rank |
| --- | --- | --- |
| Single target, PDB batch, or residue-matched multi-state ensemble | Automatic, explicit, or zero-contact secondary structure; optional frozen residues | Fine-energy scoring of FASTA candidates on single, batch, or multi-state scaffolds |
| RNA and DNA energy models | Pseudoknot-aware dot-bracket and chain breaks | CPU support and CUDA acceleration |

The model represents each residue with P, C4′, and N1/N9 sites, preserving backbone and base-anchor geometry without exposing complete nucleotide-specific heavy-atom patterns to the design procedure. See the [method overview](docs/METHOD.md) for the scientific workflow.

## Quick start

```bash
bash DS3dRNA_Installer.sh
conda activate DS3dRNA
python DS3dRNA.py --help
```

The installer first restores the four model tensors from `Energy.zip`, then prepares the Conda/PyTorch environment. Re-running it safely skips extraction when all tensors are already present and valid.

| Task | Command |
| --- | --- |
| RNA design (default) | `python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --ss none --batch 10` |
| DNA design ![Beta](https://img.shields.io/badge/BETA-blue) | `python DS3dRNA.py target_DNA.pdb --mol DNA --batch 10` |
| Sequence ranking | `python DS3dRNA.py -rank --str target.pdb --fa candidates.fasta` |

> [!NOTE]
> `--mol RNA` is the default; `--mol DNA` selects the DNA energy tensors, thermodynamic backend, pairing rules, and T-based output alphabet. See [RNA and DNA molecule modes](docs/INPUTS.md#rna-and-dna-molecule-modes).

<details>
<summary><strong>Multi-state design with explicit structural and frozen-site constraints</strong></summary>

```bash
python DS3dRNA.py -m Examples/MultiState_Design/inputs \
  --ss Examples/MultiState_Design/inputs/ss.dbn \
  --frz Examples/MultiState_Design/inputs/Frz.fasta \
  --batch 10
```

</details>

<details>
<summary><strong>Rank the bundled FASTA candidates</strong></summary>

```bash
python DS3dRNA.py -rank \
  --str 'Examples/Seq_Rank/R1138_7PTL_A:720.pdb' \
  --fa Examples/Seq_Rank/candidates.fasta \
  --rank_out ranked_candidates.csv
```

</details>

> [!IMPORTANT]
> Omitted `--ss`, `--ss auto`, and `--ss none` have deliberately different behavior. Read [Secondary structure](docs/INPUTS.md#secondary-structure) before production runs.

## Documentation

| Get started | Science and data | Reference and policy |
| --- | --- | --- |
| [Installation](INSTALL.md)<br>Environment and CUDA verification | [Method overview](docs/METHOD.md)<br>Scientific workflow | [Citation guide](CITATIONS.md)<br>DS3dRNA and dependencies |
| [Quick start](docs/QUICKSTART.md)<br>Design and ranking recipes | [Energy tensors](Energy/README.md)<br>RNA/DNA model assets | [Third-party notices](THIRD_PARTY_NOTICES.md)<br>Redistributed materials |
| [Inputs and constraints](docs/INPUTS.md)<br>RNA/DNA, PDB, DBN, FASTA, frozen masks | [Datasets](Datasets/README.md)<br>Training and benchmark collections | [License](LICENSE.txt)<br>Academic/non-commercial terms |
| [Outputs](docs/OUTPUTS.md)<br>Summary, ensemble, and trajectories | [Examples](Examples/README.md)<br>Reproducible workflows | [Optional DSSR utility](Tool/README.md)<br>Acquisition and citation |

The `Mode/` directory contains the internal computational core. Users should invoke [`DS3dRNA.py`](DS3dRNA.py) and follow the documented examples rather than calling developer modules directly.

## ✉️ Contact

**Prof. Zhi-jie Tan**  
Wuhan University  
Email: [zjtan@whu.edu.cn](mailto:zjtan@whu.edu.cn)

## Citation

If DS3dRNA contributes to published work, cite both the DS3dRNA manuscript and the TriRNASP energy model. Ready-to-copy entries are available in [CITATIONS.md](CITATIONS.md) and [CITATION.cff](CITATION.cff).

## License

DS3dRNA-authored materials are distributed under the academic and non-commercial terms in [LICENSE.txt](LICENSE.txt). The GNU LGPL 2.1 text is provided separately in [LICENSES/LGPL-2.1.txt](LICENSES/LGPL-2.1.txt); other third-party files remain governed by their respective terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Commercial licensing: `tongwei.tovi.yuan@gmail.com` · `tovi_yuen@whu.edu.cn`
