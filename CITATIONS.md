# Citation guide

[← DS3dRNA](README.md) · [Method](docs/METHOD.md) · [Datasets](Datasets/README.md)

## DS3dRNA

Please cite the DS3dRNA manuscript when using the design or ranking workflow:

> Tongwei Yuan, Dong Wang, Xin-Long Chen, Han-Lin Tao, Chen-Chen Zheng, Ya-Lan Tan, Xiaocong Zhao, Xing-Hua Zhang and Zhi-Jie Tan. “De novo design of functional RNAs through higher-order interactions.” (2026). To be published.

Preprint is coming soon…

## TriRNASP energy model

Please also cite the higher-order interaction potential:

> Tongwei Yuan, En Lou, Zouchenyu Zhou, Ya-Lan Tan, and Zhi-Jie Tan. “TriRNASP: A knowledge-based potential with three-body effects for accurate RNA structure evaluation.” *Biophysical Journal* 125(11), 2526–2540 (2026). https://doi.org/10.1016/j.bpj.2026.04.003

## Tool and parameter citations

Use the citations that match the components used in a calculation:

- DSSR-derived DBN files: Lu, Bussemaker, and Olson, *Nucleic Acids Research* 43, e142 (2015), https://doi.org/10.1093/nar/gkv716.
- Turner 2004 RNA parameters: Mathews et al., *PNAS* 101, 7287–7292 (2004), https://doi.org/10.1073/pnas.0401799101.
- ViennaRNA parameter-file distribution: Lorenz et al., *Algorithms for Molecular Biology* 6, 26 (2011), https://doi.org/10.1186/1748-7188-6-26.
- Primer3 parameter tables: Untergasser et al., *Nucleic Acids Research* 40, e115 (2012), https://doi.org/10.1093/nar/gks596.
- SantaLucia DNA nearest-neighbor model: SantaLucia, *PNAS* 95, 1460–1465 (1998), https://doi.org/10.1073/pnas.95.4.1460.

More detailed thermodynamic references are recorded next to the redistributed parameter sets in [Src/Turner2004_Par/README.md](Src/Turner2004_Par/README.md) and [Src/primer3_config/README.md](Src/primer3_config/README.md).

## Suggested acknowledgement language

> RNA sequences were designed and/or ranked with DS3dRNA, using higher-order interaction energies derived from TriRNASP. Local RNA or DNA thermodynamic screening used the parameter sources cited in the software documentation.
