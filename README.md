# CryoZeta Inference

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-blue?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/badge/uv-0.5-blue?logo=astral)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/Ruff-0.15.1-yellow?logo=ruff)](https://github.com/astral-sh/ruff)
[![prek](https://img.shields.io/badge/prek-0.3.2-green?logo=git)](https://github.com/kawarabiyu/prek)

<img src="resources/cryozeta-banner.png" alt="CryoZeta Banner" width="100%"/>

<img src="resources/cryozeta-workflow.png" alt="CryoZeta Workflow" width="100%"/>

CryoZeta is a de novo macromolecular structure modeling tool that integrates cryo-EM density information with a diffusion-model-based structure prediction pipeline.

Kihara Lab website: https://kiharalab.org/
Kihara Lab EM server: https://em.kiharalab.org/algorithm/CryoZeta

## Setup

Estimated time: < 15 minutes

### Hardware Requirements

- CUDA-capable GPU with 32 GB memory or more and CUDA 12.6

### 1. Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or visit https://docs.astral.sh/uv/getting-started/installation/ for other installation methods.

### 2. Clone and set up the project

```bash
git clone https://github.com/kiharalab/CryoZeta.git
cd CryoZeta
```

### 3. Install dependencies and the package

```bash
# Make the script executable
chmod +x initialize_script.sh

# Install all dependencies
sh initialize_script.sh
```

## Usage

### Quick Example

```bash
# Run CryoZeta inference
sh inference_demo.sh
```

### Prepare Input JSON

The input is a JSON file containing a list of entries. Each entry describes one
cryo-EM target. See `examples/example.json` for a complete example.

```json
[
    {
        "name": "9b0l",
        "modelSeeds": [],
        "map_path": "examples/emd_44046.map.gz",
        "resolution": 2.99,
        "contour_level": 0.3,
        "sequences": [ ... ]
    }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | A unique identifier for the entry (used as the output directory name). |
| `modelSeeds` | list | Reserved for future use (can be left empty `[]`). |
| `map_path` | string | Path to the cryo-EM density map (`.map` or `.map.gz`). |
| `resolution` | float | Resolution of the cryo-EM map in angstroms. |
| `contour_level` | float | Recommended contour level for the map. |
| `sequences` | list | List of biomolecular chains in the complex (see below). |

#### Sequence Types

Each element in `sequences` is an object with exactly one of the following keys:

- **`proteinChain`** -- a protein chain.
- **`dnaSequence`** -- a DNA strand.
- **`rnaSequence`** -- an RNA strand.

Every sequence object contains:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | string | One-letter residue/nucleotide codes. |
| `count` | int | Number of identical copies of this chain in the complex. |
| `msa` | object | *(optional)* Precomputed MSA information (protein and RNA only). |

The optional `msa` object has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `precomputed_msa_dir` | string | Path to a directory containing precomputed MSA `.a3m` files. For protein chains the directory should contain `uniref100_hits.a3m` (and optionally `mmseqs_other_hits.a3m`). For RNA chains it should contain `rnacentral.a3m`. |
| `pairing_db` | string | Pairing database name (e.g. `"uniref100"`). |

> **Note:** DNA sequences do not require MSA information.

### CryoZeta Inference

Estimated time: ~ 30 minutes

Specify the input JSON path, then run:

```bash
sh inference_demo.sh
```

#### Output Files

Results are saved under the `dump_dir` (default: `output/example`), organized by
entry name:

```
output/example/
└── <name>/                              # one directory per entry in the input JSON
    ├── CryoZeta-Detection/              # Step 1: atom detection
    │   ├── <name>.pt                    #   detected atom coordinates, features, and cluster ids
    │   ├── <name>_CA.pdb                #   protein C-alpha coordinates (PDB format)
    │   ├── <name>_C1P.pdb               #   nucleic acid C1' coordinates (PDB format)
    │   └── <name>_timing.txt            #   per-stage timing breakdown
    ├── CryoZeta/                        # Step 2a: standard CryoZeta model
    │   ├── seed_<seed>/
    │   │   ├── predictions/             #   predicted structures and confidence scores
    │   │   │   ├── <name>_sample_N.cif  #     predicted structure (mmCIF format)
    │   │   │   └── <name>_summary_confidence_sample_N.json
    │   │   ├── predictions_superimposed/#   structures superimposed onto the EM map
    │   │   ├── predictions_teaser/      #   structures fitted via TEASER++
    │   │   ├── predictions_svd_0.8/     #   structures fitted via SVD (weight 0.8)
    │   │   └── predictions_svd_0.4/     #   structures fitted via SVD (weight 0.4)
    │   └── saved_data/
    │       ├── scores.csv               #   fitting scores for all samples and methods
    │       └── output_dict_<name>.pt    #   intermediate tensors for downstream use
    ├── CryoZeta-Interpolate/            # Step 2b: CryoZeta with interpolation (same layout as above)
    └── CryoZeta-Final/                  # Step 3: top N results selected from both models
        └── <name>_sample_{0..N}.cif     #   best structures ranked by recall_ccmask_ca
```

The primary output is the `CryoZeta-Final/` directory, which contains the
top-ranked predicted structures selected from both the standard and interpolation
models. Each structure is in **mmCIF** format and can be opened with molecular
viewers such as ChimeraX or PyMOL.

#### Reference

<details>
<summary><strong>## Development</strong></summary>

### Install dev dependencies

```bash
pixi install -e dev
```

### Linting and formatting with Ruff

```bash
# Check for lint errors
pixi run -e dev ruff check .

# Auto-fix lint errors where possible
pixi run -e dev ruff check --fix .

# Format code
pixi run -e dev ruff format .

# Check formatting without applying changes
pixi run -e dev ruff format --check .
```

### Pre-commit hooks with prek

[prek](https://github.com/kawarabiyu/prek) is a lightweight pre-commit hook manager.

```bash
# Install hooks into your local .git/hooks
pixi run -e dev prek install

# Run all hooks manually against staged files
pixi run -e dev prek run
```

</details>

## Acknowledgements

CryoZeta is built upon **[Protenix](https://github.com/bytedance/Protenix)** and **[OpenFold](https://github.com/aqlaboratory/openfold)**. If you use CryoZeta, please also cite their work.

CryoZeta builds upon and is inspired by several excellent open-source projects:

- **[Protenix](https://github.com/bytedance/Protenix)** (ByteDance) -- An open-source biomolecular structure prediction framework. CryoZeta's structure prediction pipeline is built upon Protenix.
- **[OpenFold](https://github.com/aqlaboratory/openfold)** (Ahdritz et al.) -- An open-source protein structure prediction framework. CryoZeta reuses and adapts several OpenFold modules (e.g. layer normalization, tensor utilities). Our local copy is based on commit [`bb3f51`](https://github.com/aqlaboratory/openfold/commit/bb3f51e5a2cf2d5e3b709fe8f7d7a083c870222e).
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** (NVIDIA) -- CUDA Templates for Linear Algebra Subroutines and Solvers, used for DeepSpeed DS4Sci EvoformerAttention kernels.
- **[TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)** (MIT SPARK Lab) -- A fast and certifiably robust point cloud registration library, used for superimposing predicted structures onto cryo-EM maps.
- **[VESPER](https://github.com/kiharalab/VESPER_CUDA)** (Kihara Lab) -- A GPU-accelerated cryo-EM map comparison and fitting tool.

### Citations

<details>
<summary>Protenix</summary>

Please refer to the [Protenix repository](https://github.com/bytedance/Protenix) for citation information.

</details>

<details>
<summary>OpenFold</summary>

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	journal = {bioRxiv}
}
```

</details>
