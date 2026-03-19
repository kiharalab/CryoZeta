# CryoZeta

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11%20%7C%2012%20%7C%2013-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![pixi](https://img.shields.io/badge/pixi-0.41.1-blue?logo=pixi)](https://pixi.sh)
[![Ruff](https://img.shields.io/badge/Ruff-0.15.1-yellow?logo=ruff)](https://github.com/astral-sh/ruff)
[![prek](https://img.shields.io/badge/prek-0.3.2-green?logo=git)](https://github.com/kawarabiyu/prek)

<img src="resources/cryozeta-banner.png" alt="CryoZeta Banner" width="100%"/>

<img src="resources/cryozeta-workflow.png" alt="CryoZeta Workflow" width="100%"/>

CryoZeta is a de novo macromolecular structure modeling tool that integrates cryo-EM density information with a diffusion-model-based structure prediction pipeline.

Kihara Lab website: https://kiharalab.org/
Kihara Lab EM server: https://em.kiharalab.org/algorithm/CryoZeta

## Latest Updates

- **2026-03-09: Memory Optimization** 
  - Optimize CUDA memory allocation
  - Supported up to ~2,800 residues/nucleotides with ~2,000 support points.
    
## Setup

Estimated time: < 15 minutes

### Hardware Requirements

- CUDA-capable GPU with 32 GB memory or more
- NVIDIA driver that supports CUDA 11.0 or higher (check with `nvidia-smi`)

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Or visit https://pixi.sh/latest/#installation for other installation methods.

### 2. Clone and set up the project

```bash
git clone https://github.com/kiharalab/CryoZeta.git
cd CryoZeta
pixi run setup
```

The setup command automatically:

1. Installs all dependencies (Python, CUDA, C++ libraries, etc.).
2. Detects your GPU and selects the matching CUDA version (11, 12, or 13).
3. Downloads CryoZeta model weights from Hugging Face.
4. Clones and builds [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus).

### CUDA Environment Selection

CryoZeta ships three CUDA environments. During setup and inference the correct
one is **auto-detected** from two properties of your system:

| Property | How it is read | What it controls |
|----------|---------------|-----------------|
| **Driver-supported CUDA version** | `nvidia-smi` "CUDA Version" field | Hard upper bound -- the driver cannot run a newer CUDA toolkit than this. |
| **GPU compute capability** | `nvidia-smi --query-gpu=compute_cap` | Architectural eligibility -- very old GPUs are not supported by newer CUDA releases. |

The auto-detection logic combines both:

| Compute Capability | Driver CUDA | Selected Environment | CUDA Toolkit | PyTorch |
|---|---|---|---|---|
| >= 10.0 (Blackwell) | >= 13 | `cu13` | 13.x | >= 2.7 |
| >= 8.0 (Ampere / Ada / Hopper) | >= 12 | `default` (cu12) | 12.8 | >= 2.7 |
| < 8.0 (Volta / Turing / older) | >= 11 | `cu11` | 11.8 | >= 2.0, < 2.5 |

> **Tip:** Run `nvidia-smi` to see both the driver version and the maximum CUDA
> version your driver supports. The "CUDA Version" shown by `nvidia-smi` is the
> **ceiling** -- you can install any CUDA toolkit up to that version.

#### Manual Override

If you need to use a specific CUDA version (e.g. your driver supports CUDA 12
but you want to test with CUDA 11), you can override the auto-detection by
entering a pixi shell for the desired environment:

```bash
# Enter the CUDA 11 environment
pixi shell -e cu11

# Enter the CUDA 13 environment
pixi shell -e cu13

# Enter the default (CUDA 12) environment
pixi shell
```

Once inside the shell, all commands (including `sh inference_demo.sh`) will use
the environment you selected. You can also target an environment directly
without entering a shell:

```bash
# Run a single command in the CUDA 11 environment
pixi run -e cu11 cryozeta-detection json-run examples/example.json output/example --device cuda

# Install a specific environment
pixi install -e cu11
```

Available environments:

| Environment | CUDA | Description |
|-------------|------|-------------|
| `default` | 12 | Default environment (CUDA 12.8) |
| `cu11` | 11 | CUDA 11.8 with PyTorch < 2.5 |
| `cu13` | 13 | CUDA 13.x for Blackwell GPUs |
| `dev` | 12 | Development (CUDA 12 + linting tools) |
| `dev-cu11` | 11 | Development (CUDA 11 + linting tools) |
| `dev-cu13` | 13 | Development (CUDA 13 + linting tools) |

## Usage

### Quick Start

```bash
sh inference_demo.sh
```

This runs the full CryoZeta pipeline on the bundled example (`examples/example.json`) and writes results to `output/example/`. The correct CUDA environment is auto-detected from your GPU and driver.

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

### Running Inference

Estimated time: ~ 30 minutes

Edit the parameters at the top of `inference_demo.sh` (input path, GPU id, etc.),
then run:

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

## License

- The **source code** in this repository is released under the [GNU General Public License v3.0](LICENSE).
- The **trained model weights** are distributed under a separate license and are **free for academic and non-commercial research use only**.

Commercial use of the model weights is not permitted without permission.  
For commercial licensing inquiries, please contact the authors.

See [WEIGHT_LICENSE.md](WEIGHT_LICENSE.md) for full terms.


## Development

<details>
<summary><b>Development Instructions</b></summary>

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

Please cite our paper:

```bibtex
@article{zhang2026accurate,
  title        = {Accurate macromolecular complex modeling for cryo-EM},
  author       = {Zhang, Zicong and Li, Shu and Farheen, Farhanaz and Kagaya, Yuki and Liu, Boyuan and Ibtehaz, Nabil and Terashi, Genki and Nakamura, Tsukasa and Zhu, Han and Khan, Kafi and Zhang, Yuanyuan and Kihara, Daisuke},
  journal      = {bioRxiv},
  year         = {2026},
  doi          = {10.64898/2026.02.13.705846},
  url          = {https://www.biorxiv.org/content/10.64898/2026.02.13.705846v1},
  note         = {Preprint}
}
```


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

<details>
<summary>SmartFold</summary>

```bibtex
@article{smartfold2023,
  title   = {SMARTFold: Structure Modeling from Atomic-Resolution cryo-EM Maps with Deep Learning},
  author  = {Li, Peng and Guo, Liang and Liu, Haotian and Li, Bo and Ma, Feng and Ni, Xiang and Gao, Chao},
  journal = {bioRxiv},
  year    = {2023},
  doi     = {10.1101/2023.11.02.565403},
  url     = {https://www.biorxiv.org/content/10.1101/2023.11.02.565403v1},
  eprint  = {https://www.biorxiv.org/content/10.1101/2023.11.02.565403v1.full.pdf},
  publisher = {Cold Spring Harbor Laboratory}
}
```

</details>
