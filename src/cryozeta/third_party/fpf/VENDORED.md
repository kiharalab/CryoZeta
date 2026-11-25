# Vendored FlashPairformer kernels (kernel-bench candidates)

Source: `https://github.com/anthropics/uplifting-biomolecular-modeling` @ `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`,
directory `common/opt_core/opt_core/`. That repository is a reference release and is not maintained, so CryoZeta owns this copy.
Only these two providers are carried. The stock-Protenix patch kit is not applied.

| Here | Upstream (`common/opt_core/`) | Status |
|---|---|---|
| `LICENSE`, `NOTICE` | repository root | identical (Apache-2.0, Copyright 2026 Anthropic, PBC) |
| `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` | `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` | identical |
| `fpf_triatt_k2b.json`, `fpf_trimul_v4.json` | `opt_core/kernels/META/` | identical |
| `opt_core/oom.py`, `opt_core/kernels/safe_settings.py` | same paths | identical (stdlib dependencies of both kernels) |
| `opt_core/kernels/fpf_flashpairformer.PROTENIX_LICENSE` | same path | identical (portions derive from Protenix, Apache-2.0) |
| `opt_core/kernels/fpf_trimul_v4/{__init__,generic,cells,kernels,kdesc,table}.py`, `table.json`, `NOTICE` | same paths | identical except `cells.py` (below) |
| `opt_core/kernels/fpf_triatt_k2b/triatt_k2b.py`, `K2B_CELLS.json`, `NOTICE` | same paths | identical except `triatt_k2b.py` (below) |
| `opt_core/__init__.py`, `opt_core/kernels/__init__.py`, `opt_core/kernels/fpf_triatt_k2b/__init__.py` | — | new one-line docstring packages. The upstream K2B `__init__` (a `sys.path` loader plus registry wrappers) is not carried. |

Import edits (absolute `opt_core.*` to relative imports; kernel statements unchanged):

```
fpf_triatt_k2b/triatt_k2b.py:439  from opt_core.kernels import safe_settings as _safe  ->  from .. import safe_settings as _safe
fpf_triatt_k2b/triatt_k2b.py:650  from opt_core.oom import is_oom                       ->  from ...oom import is_oom
fpf_triatt_k2b/triatt_k2b.py:678  from opt_core.oom import is_oom                       ->  from ...oom import is_oom
fpf_trimul_v4/cells.py:12         from opt_core.kernels import safe_settings as SAFE   ->  from .. import safe_settings as SAFE
```

Behaviour of the standalone copy (upstream code paths, not edits):
- `generic.trimul_packed` tries `opt_core.kernels.trimul._pystack`. It gets ImportError and dispatches directly, without the padded first call.
- `generic._stamp_api` tries `opt_core.kernels.trimul.native`. It gets ImportError and returns None, so no probe-verdict stamps are written and the warm probe runs in every process.
- Version strings: `fpf_trimul_v4.__version__` = 4.4.6 (the META json says 4.4.4); K2B v1.1.1.

CryoZeta loads both providers lazily from `cryozeta/kernels/flashpairformer.py`.
K2B and TriMul v4 are available through the inference options documented in
[`docs/inference-kernels.md`](../../../../docs/inference-kernels.md). The Protenix
license copy has normalized trailing whitespace.
