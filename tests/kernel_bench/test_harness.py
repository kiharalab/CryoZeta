"""Torch-free checks for the kernel-bench harness. The run_trial dry-runs also run its preflight,
which parses every model step through CryoZeta's real config parsers in a CPU-only child (it imports the model, so it
needs the pixi activation for TEASER++, ~1 min per dry-run):

  CUDA_VISIBLE_DEVICES= pixi run -e default --frozen python -I -m unittest discover -s tests/kernel_bench -p test_harness.py -v
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

from cryozeta.kernel_bench import common as C
from cryozeta.kernel_bench.compare_outputs import compare_scores, kendall_tau
from cryozeta.kernel_bench.synth_inputs import scale_sequences

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VARIANTS = os.path.join(REPO, "docs", "kernel-bench", "variants")
needs_docs = unittest.skipUnless(os.path.isdir(VARIANTS), "needs the local docs/kernel-bench variant YAMLs, which are not tracked")
HARNESSES = ("op_bench", "capture_activations", "run_trial", "synth_inputs", "compare_outputs", "provenance",
             "mode_b", "v0_regression")


def cli(mod, *args):
    return subprocess.run([sys.executable, "-I", "-m", f"cryozeta.kernel_bench.{mod}", *args],
                          capture_output=True, text=True, timeout=900)


def ctr(op="triatt", site="empairformer", req="cueq", served="cueq", scope=False, allowed=True, reason="", n=1):
    return {"op": op, "call_site": site, "requested_backend": req, "served_backend": served,
            "in_replacement_scope": scope, "route_allowed": allowed, "reason": reason, "count": n}


def cls(**kw):
    base = dict(exit_code=0, timed_out=False, oom_killed=False, err_texts=[], log_text="", counters=[],
                missing=[], nonfinite=[])
    return C.classify(**{**base, **kw})[0]


class Timeout(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(C.effective_timeout(9000, "B", same_cell_t4=10)[0], 9000)
        self.assertEqual(C.effective_timeout(9000, "A", same_cell_t4=10)[0], 1800)
        self.assertEqual(C.effective_timeout(9000, "A", same_cell_t4=1000)[0], 3000)
        self.assertEqual(C.effective_timeout(4000, "A", same_cell_t4=2000)[0], 4000)
        self.assertEqual(C.effective_timeout(1e6, "A", 2000, smaller=(1000, 100))[0], 2400)
        self.assertEqual(C.effective_timeout(9000, "A")[0], 9000)
        with self.assertRaises(ValueError):
            C.effective_timeout(0, "A")
        with self.assertRaises(ValueError):
            C.effective_timeout(9000, "A", smaller=(1000, 100))


class Classify(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(cls(), "SUCCESS")
        self.assertEqual(cls(err_texts=["Traceback\nRuntimeError: boom"]), "CRASH")  # ERR file + exit 0
        self.assertEqual(cls(log_text="[pre-check] 9b0l: no interpolation features"), "SKIPPED")
        self.assertEqual(cls(timed_out=True, exit_code=-9), "TIMEOUT")
        self.assertEqual(cls(exit_code=-9), "CRASH")  # SIGKILL without an oom-kill event
        self.assertEqual(cls(exit_code=-9, oom_killed=True), "HOST_OOM")
        self.assertEqual(cls(exit_code=1, log_text="KernelUnsupported: triatt@msa"), "UNSUPPORTED")
        self.assertEqual(cls(exit_code=1, log_text="KernelFallback: x"), "FALLBACK")
        self.assertEqual(cls(exit_code=1, log_text="torch.OutOfMemoryError: CUDA out of memory"), "GPU_OOM")
        self.assertEqual(cls(missing=["a.cif"]), "CRASH")
        self.assertEqual(cls(nonfinite=["a.cif"]), "NUMERICAL")
        self.assertEqual(cls(gates_failed=["G-coord"]), "NUMERICAL")

    def test_routing_classes(self):
        out_of_scope = ctr(req="cueq", served="torch", allowed=False, reason="ineligible:shape")
        self.assertEqual(cls(counters=[out_of_scope]), "FALLBACK")
        in_scope = ctr(req="fpf_k2b", served="cueq", scope=True, reason="ineligible:dtype")
        self.assertEqual(cls(counters=[in_scope]), "UNSUPPORTED")
        rescue = ctr(req="cueq", served="ds4sci", reason="ineligible:shape")  # allowed V0 rescue
        self.assertEqual(cls(counters=[rescue]), "SUCCESS")
        self.assertEqual(C.rescue_counts([rescue]), {"triatt@empairformer:cueq->ds4sci:ineligible:shape": 1})

    def test_scope_must_execute(self):
        scope = [{"op": "triatt", "call_site": "msa", "provider": "fpf_k2b"}]
        self.assertEqual(cls(scope=scope, counters=[ctr()]), "UNSUPPORTED")  # requested kernel never ran
        hit = ctr(site="msa", req="fpf_k2b", served="fpf_k2b", scope=True)
        self.assertEqual(cls(scope=scope, counters=[hit]), "SUCCESS")
        for bad in (dict(call_site="confidence"), dict(op="opm"), dict(count=0),
                    dict(requested_backend="torch", served_backend="torch")):
            self.assertEqual(cls(scope=scope, counters=[{**hit, **bad}]), "UNSUPPORTED")
        for site in ("empairformer", "confidence"):
            scope.append({**scope[0], "call_site": site})
        self.assertEqual(cls(scope=scope, counters=[hit]), "UNSUPPORTED")
        self.assertEqual(C.scope_coverage(scope, [hit]), 0.0)
        self.assertEqual(cls(scope=scope, counters=[{**hit, "call_site": r["call_site"]} for r in scope]), "SUCCESS")

    def test_short_trimul_scope(self):
        scope = [{"op": "trimul", "call_site": "msa", "provider": "fpf_trimul_v4", "min_tokens": 101}]
        short = dict(ctr(op="trimul", site="msa", reason="below_min_tokens"), first_shape=[1, 63, 63, 128])
        long = ctr(op="trimul", site="msa", req="fpf_trimul_v4", served="fpf_trimul_v4", scope=True)
        long["first_shape"] = [1, 101, 101, 128]
        inputs = [{"n_token": 63}, {"n_token": 101}]
        self.assertEqual(C.check_scope(scope, [short], inputs[:1]), ("N/A", ""))
        self.assertEqual(C.check_scope(scope, [short, long], inputs), (1.0, ""))
        self.assertTrue(C.check_scope(scope, [short], inputs)[1])
        self.assertTrue(C.check_scope(scope, [long], inputs)[1])
        self.assertTrue(C.check_scope(scope, [long, {**long, "in_replacement_scope": False}], inputs[1:])[1])
        for bad in (dict(reason="served"), dict(first_shape=[1, 101, 101, 128]), dict(count=0)):
            self.assertTrue(C.check_scope(scope, [{**short, **bad}], inputs[:1])[1])

    def test_unparseable_output(self):
        with tempfile.TemporaryDirectory() as d:
            ok, cut = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            with open(ok, "w") as f:
                json.dump({"plddt": 0.5}, f)
            with open(cut, "w") as f:
                f.write('{"plddt": 0.')
            self.assertEqual(C.scan_nonfinite([ok, cut]), [f"{cut} (unparseable)"])

    def test_scope_coverage(self):
        self.assertEqual(C.scope_coverage([], [ctr()]), "N/A")
        cs = [ctr(req="fpf_k2b", served="fpf_k2b", scope=True, n=3), ctr(req="fpf_k2b", served="cueq", scope=True)]
        self.assertEqual(C.scope_coverage([{"op": "triatt", "call_site": "empairformer", "provider": "fpf_k2b"}], cs), 0.75)


class Inputs(unittest.TestCase):
    seqs = [{"proteinChain": {"sequence": "A" * 100, "count": 2, "msa": {"precomputed_msa_dir": "m"}}},
            {"rnaSequence": {"sequence": "G" * 30, "count": 1}}]

    def test_scale_exact(self):
        for n in (1, 99, 230, 231, 1000):
            out, _ = scale_sequences(self.seqs, n)
            self.assertEqual(C.n_polymer_tokens({"sequences": out}), n)

    def test_truncation_drops_msa(self):
        out, notes = scale_sequences(self.seqs, 150)
        self.assertEqual(out[0]["proteinChain"]["count"], 1)
        self.assertNotIn("msa", out[1]["proteinChain"])
        self.assertEqual(len(out[1]["proteinChain"]["sequence"]), 50)
        self.assertTrue(notes)

    def test_query_only_msa(self):
        """A protein/RNA chain without an MSA gets a query-only one (the featurizer needs one; without it
        inference skips the sample and still exits 0). Chains that kept theirs are untouched."""
        from cryozeta.kernel_bench import synth_inputs

        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/b.json", "w") as f:
                json.dump([{"name": "x", "sequences": self.seqs}], f)
            out = f"{d}/s"
            self.assertEqual(synth_inputs.main(["--base-json", f"{d}/b.json", "--entry", "x", "--root", d,
                                                "--n-token", "150", "--out", out]), 0)
            with open(f"{out}/entries/x_syn150_real.json") as f:
                seqs = [next(iter(s.items())) for s in json.load(f)[0]["sequences"]]
            self.assertEqual(seqs[0][1]["msa"]["precomputed_msa_dir"], f"{d}/m")  # real MSA kept
            for i, (kind, v) in enumerate(seqs[1:], 1):
                m = v["msa"]["precomputed_msa_dir"]
                self.assertEqual(m, os.path.join(out, "msa", str(i)))
                self.assertEqual(sorted(os.listdir(m)), sorted(synth_inputs.QUERY_ONLY_MSA[kind]))
                for fn in os.listdir(m):
                    with open(os.path.join(m, fn)) as f:
                        self.assertEqual(f.read(), f">query\n{v['sequence']}\n")

    def test_entry_paths(self):
        e = {"name": "x", "map_path": "assets/m.map", "sequences": self.seqs}
        a = C.absolute_entry(e, "/root")
        self.assertEqual(a["map_path"], "/root/assets/m.map")
        self.assertEqual(a["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"], "/root/m")
        self.assertEqual(e["map_path"], "assets/m.map")  # input untouched
        self.assertEqual(C.n_polymer_tokens(e), 230)
        with self.assertRaises(ValueError):
            C.n_polymer_tokens({"sequences": [{"ligand": {"ligand": "ATP"}}]})


class Compare(unittest.TestCase):
    def test_kendall(self):
        self.assertEqual(kendall_tau([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertEqual(kendall_tau([1, 2, 3], [6, 5, 4]), -1.0)
        self.assertIsNone(kendall_tau([1, 1], [2, 2]))

    def test_scores(self):
        row = lambda r: {"recall": r, "recall_ccmask_ca": r, "num_dist_over_4": "0"}
        ref = {("svd", 0): row("0.9"), ("svd", 1): row("0.5")}
        cand = {("svd", 0): row("0.4"), ("svd", 1): row("0.6")}
        out = compare_scores(ref, cand)
        self.assertFalse(out["top1_match"])
        self.assertEqual(out["kendall_tau"], -1.0)
        self.assertEqual(compare_scores({}, cand)["status"], "scores.csv missing")


class Cell(unittest.TestCase):
    """Requested (N_token, P) vs the recorded model input; the V0 cap is never clamped."""

    def rec(self, n="x", tok=100, pt=50, valid=None, pp=None):
        return {"name": n, "n_token": tok, "n_point": pt, "valid_points": pt if valid is None else valid,
                "shapes": {"p": [pt, pt, 114] if pp is None else pp}}

    def test_expected(self):
        man = {"entries": {"x": {"N_token": 100, "N_point": 50}, "y": {"N_token": None, "N_point": None}}}
        self.assertEqual(C.expected_cells(man, ["x"]), ({"x": (100, 50)}, ""))
        for m, names in ((man, ["y"]), (man, ["z"]), (man, ["x", "z"]), (None, ["x"])):
            self.assertEqual(C.expected_cells(m, names)[0], {})

    def test_check(self):
        exp = {"x": (100, 50)}
        self.assertEqual(C.check_cell(exp, [self.rec(), self.rec()]), ("match", ""))  # e.g. P-COMB, Mode B
        self.assertEqual(C.check_cell(exp, [])[0], "unverified")
        for bad in (self.rec(pt=60), self.rec(tok=99), self.rec(valid=49), self.rec(pp=[50, 49, 114]),
                    self.rec(n="y")):
            self.assertEqual(C.check_cell(exp, [self.rec(), bad])[0], "mismatch", bad)

    def test_runs(self):
        exp = {"x": (100, 50)}
        self.assertEqual(C.check_runs(exp, [[self.rec()], [self.rec()]]), ("match", ""))
        self.assertEqual(C.check_runs(exp, [[self.rec()], []])[0], "unverified")  # one run cannot vouch for another
        self.assertEqual(C.check_runs(exp, [[], [self.rec(pt=60)]])[0], "mismatch")
        self.assertEqual(C.check_runs(exp, [])[0], "unverified")

    def test_outcome(self):
        self.assertEqual(C.cell_outcome("SUCCESS", "", "match", ""), ("SUCCESS", ""))
        self.assertEqual(C.cell_outcome("SUCCESS", "", "unverified", "d")[0], "INVALID_CELL")
        self.assertEqual(C.cell_outcome("GPU_OOM", "x", "mismatch", "d")[0], "INVALID_CELL")
        self.assertEqual(C.cell_outcome("GPU_OOM", "x", "unverified", "d"), ("GPU_OOM", "x"))
        self.assertIn("INVALID_CELL", C.OUTCOMES)

    def test_matrix(self):
        from cryozeta.kernel_bench.synth_inputs import capacity_cells

        cells, skip = capacity_cells([512, 2048], {"min": 300, "med": 900, "max": 900, "2xmax": 1800})
        self.assertEqual(cells, [{"n_token": 512, "n_point": 300, "levels": ["min"]},
                                 {"n_token": 2048, "n_point": 300, "levels": ["min"]},
                                 {"n_token": 2048, "n_point": 900, "levels": ["med", "max"]},  # deduplicated
                                 {"n_token": 2048, "n_point": 1800, "levels": ["2xmax"]}])
        self.assertEqual([(x["n_token"], x["level"]) for x in skip], [(512, "med"), (512, "max"), (512, "2xmax")])
        self.assertEqual(capacity_cells([4096], {"2xmax": 2400})[1][0]["cap"], 2000)  # never clamped to 2000

    def test_synth_cap(self):
        from cryozeta.kernel_bench.synth_inputs import max_points

        self.assertEqual((max_points(805), max_points(2000), max_points(4752)), (805, 2000, 2000))
        with tempfile.TemporaryDirectory() as d:
            base = ["--base-json", f"{d}/no.json", "--entry", "x", "--root", d, "--out", f"{d}/s"]
            for n_tok, n_pt in ((100, 101), (5000, 2001)):
                r = cli("synth_inputs", *base, "--n-token", str(n_tok), "--n-point", str(n_pt))
                self.assertEqual(r.returncode, 2)
                self.assertIn("not realizable", r.stderr)
            r = cli("synth_inputs", *base, "--n-token", "5000", "--n-point", "2000")  # realizable; fails on no.json
            self.assertEqual(r.returncode, 2)
            self.assertNotIn("not realizable", r.stderr)


TRIAL = {"variant_id": "V0", "synthetic": False, "provenance": {"variant_sha256": None}, "outcome": "SUCCESS",
         "protocol": "P-CZ", "entries": ["x"], "frozen_det_sha256_before": None, "settings": {"seed": 101}}


def write_trial(d, **kw):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/trial.json", "w") as f:
        json.dump({**TRIAL, **kw}, f)


@needs_docs
class MSdc(unittest.TestCase):
    """M-SDC-5 regroups RNG draws, so sample-paired gates are refused, not substituted."""

    def test_metadata(self):
        from cryozeta.kernel_bench.run_trial import read_variant

        v = read_variant(os.path.join(VARIANTS, "M-SDC-5.yaml"))
        self.assertEqual((v["status"], v["sample_paired"]), ("deferred", False))
        self.assertIn("noise-alignment", v["deferred_reason"])
        for f in sorted(os.listdir(VARIANTS)):
            if f != "M-SDC-5.yaml":
                self.assertTrue(read_variant(os.path.join(VARIANTS, f))["sample_paired"], f)

    def test_gates_refused(self):
        from cryozeta.kernel_bench.compare_outputs import parse_args

        tol = os.path.join(REPO, "docs", "kernel-bench", "tolerances.template.yaml")
        with tempfile.TemporaryDirectory() as d:
            args = ["--ref", d, "--cand", d, "--name", "x", "--out", f"{d}/c.json", "--tolerances", tol]
            r = cli("compare_outputs", *args, "--variant", os.path.join(VARIANTS, "M-SDC-5.yaml"))
            self.assertEqual(r.returncode, 2)
            self.assertIn("M-SDC-5 is not sample-paired", r.stderr)
            r = cli("compare_outputs", *args)  # no variant: cannot tell, so refused too
            self.assertEqual(r.returncode, 2)
            self.assertIn("needs --variant", r.stderr)
            v0 = args + ["--variant", os.path.join(VARIANTS, "V0.yaml")]
            r = cli("compare_outputs", *v0)
            self.assertEqual(r.returncode, 2)  # gates apply only to run_trial outputs
            self.assertIn("no trial.json", r.stderr)
            write_trial(d)
            self.assertTrue(parse_args(v0).tolerances)
            r = cli("compare_outputs", *args, "--variant", os.path.join(VARIANTS, "M-LMA.yaml"))
            self.assertEqual(r.returncode, 2)  # --variant must be the candidate's variant
            self.assertIn("not the variant the candidate trial ran", r.stderr)
            for kw, msg in (({"synthetic": True}, "synthetic"), ({"outcome": "NUMERICAL"}, "two SUCCESS"),
                            ({"settings": {"seed": 7}}, "--seed 101")):
                write_trial(d, **kw)
                r = cli("compare_outputs", *v0)
                self.assertEqual(r.returncode, 2, kw)
                self.assertIn(msg, r.stderr)

    def test_unpaired_refused(self):
        tol = os.path.join(REPO, "docs", "kernel-bench", "tolerances.template.yaml")
        with tempfile.TemporaryDirectory() as d:
            args = ["--ref", f"{d}/ref", "--cand", f"{d}/cand", "--name", "x", "--out", f"{d}/c.json",
                    "--tolerances", tol, "--variant", os.path.join(VARIANTS, "V0.yaml")]
            write_trial(f"{d}/cand")
            for kw, msg in (({"settings": {"seed": 102}}, "not paired"), ({"variant_id": "M-LMA"}, "baseline V0")):
                write_trial(f"{d}/ref", **kw)
                r = cli("compare_outputs", *args)
                self.assertEqual(r.returncode, 2, kw)
                self.assertIn(msg, r.stderr)

    def test_sealed_needs_calibration(self):
        from cryozeta.kernel_bench.compare_outputs import gates

        with tempfile.TemporaryDirectory() as d:
            write_trial(d)
            tol = os.path.join(d, "tol.yaml")
            with open(tol, "w") as f:  # all gates filled but no calibration trials
                f.write("sealed: true\nca_rmsd_max: 1\nconfidence_abs_max: 1\ntop1_must_match: true\n"
                        "kendall_tau_min: 0\ntrunk_rel_max: 1\n")
            r = cli("compare_outputs", "--ref", d, "--cand", d, "--name", "x", "--out", f"{d}/c.json",
                    "--tolerances", tol, "--variant", os.path.join(VARIANTS, "V0.yaml"))
            self.assertEqual(r.returncode, 2)
            self.assertIn("calibration.trials", r.stderr)
            r = cli("compare_outputs", "--ref", d, "--cand", d, "--name", "x", "--out", tol)
            self.assertEqual(r.returncode, 2)  # never overwrite an existing result
            self.assertIn("exists", r.stderr)
        res = {"coordinates": {"ca_rmsd_kabsch": [0.1]}, "confidence": {"abs_max": {"plddt": 0.0}},
               "scores": {"top1_match": True, "kendall_tau": 1.0}, "trunk": None}
        tol = {"ca_rmsd_max": 1, "confidence_abs_max": 1, "top1_must_match": True, "kendall_tau_min": 0,
               "trunk_rel_max": 1}
        g = gates(res, tol)
        self.assertIsNone(g["G-trunk"])  # threshold set, no trunk evidence: not a pass
        self.assertFalse(g["all_pass"])
        self.assertTrue(gates({**res, "trunk": {"z": 0.5}}, tol)["all_pass"])
        nan = float("nan")  # max() skips a NaN unless it comes first: a NaN must fail, never pass
        self.assertFalse(gates({**res, "coordinates": {"ca_rmsd_kabsch": [0.1, nan]}}, tol)["G-coord"])
        self.assertFalse(gates({**res, "trunk": {"s": 0.5, "z": nan}}, tol)["G-trunk"])

    def test_confidence_keeps_nan(self):
        from cryozeta.kernel_bench.compare_outputs import compare_confidence

        with tempfile.TemporaryDirectory() as d:
            for stage, vals in (("r", [0.5, 0.5]), ("c", [float("nan"), 0.4])):
                os.makedirs(f"{d}/{stage}/seed_1/predictions")
                for i, v in enumerate(vals):
                    with open(f"{d}/{stage}/seed_1/predictions/x_summary_confidence_sample_{i}.json", "w") as f:
                        json.dump({"plddt": v}, f)
            self.assertTrue(math.isnan(compare_confidence(f"{d}/r", f"{d}/c", "x", 1)["abs_max"]["plddt"]))

    def test_trial_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = cli("run_trial", "--protocol", "P-CZ", "--mode", "A", "--entry", os.path.join(VARIANTS, "V0.yaml"),
                    "--frozen-det", d, "--assets", d, "--gpu", "0", "--timeout", "60", "--cache", "cold",
                    "--variant", os.path.join(VARIANTS, "M-SDC-5.yaml"), "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 2)
            self.assertIn("M-SDC-5 is deferred", r.stderr)
            self.assertFalse(os.path.exists(f"{d}/t"))


class V0Regression(unittest.TestCase):
    """v0_regression's comparator and routing-arm checks on synthetic dumps/counters."""

    def test_compare_dirs(self):
        import numpy as np

        from cryozeta.kernel_bench.v0_regression import compare_dirs

        with tempfile.TemporaryDirectory() as d:
            for side, x in (("a", 1.0), ("b", 1.5)):
                os.makedirs(f"{d}/{side}/s")
                with open(f"{d}/{side}/s/c.json", "w") as f:
                    json.dump({"plddt": [1.0, x], "name": "n"}, f)
                with open(f"{d}/{side}/s/m.cif", "w") as f:
                    f.write("data_x\nATOM 1\n" if side == "a" else "data_x\nATOM 2\n")
                np.savez(f"{d}/{side}/s/p.npz", c=np.arange(3.0))  # value-equal, bytes may differ
                names = np.array(["CA", "N" if side == "a" else "O"], dtype=object)  # pickled member
                np.savez(f"{d}/{side}/s/o.npz", c=np.arange(3.0) * x, atom=names)
                with open(f"{d}/{side}/same.txt", "w") as f:
                    f.write("same")
            with open(f"{d}/a/only.txt", "w") as f:
                f.write("")
            r = compare_dirs(f"{d}/a", f"{d}/b")
            self.assertFalse(r["bitwise"])
            self.assertEqual((r["only_ref"], r["only_cand"], r["n_files"]), (["only.txt"], [], 6))
            self.assertEqual((r["differing"]["s/o.npz"]["max_abs_diff"], r["differing"]["s/o.npz"]["incomparable"]),
                             (1.0, ["/atom"]))  # object member: raw bytes differ, never unpickled
            import zipfile

            with zipfile.ZipFile(f"{d}/a/s/p.npz") as z, zipfile.ZipFile(f"{d}/a/s/p2.npz", "w") as w:
                for i in z.infolist():
                    w.writestr(zipfile.ZipInfo(i.filename, (2001, 1, 1, 0, 0, 0)), z.read(i))
            from cryozeta.kernel_bench.v0_regression import compare_file

            self.assertEqual(compare_file(f"{d}/a/s/p.npz", f"{d}/a/s/p2.npz"),
                             {"identical": True, "npz_members_identical": True})  # only zip header times differ
            os.remove(f"{d}/a/s/p2.npz")
            self.assertEqual(r["differing"]["s/c.json"]["max_abs_diff"], 0.5)
            self.assertEqual(r["differing"]["s/m.cif"]["first_diff_line"], 2)
            self.assertTrue(r["differing"].get("s/p.npz", {"value_equal": True})["value_equal"])
            self.assertTrue(compare_dirs(f"{d}/a", f"{d}/a")["bitwise"])
            from cryozeta.kernel_bench.v0_regression import _err_tail

            self.assertEqual(_err_tail(f"{d}/a"), "no ERR file")
            os.makedirs(f"{d}/a/s/CryoZeta/ERR")
            with open(f"{d}/a/s/CryoZeta/ERR/s.txt", "w") as f:
                f.write("Traceback\nKernelFallback: apb@atom\n")
            self.assertEqual(_err_tail(f"{d}/a"), "KernelFallback: apb@atom")

    def test_bf16_pt(self):  # GPU dumps carry bf16 tensors, which numpy cannot hold
        import torch

        from cryozeta.kernel_bench.v0_regression import compare_file

        with tempfile.TemporaryDirectory() as d:
            x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
            torch.save({"l": x}, f"{d}/a.pt")
            torch.save({"l": x + torch.tensor([0.0, 0.5], dtype=torch.bfloat16)}, f"{d}/b.pt")
            r = compare_file(f"{d}/a.pt", f"{d}/b.pt")
            self.assertEqual((r["value_equal"], r["max_abs_diff"]), (False, 0.5))

    def test_routing(self):
        from cryozeta.kernel_bench.v0_regression import check_routing, route_variants

        v0 = {"counters": [ctr("opm", "empairformer", "tilelang", "chunked"), ctr("opm", "msa", "chunked", "chunked"),
                           ctr("rowcol", "msa", "ds4sci", "torch"), ctr("trimul", "msa", "cueq", "cueq"),
                           ctr("trimul", "empairformer", "cueq", "cueq"), ctr("triatt"),
                           ctr("apb", "atom", "torch", "torch"), ctr("apb", "diffusion", "cueq", "torch")]}
        v = route_variants(v0)
        self.assertEqual(set(v), {"opm", "rowcol", "trimul", "apb"})  # triatt has no ALT
        self.assertEqual({(r["call_site"], r["provider"]) for r in v["apb"]["replace"]},
                         {("atom", "torch"), ("diffusion", "torch")})  # identity: cueq APB is never servable
        self.assertEqual({k: v[k]["expect_bitwise"] for k in v},
                         {"opm": False, "rowcol": False, "trimul": True, "apb": True})
        self.assertFalse(route_variants({"counters": [ctr("trimul", "msa", "torch", "torch")]})["trimul"]["expect_bitwise"])
        self.assertEqual({(r["call_site"], r["provider"]) for r in v["opm"]["replace"]},
                         {("empairformer", "einsum"), ("msa", "einsum")})
        self.assertEqual(v["rowcol"]["replace"], [{"op": "rowcol", "call_site": "msa", "provider": "lma"}])
        self.assertEqual(v["trimul"]["v0_served"], {"msa": ["cueq"], "empairformer": ["cueq"]})
        # one site, two V0 rows (cueq for some shapes, torch for others): C-CAT replaces both, so not bitwise
        mixed = route_variants({"counters": [ctr("trimul", "msa", "cueq", "cueq"), ctr("trimul", "msa", "cueq", "torch")]})
        self.assertEqual((mixed["trimul"]["replace"], mixed["trimul"]["v0_served"], mixed["trimul"]["expect_bitwise"]),
                         ([{"op": "trimul", "call_site": "msa", "provider": "cueq_cached_cat"}], {"msa": ["cueq", "torch"]},
                          False))
        good = {"counters": [ctr("rowcol", "msa", "lma", "lma", scope=True)]}
        self.assertEqual(check_routing(v["rowcol"], good), [])
        fell_back = {"counters": [ctr("rowcol", "msa", "lma", "torch", scope=True)]}
        self.assertEqual(len(check_routing(v["rowcol"], fell_back)), 1)
        self.assertEqual(check_routing(v["rowcol"], {"counters": []}), ["not reached: ('rowcol', 'msa')"])


class Variants(unittest.TestCase):
    """read_variant guards: a variant cannot override harness settings or claim pairing it cannot have."""

    def read(self, text):
        from cryozeta.kernel_bench.run_trial import read_variant

        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/v.yaml", "w") as f:
                f.write("variant_id: X\n" + text)
            return read_variant(f"{d}/v.yaml")

    def test_guards(self):
        for text in ("cli_args: [--seeds, '7']\n", "cli_args: [--em_file_dir, /x]\n",
                     "env: {CUDA_VISIBLE_DEVICES: '1'}\n", "env: {CUEQ_TRITON_CACHE_DIR: /x}\n",
                     "cli_args: '--use_lma true'\n",
                     "cli_args: [--infer_setting.sample_diffusion_chunk_size, '5']\n",
                     # CLI spellings that reach the same settings: '=' form, argparse prefix, a value re-split on spaces
                     "cli_args: ['--seeds=7', '--x=y']\n", "cli_args: [--load_checkpoint, /evil.safetensors]\n",
                     "cli_args: [--use_opm_tilelang, 'false --seeds 7']\n",
                     "cli_args: [--infer_setting.sample_diffusion_chunk, '5']\n",
                     "env: {CRYOZETA_SYNC_STAGE_TIMER: '0'}\n", "env: {CRYOZETA_TRUNK_DUMP: /x}\n",
                     # p_dim is overwritten after parsing (inference.build_configs): a silent no-op otherwise
                     "cli_args: [--model.input_embedder.p_dim, '3']\n", "cli_args: [--model.input_embedder.p, '3']\n",
                     "cli_args: [--num_workers, '8']\n"):  # DataLoader workers: a different P subsample
            with self.subTest(text), self.assertRaises(SystemExit):
                self.read(text)
        v = self.read("status: deferred\ndeferred_reason: r\n")
        self.assertIs(v["sample_paired"], False)  # deferred defaults to not paired
        self.read("sample_paired: false\ncli_args: [--infer_setting.sample_diffusion_chunk_size, '5']\n")

    @needs_docs
    def test_one_attention_algorithm(self):
        """Attention allows at most one of the memory-efficient kernel, DS4Sci/cueq and LMA (primitives.py:610; M-LMA
        used to crash); configs.use_flash never reaches the model."""
        from cryozeta.kernel_bench.run_trial import _flags, read_variant

        for f in sorted(os.listdir(VARIANTS)):
            fl = dict(zip(*[iter(_flags(read_variant(os.path.join(VARIANTS, f))["cli_args"]))] * 2))
            on = lambda k: fl.get(k, "false").lower() == "true"
            n = on("--use_lma") + on("--use_memory_efficient_kernel") + (
                on("--use_deepspeed_evo_attention") or on("--use_cuequivariance_attention"))
            self.assertLessEqual(n, 1, f)
            self.assertFalse(on("--use_flash") or on("--use_memory_efficient_kernel"), f)  # no-ops


class Cli(unittest.TestCase):
    """--help exits 0 and an invalid config exits 2 (argparse) for every harness, without torch work."""

    def test_help(self):
        for m in HARNESSES:
            with self.subTest(m):
                r = cli(m, "--help")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("usage", r.stdout)

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            bad = {
                "op_bench": ["--module", "m", "--backend", "torch", "--out", f"{d}/o.jsonl"],  # no --n/--inputs
                "capture_activations": ["--out", d, "--modules", "x", "--", "--a", "b"],  # --out exists
                "run_trial": ["--protocol", "P-COMB", "--mode", "B", "--entry", d, "--assets", d, "--gpu", "0",
                              "--timeout", "10", "--cache", "cold", "--out", f"{d}/t", "--frozen-det", d],
                "synth_inputs": ["--base-json", f"{d}/no.json", "--entry", "x", "--root", d, "--n-token", "5",
                                 "--out", f"{d}/s"],
                "compare_outputs": ["--ref", d, "--cand", d, "--name", "x", "--out", "o", "--ref-trunk", "t"],
                "provenance": ["snapshot", "--out", d],  # --out exists
                "mode_b": ["--warmup", "0", "--iter-root", d, "--record", "r", "--", "--a"],
                "v0_regression": ["--out", d, "--", "--seeds", "1"],  # --out exists
            }
            for m, args in bad.items():
                with self.subTest(m):
                    self.assertEqual(cli(m, *args).returncode, 2)

    def test_run_trial_dry_run_and_fresh_out(self):
        with tempfile.TemporaryDirectory() as d:
            entry = os.path.join(d, "e.json")
            with open(entry, "w") as f:
                json.dump([{"name": "x", "sequences": []}], f)
            det = os.path.join(d, "det")
            os.makedirs(det)
            args = ["--protocol", "P-CZ", "--mode", "A", "--entry", entry, "--assets", d, "--gpu", "0",
                    "--timeout", "9000", "--baseline-t4", "1000", "--cache", "cold", "--frozen-det", det]
            r = cli("run_trial", *args, "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 2)  # no freeze-det manifest: the requested cell is unverifiable
            self.assertIn("cannot verify", r.stderr)
            man = {"entries": {"x": {"N_token": 805, "N_point": 805}}, "tree_sha256": {"sha256": "0", "files": 0}}
            with open(det + ".manifest.json", "w") as f:
                json.dump(man, f)
            r = cli("run_trial", *args, "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 2)  # the tree is not the one that was frozen
            self.assertIn("tree_sha256", r.stderr)
            with open(det + ".manifest.json", "w") as f:
                json.dump({**man, "tree_sha256": C.sha256_tree(det)}, f)
            r = cli("run_trial", *args, "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 0, r.stderr)
            plan = json.loads(r.stdout)
            self.assertEqual(plan["expected_cells"], {"x": [805, 805]})
            self.assertFalse(plan["synthetic"])
            self.assertTrue(plan["env"]["CUEQ_TRITON_CACHE_DIR"].startswith(f"{d}/t/"))  # not ~/.cache
            self.assertEqual(plan["effective_timeout"], 3000)
            self.assertEqual(plan["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertTrue(plan["env"]["TRITON_CACHE_DIR"].startswith(f"{d}/t/"))  # task-scoped cold cache
            self.assertFalse(os.path.exists(f"{d}/t"))
            self.assertEqual(cli("run_trial", *args, "--out", d, "--dry-run").returncode, 2)
            with open(entry, "w") as f:  # em_file would bypass the frozen tree
                json.dump([{"name": "x", "sequences": [], "em_file": "/elsewhere/x.pt"}], f)
            r = cli("run_trial", *args, "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 2)
            self.assertIn("em_file", r.stderr)

    def test_synthetic_auto_label(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/entries")
            entry, det = f"{d}/entries/x_syn900_real.json", f"{d}/det"
            os.makedirs(det)
            with open(entry, "w") as f:
                json.dump([{"name": "x_syn900_real", "sequences": []}], f)
            with open(det + ".manifest.json", "w") as f:
                json.dump({"entries": {"x_syn900_real": {"N_token": 900, "N_point": 600}},
                           "tree_sha256": C.sha256_tree(det)}, f)
            r = cli("run_trial", "--protocol", "P-CZ", "--mode", "A", "--entry", entry, "--assets", d, "--gpu", "0",
                    "--timeout", "60", "--cache", "cold", "--frozen-det", det, "--out", f"{d}/t", "--dry-run")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(json.loads(r.stdout)["synthetic"])  # labelled without --synthetic

    @needs_docs
    def test_plc_stage_args(self):
        """P-LC: a variant's non-native flags ride after -- to every cycle_predict stage; an unsupported combination
        is refused by the preflight (real parsers, CPU-only child) before anything is created or launched."""
        with tempfile.TemporaryDirectory() as d:
            entry = os.path.join(d, "e.json")
            with open(entry, "w") as f:
                json.dump([{"name": "x", "sequences": [], "em_file": os.path.join(d, "x.pt")}], f)
            with open(f"{d}/conflict.yaml", "w") as f:
                f.write("variant_id: X\ncli_args: [--use_lma, 'true']\n")  # DS4Sci/cueq attention stay on
            args = ["--protocol", "P-LC", "--mode", "A", "--entry", entry, "--assets", d, "--gpu", "0",
                    "--timeout", "60", "--cache", "cold", "--dry-run"]
            for v, tail in (("M-LMA", ["--", "--use_lma", "true"]), ("M-CHUNK-128", ["--", "--infer_setting.chunk_size", "128"]),
                            ("V0", ["--num_dl_workers", "1"])):
                r = cli("run_trial", *args, "--variant", os.path.join(VARIANTS, f"{v}.yaml"), "--out", f"{d}/{v}")
                with self.subTest(v):
                    self.assertEqual(r.returncode, 0, r.stderr)
                    steps = dict(json.loads(r.stdout)["steps"])
                    self.assertEqual(steps["cycle_predict"][-len(tail):], tail)
            r = cli("run_trial", *args, "--variant", f"{d}/conflict.yaml", "--out", f"{d}/c")
            self.assertEqual(r.returncode, 2)
            self.assertIn("preflight cycle_predict", r.stderr)
            self.assertIn("preflight cycle_predict: -c: error: at most one attention algorithm", r.stderr)
            self.assertFalse(os.path.exists(f"{d}/c"))


if __name__ == "__main__":
    unittest.main()
