#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
# One kernel-bench cell on one GPU, no ballast. Account, partition and QOS go on the sbatch line:
#   sbatch -A <acct> -p <partition> -q <qos> --job-name=<V>-<N>-<TAG> --output=$GKB/logs/%x-%j.log \
#     --export=ALL,GKB=$GKB,V=<variant>,N=<n_token|real>,TAG=r1[,ENTRY=<name>,DET=<tree>,TRIALS=<dir>,PIXI=<pixi>] cell.slurm
# $GKB holds repo/CryoZeta (this checkout and its pixi env), assets/, entries/, det/, synth/, trials/ and logs/.
# N=real: the real entry ENTRY (default 9b0l), seed 101, frozen detection $GKB/det/$DET (default v1).
#   9b0l also writes the trunk dump (input for the sealed P0-5 gates, compared on CPU elsewhere); other entries are capacity cells.
# Otherwise: synthetic 9b0l at N tokens, P 620, from $GKB/synth/9b0l-N-620 (synth_inputs).
# Hard cap H (default 13800 s) keeps a run inside a 4 h job limit; a cell that hits it is TIMEOUT, never SUCCESS.
# Output: $TRIALS/{P0-5/9b0l/V/s101-TAG | L3-real/ENTRY/V/s101-TAG | L3/9b0l-P620/V/N<N>-TAG}. An existing dir is never reused.
set -u
: "${GKB:?}" "${V:?}" "${N:?}" "${TAG:?}"
cd "$GKB/repo/CryoZeta" || exit 2
T=${TRIALS:-$GKB/trials/a100-80gb}; VD=docs/kernel-bench/variants; PIXI=${PIXI:-pixi}; E=${ENTRY:-9b0l}
test -e $VD/$V.yaml || { echo "NOVARIANT $V"; exit 2; }
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4 PYTHONUNBUFFERED=1 \
  CRYOZETA_ASSETS_DIR=$GKB/assets TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
# run_trial takes a physical GPU index and sets CUDA_VISIBLE_DEVICES itself. Under Slurm device cgroups nvidia-smi lists only
# the allocated GPU, as index 0.
NG=$(nvidia-smi -L | wc -l); if [ "$NG" = 1 ]; then GI=0; else GI=${CUDA_VISIBLE_DEVICES%%,*}; fi; unset CUDA_VISIBLE_DEVICES
echo "START $(date -Is) job=${SLURM_JOB_ID:-} gpus_seen=$NG gpu=$GI commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
nvidia-smi -L
if [ "$N" = real ] && [ "$E" = 9b0l ]; then
  O=$T/P0-5/9b0l/$V/s101-$TAG
  A=(--entry $GKB/entries/9b0l.json --frozen-det $GKB/det/${DET:-v1} --timeout 7200 --instrument trunk_dump --seed 101)
elif [ "$N" = real ]; then
  O=$T/L3-real/$E/$V/s101-$TAG
  test -e $GKB/det/${DET:?}.manifest.json || { echo "NOINPUT $GKB/det/$DET"; exit 5; }
  A=(--entry $GKB/entries/$E.json --frozen-det $GKB/det/$DET --timeout ${H:-13800} --seed 101)
else
  S=$GKB/synth/9b0l-$N-620; O=$T/L3/9b0l-P620/$V/N$N-$TAG
  test -e $S/det.manifest.json || { echo "NOINPUT $S"; exit 5; }
  A=(--synthetic --entry $S/entries/9b0l_syn${N}_real.json --frozen-det $S/det --timeout ${H:-13800} --smaller 805:1413.3 --n-token $N)
fi
test ! -e $O || { echo "EXISTS $O"; exit 4; }
mkdir -p "$(dirname $O)"
$PIXI run -e default --frozen python -m cryozeta.kernel_bench.run_trial --protocol P-CZ --mode A --assets $GKB/assets \
  --variant $VD/$V.yaml --gpu $GI --cache cold --out $O "${A[@]}"; rc=$?
test -e $O/trial.json || { echo "NOTRIAL rc=$rc $O"; exit 8; }
# Verdict as in gpu-chain.sh (no ballast, no foreign apps): CONTENDED when another compute app is left on the GPU or M3 - M2 > 1024 MiB.
python3 -I - $O/trial.json <<'EOF'
import json, sys
t = json.load(open(sys.argv[1])); m2 = t.get("M2_invocation"); m3 = t.get("M3_nvml_peak_mib")
m2 = max([x for x in (m2 if isinstance(m2, list) else [m2]) if isinstance(x, (int, float))], default=None)
c = bool(t["gpu_after"].get("compute_apps")) or (m2 is not None and m3 is not None and m3 - m2 / 2**20 > 1024)
print("VERDICT outcome=%s M3_MiB=%s T5_s=%s %s" % (t["outcome"], m3, t.get("T5_s"), "CONTENDED" if c else "clean"))
EOF
echo "END $(date -Is) rc=$rc $O"
