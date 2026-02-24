# Running CryoZeta via Apptainer Container

## Stage 1 — Atom Detection

```bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --bind $(pwd):$(pwd) CryoZeta.sif bash -c "cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && cryozeta-detection json-run $(pwd)/examples/example.json $(pwd)/output/example --device cuda"
```

## Stage 2 — Structure Prediction

```bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --bind $(pwd):$(pwd) CryoZeta.sif bash -c "cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && export LAYERNORM_TYPE=fast_layernorm && cryozeta-inference --seeds 101 --load_checkpoint_path /app/CryoZeta/assets/cryozeta-v0.0.1.safetensors --em_file_dir $(pwd)/output/example --dump_dir $(pwd)/output/example --input_json_path $(pwd)/examples/example.json --use_deepspeed_evo_attention true --model.N_cycle 10 --sample_diffusion.N_sample 5 --sample_diffusion.N_step 20 --data.num_dl_workers 1 --use_interpolation false --overwrite false"
```

With interpolation:

```bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --bind $(pwd):$(pwd) CryoZeta.sif bash -c "cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && export LAYERNORM_TYPE=fast_layernorm && cryozeta-inference --seeds 101 --load_checkpoint_path /app/CryoZeta/assets/cryozeta-interpolate-v0.0.1.safetensors --em_file_dir $(pwd)/output/example --dump_dir $(pwd)/output/example --input_json_path $(pwd)/examples/example.json --use_deepspeed_evo_attention true --model.N_cycle 10 --sample_diffusion.N_sample 5 --sample_diffusion.N_step 20 --data.num_dl_workers 1 --use_interpolation true --overwrite false"
```

## Stage 3 — Combine Results

```bash
apptainer exec --bind $(pwd):$(pwd) CryoZeta.sif bash -c "cd /app/CryoZeta && source /opt/conda/etc/profile.d/conda.sh && conda activate cryozeta && cryozeta-combine --dump-dir $(pwd)/output/example --input-json-path $(pwd)/examples/example.json --seeds 101 --num-select 5"
```
