# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runnable entrypoints and core Python modules. Use `src/train_stage1.py` for stage-1 RAE decoder training, `src/train.py` for stage-2 transport training, `src/stage1_sample*.py` and `src/sample*.py` for inference, and `src/calculate_stat.py` for latent statistics. Shared helpers live in `src/utils/`, evaluation code in `src/eval/`, stage-specific models in `src/stage1/` and `src/stage2/`, and adversarial components in `src/disc/`. YAML configs are organized under `configs/stage1/` and `configs/stage2/`; decoder JSON configs live in `configs/decoder/`. `assets/` holds reference images. Keep generated outputs in ignored paths such as `ckpts/`, `samples/`, `stats/`, `models/`, and `wandb/`.

## Build, Test, and Development Commands
`conda env create -f environment.yml && conda activate rae` creates the base Python 3.10 environment.

`pip install -r requirements.txt` installs the remaining Python dependencies used by training and evaluation scripts.

`torchrun --standalone --nproc_per_node=4 src/train_stage1.py --config configs/stage1/training/DINOv2-B_decXL.yaml --data-path <imagenet_train> --results-dir ckpts/stage1` starts stage-1 decoder training.

`torchrun --standalone --nproc_per_node=4 src/train.py --config <stage2_config.yaml> --data-path <imagenet_train> --results-dir ckpts/stage2` starts stage-2 training.

`python src/stage1_sample.py --config <config.yaml> --image assets/pixabay_cat.png` is the fastest smoke test for a trained stage-1 checkpoint.

`python pack_images.py <sample_dir> [image_size] [save_dir]` converts generated image folders into `.npz` files for FID-style workflows.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python changes PEP 8 aligned. Match existing naming: `snake_case` for modules, functions, and variables; `CamelCase` for classes. Prefer descriptive CLI flags and explicit config names such as `DINOv2-B_decXL.yaml` or `DiTDH-XL_DINOv2-B.yaml`. Add reusable logic under `src/utils/` or the relevant stage package instead of duplicating code in top-level scripts.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with targeted smoke tests against the affected entrypoint, using a small `ImageFolder` subset and a known config. For model, sampling, or metric changes, rerun the relevant sampling script or `src/calculate_stat.py` and record the exact config, checkpoint, and output directory used.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects, often with a prefix such as `fix:` (`fix: stage1 training`). Keep each commit focused on one logical change and keep the subject line concise. Pull requests should explain the intent, list the configs or scripts touched, note required environment variables (`EXPERIMENT_NAME`, `WANDB_KEY`, `ENTITY`, `PROJECT`), and include representative metrics or sample outputs when behavior or model quality changes.
