# REMAP

REMAP prunes MoE experts by accounting for fallback rerouting after removal and encouraging diversity in the retained expert set, outperforming prior methods across multiple MoE models and benchmarks.

This repository contains the official PyTorch implementation of REMAP.

## Installation

**Step 1**: Create a new Conda environment:

```bash
conda create -n remap python=3.10 -y
conda activate remap
```

**Step 2**: Install PyTorch matching your CUDA version (e.g., CUDA 12.1):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

The repository does not include calibration datasets. Please prepare them under `./data`:

- **C4**: Please download the first part of the C4 training data `c4-train.00000-of-01024.json` from [allenai/c4](https://huggingface.co/datasets/allenai/c4).
- **MATH**: You can use our pre-built calibration set in `./data/math_pretrain_style.json`. To reproduce our construction, please download the training set of MATH and use the script `./data/math_calib_construction.py`.
- **Alpaca**: For speedup benchmark, please download `alpaca_data_cleaned.json` from [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).

Finally, please organize the datasets as follows:
```text
./data
|-- __init__.py
|-- alpaca_data_cleaned.json
|-- build.py
|-- c4-train.00000-of-01024.json
|-- dataset.py
|-- math_calib_construction.py
`-- math_pretrain_style.json
```

## Usage

Run REMAP with a local model path or a Hugging Face model id:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --method remap_pruning \
    --model_path /path/to/Mixtral-8x7B-v0.1 \
    --calib_set c4 \
    --sp_ratio 0.5 \
    --remap_diversity_lambda 0.25 \
    --eval_tasks mmlu,boolq,hellaswag,winogrande,rte,arc_challenge,arc_easy,openbookqa
```

Run NAEE:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --method naee_pruning \
    --model_path /path/to/Mixtral-8x7B-v0.1 \
    --calib_set c4 \
    --sp_ratio 0.5 \
    --eval_tasks mmlu,boolq,hellaswag,winogrande,rte,arc_challenge,arc_easy,openbookqa
```

## Key Arguments

| Argument | Description |
| :--- | :--- |
| `--method` | One of `remap_pruning`, `naee_pruning`, or `progressive_pruning`. |
| `--model_path` | Local model path or Hugging Face model id. |
| `--calib_set` | Calibration set: `c4` or `math`. |
| `--sp_ratio` | Sparsity ratio (fraction of routed experts to remove, e.g., 0.5 removes 50%). |
| `--remap_diversity_lambda` | Diversity regularization strength (λ). Set to 0 to disable. Higher values penalize redundant retained experts more strongly. |
| `--eval_tasks` | Comma-separated LM Evaluation Harness tasks. |

## Outputs

Each run writes to:

```text
output/<model>-<method>/<date>/<time>_<run-details>/
```

The output directory includes console logs, evaluation logs, and `pruning_config.json` when expert identities can be recovered after pruning.

## Repository Layout

```text
main.py                    Entry point for pruning and evaluation
method/remap/              REMAP implementations for Mixtral and Qwen-MoE
method/naee/               NAEE pruning implementations
model/                     Local Mixtral model definitions
data/                      Dataset loading utilities
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgment

This repository is built upon the official codebase of [NAEE](https://github.com/NJU-MC/NAEE).
