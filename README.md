# lotteryml

A minimal, research-oriented PyTorch project exploring whether historical Lotto Max draw data
contains any detectable non-random structure using modern sequence models.

> **Important note**  
> This project is not intended to “beat” the lottery. If the process is truly random,
> out-of-sample performance should match chance.  
> The value of this repository is as a **clean, reproducible ML training harness**
> for studying randomness, bias, leakage, and evaluation methodology.

---

## Quick summary

- Transformer-based sequence model over historical Lotto Max draws  
- Treats main numbers as an **unordered set** (multi-label prediction)  
- Predicts the bonus number separately with explicit constraints  
- Supports CPU, single-GPU, and multi-GPU (DDP) training  
- Designed for research, education, and methodology validation  

---

## Project overview

This repository implements a Transformer-based model that predicts the **next Lotto Max draw**
given a fixed window of past draws.

The formulation closely matches real Lotto Max mechanics:

- **7 main numbers** are treated as an unordered set  
  (multi-label classification over 50 numbers)
- **1 bonus number** is predicted separately  
  (single-label classification)
- The bonus prediction is explicitly constrained so it cannot overlap the main numbers

The project is intentionally:

- minimal in dependencies
- explicit in assumptions
- suitable for CPU, single-GPU, or multi-GPU (DDP) training

---

## Data format

Input data must be a CSV file with the following header:

    date,n1,n2,n3,n4,n5,n6,n7,bonus

Example row:

    2025-12-23,3,28,37,38,39,41,43,45

Rules:

- Numbers must be in the range **1–50**
- The 7 main numbers must be unique
- The bonus number must not overlap the main numbers
- Rows should be ordered chronologically (oldest → newest)

A cleaned example dataset (`lottomax.csv`) is included in the repository.

---

## Model formulation

### Input representation

Each draw is encoded as a 100-dimensional vector:

- 50-dimensional multi-hot vector for the 7 main numbers
- 50-dimensional one-hot vector for the bonus number

A sequence of past draws (`context_len`) is passed to a Transformer encoder.

### Outputs

The model produces:

- **Main number logits:** 50-dimensional multi-label output
- **Bonus logits:** 50-dimensional single-label output, masked to exclude main numbers

### Losses

- Binary cross-entropy for the main number set
- Cross-entropy for the bonus number
- Final loss = main loss + bonus loss

---

## Chance-level baselines (sanity checking)

These values are printed automatically during training:

- **Expected main hits** when choosing 7 uniformly from 50:  
  ≈ `7 × (7 / 50) ≈ 0.98`  
  → hit rate ≈ **0.14**

- **Bonus accuracy** with main-number masking:  
  ≈ `1 / (50 − 7) ≈ 0.023`

Any meaningful improvement over these baselines should be treated with caution
and validated out-of-sample.

---

## Installation

### Requirements

- Python 3.9+
- PyTorch
- NumPy

### Install

    pip install torch numpy

Or, if using a requirements file:

    pip install -r requirements.txt

For GPU support, install the appropriate PyTorch build from:  
https://pytorch.org/get-started/locally/

---

## Usage

### Single-process (CPU or 1 GPU)

    python train_lottomax_set.py \
      --data_path lottomax.csv \
      --epochs 16 \
      --context_len 32 \
      --batch_size 16

### Multi-GPU (Distributed Data Parallel)

# Single-process (CPU or 1 GPU) — same hyperparameters as above
# Multi-GPU (2 GPUs, DDP)
torchrun --standalone --nproc_per_node=2 train_lottomax_set.py --data_path lottomax.csv --epochs 1024 --context_len 32 --batch_size 16 --d_model 128 --nhead 4 --num_layers 8 --dropout 0.1 --eval_every 1000000 --no_amp

The training script supports:

- automatic mixed precision (AMP)
- gradient clipping
- checkpointing
- periodic evaluation
- CPU or GPU execution

---

## Project goals

This repository is intended to serve as:

- a reference implementation for set-based prediction problems
- a teaching example for proper evaluation of random processes
- a baseline harness for testing alternative models or representations
- a clean public artifact demonstrating ML engineering practices

This project is **not** intended as financial or gambling advice.

---

## Limitations

- The dataset is relatively small by modern ML standards
- Any apparent predictive signal may be due to noise or data artifacts
- Results should always be evaluated against chance baselines
- No claims are made about real-world predictive power

---

## License

This project is released under the MIT License.
