# 🌟 ArcanaGPT LLM Project 🌟

Welcome to the **ArcanaGPT LLM** project—an educational codebase for training a GPT-style language model on large corpora and evaluating on datasets such as HellaSwag. This repository is designed for researchers, engineers, and enthusiasts who want to experiment with a minimalistic yet powerful GPT training setup.

## 📚 Table of Contents
1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Repository Structure](#-repository-structure)
4. [Installation](#-installation)
5. [Data Preparation](#-data-preparation)
6. [Training](#-training)
7. [Evaluation](#-evaluation)
8. [Analysing Results](#-analysing-results)
9. [Contributing](#-contributing)
10. [Licence](#-licence)

## 🧠 Project Overview
- **Objective**: Provide a compact codebase to train GPT-like models on large text corpora.
- **Design Philosophy**:
    - **Modularity**: Logical packages (`data/`, `models/`, `training/`, `evaluation/`) for clarity and maintainability.
    - **Simplicity**: Minimal dependencies, straightforward usage.
    - **Scalability**: Built with Distributed Data Parallel (DDP) to leverage multi-GPU or multi-node clusters.
- **Target Audience**: AI researchers, machine learning engineers, and hobbyists interested in GPT-like language models.

## ✨ Features
- **GPT Architecture**:
    - Multi-head self-attention using `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`.
    - Layer Normalisation, MLP feed-forward blocks, and weight tying for efficient memory usage.
- **Distributed Training**:
    - Simple DDP setup with automatic gradient synchronisation.
    - Support for gradient accumulation to fit large effective batch sizes.
- **Evaluation**:
    - HellaSwag dataset integration for multiple choice reasoning tasks.
    - Regular validation loop for perplexity / cross-entropy monitoring.
- **Analytics**:
    - Automatic logging of training loss, validation loss, HellaSwag accuracy, etc.
    - Script for plotting logs to compare model performance against baselines (GPT-2 or GPT-3).

## 🗂️ Repository Structure

```
my_llm_project/
├── data/
│   ├── __init__.py
│   ├── data_loader.py         # Loads token shards as PyTorch Tensors
│   └── prepare_fineweb.py     # Downloads & tokenises large text (FineWeb-edu) into shards
├── evaluation/
│   ├── __init__.py
│   └── hellaswag_eval.py      # HellaSwag downloading, rendering, & evaluation logic
├── models/
│   ├── __init__.py
│   └── gpt_model.py           # GPT architecture & config (attention, MLP, blocks, etc.)
├── training/
│   ├── __init__.py
│   └── ddp_training.py        # Main DDP loop (initialisation, forward, backward, logging)
├── main.py                    # Single entry point for training
├── analyse.ipynb              # Jupyter notebook for parsing & visualising logs
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

### 🔑 Key Files
- **main.py**: Entry point that invokes `ddp_training.main_training_loop()`.
- **prepare_fineweb.py**: Automates downloading and tokenising a large dataset (FineWeb-edu).
- **ddp_training.py**: Sets up Distributed Data Parallel, gradient accumulation, model checkpointing, etc.
- **hellaswag_eval.py**: Handles retrieval of HellaSwag data and computing the model’s accuracy on the multiple-choice task.

## 🛠️ Installation
1. **Clone the Repository**
    ```sh
    git clone https://github.com/maxhartml/ArcanaGPT.git
    cd ArcanaGPT
    ```

2. **Install Dependencies**
    - Create a Python virtual environment (optional but recommended):
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Install required libraries:
        ```sh
        pip install -r requirements.txt
        ```
    - Ensure you have an appropriate version of PyTorch that supports your GPU hardware.

3. **(Optional) Configure GPUs**
    - If you have NVIDIA GPUs, install CUDA toolkits, drivers, etc.
    - For Apple Silicon (M1/M2), ensure `torch.backends.mps.is_available()`.

## 📊 Data Preparation
1. **Download & Tokenise Data**
    Before training, you need a tokenised dataset. For example, if you’re using the FineWeb-edu corpus:
    ```sh
    # from project root
    python -m data.prepare_fineweb
    ```
    This script:
    - Downloads the FineWeb-edu dataset via HuggingFace Datasets.
    - Uses `tiktoken` (GPT-2 encoder) to tokenise the text.
    - Saves shards as `.npy` files in `edu_fineweb10B/`.

2. **Shard Explanation**
    Each `.npy` file is a “shard” of tokens that can be loaded in parallel to feed the model during training.

## 🚀 Training
1. **Single GPU / CPU Training**
    ```sh
    python main.py
    ```
    This will:
    - Launch the training loop (`ddp_training.main_training_loop()`).
    - Detect available GPUs or MPS. If no GPU is found, it will use CPU.

2. **Distributed GPU Training**
    For an 8-GPU machine, run:
    ```sh
    torchrun --standalone --nproc_per_node=8 main.py
    ```
    - `torchrun` spawns multiple processes, each pinned to a separate GPU.
    - The script sets up DDP using environment variables like `RANK`, `WORLD_SIZE`, etc.

3. **Hyperparameters**
    - **Batch Size**: Controlled by `total_batch_size`, `B`, `T`, and `grad_accum_steps` in `ddp_training.py`.
    - **Learning Rate**: Cosine decay with warmup (see `get_learning_rate`).
    - **Checkpoints**: Created every 5000 steps (or final step) in the log directory.

4. **Logs & Checkpoints**
    - Training and validation loss is logged to a `log.txt` inside `log/`.
    - HellaSwag accuracy is also appended.
    - Model states are saved as `.pt` files in the same directory.

## 📈 Evaluation
1. **Built-in Validation Loop**
    Every 100 steps, it computes validation loss on a small subset of the data (`val_loader`).

2. **HellaSwag Accuracy**
    Every 250 steps, or at the last step, the code runs a HellaSwag evaluation.
    - The script downloads the HellaSwag JSON if it’s not already present.
    - Each process evaluates its slice of examples, then an all-reduce step aggregates results.
    - Logged to `log.txt` under `hella`.

3. **Custom Evaluations**
    - If you have additional tasks or datasets, add a new function in `evaluation/`, or adapt `ddp_training.py` for your new logic.

## 📊 Analysing Results
We provide an example Jupyter notebook `analyse.ipynb` to parse and visualise the training logs:

- **Log File Parsing**: The script extracts training steps, stream type (e.g. train, val, hella), and numerical value.
- **Comparison to Baselines**: We draw dashed lines at known GPT-2 / GPT-3 metrics.
- **Plotting**: The first subplot shows training/validation loss on a log scale; the second shows HellaSwag accuracy.

## 🤝 Contributing
We welcome pull requests, issues, or feature requests. If you add new datasets or modules, consider placing them in the appropriate folder (`data/`, `evaluation/`, etc.) and ensure it remains consistent with the project’s structure.

## 📜 Licence
This code is offered under the MIT licence. Feel free to modify and reuse for your own experiments, with appropriate credit.

We hope this codebase helps you explore GPT-like architectures with ease! If you have questions or suggestions, please open an issue or send a pull request.
