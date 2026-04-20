# Modal Setup Notes

This note records the one-time Modal setup for running the FlashInfer benchmark on Modal.

## 1. Log in to Modal

Enter the Python environment first.

This repository uses Python 3.12 in the README. If `conda` is available, use the `fi-bench` environment:

```bash
conda activate fi-bench
```

If the environment does not exist yet, create it first:

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
```

If `conda activate fi-bench` prints `zsh: command not found: conda`, install Miniforge first:

```bash
brew install miniforge
```

Initialize conda for zsh:

```bash
conda init zsh
```

Then restart the terminal, or reload the shell config:

```bash
source ~/.zshrc
```

Create and activate the environment:

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
```

If `conda activate fi-bench` prints `CondaError: Run 'conda init' before 'conda activate'`, run:

```bash
conda init zsh
source ~/.zshrc
conda activate fi-bench
```

If you do not want to install conda, use a local virtual environment instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the Modal CLI inside the active Python environment if it is not installed yet:

```bash
python -m pip install modal
```

Log in from the terminal:

```bash
modal setup
```

The command opens a browser authentication flow. After login succeeds, Modal stores the local credentials for future `modal run` commands.

To confirm the CLI can access your account:

```bash
modal profile current
```

## 2. Download the contest dataset

The contest dataset is hosted on Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

If `git lfs install` prints `git: 'lfs' is not a git command`, install Git LFS first:

```bash
brew install git-lfs
git lfs install
```

Then clone the dataset:

```bash
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

If `mlsys26-contest/` already exists locally, you do not need to download it again just because it is listed in `.gitignore`.

`.gitignore` only means Git will not track or commit the dataset directory. It does not delete local files.

## 3. Upload the dataset to Modal

Create the Modal volume once:

```bash
modal volume create flashinfer-trace
```

Upload the dataset directory:

```bash
modal volume put flashinfer-trace /path/to/mlsys26-contest
```

For this repository's Modal scripts, the expected path inside Modal is:

```text
/data/mlsys26-contest
```

The scripts also try `/data` as a fallback if the dataset contents were uploaded directly to the volume root.

## 4. Run benchmarks on Modal

For decode:

```bash
modal run gdn_decode/scripts/run_modal.py
```

For prefill:

```bash
modal run gdn_prefill/scripts/run_modal.py
```

## Dataset Notes

You only need to download the Hugging Face dataset again if one of these is true:

- The dataset directory is missing from your machine.
- The local dataset clone is incomplete or corrupted.
- You want to update it to the latest remote contents.
- You are working on a different machine that does not have the dataset.

You only need to upload to Modal again if one of these is true:

- The `flashinfer-trace` Modal volume does not exist.
- The dataset was never uploaded to that Modal workspace.
- You deleted or replaced the Modal volume.
- You updated the local dataset and want Modal to use the updated copy.

Because `mlsys26-contest/` is ignored by Git, it is safe to keep the dataset next to this repository while avoiding accidental commits of large data files.
