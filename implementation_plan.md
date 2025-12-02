# Implementation Plan - Local GPU Adaptation for Pro Trader RL

## Goal
Adapt `ptrl_TW50_optimized.py` (originally for Google Colab) to run on a local Windows/WSL2 Ubuntu environment with Nvidia RTX 4070 GPU support.

## User Review Required
- **Path Configuration**: The script will use relative paths (e.g., `./data`, `./models`) by default. Please confirm if `D:\wsl\Ubuntu_D` requires specific absolute paths or if running from the project root is sufficient.
- **Environment**: Assumes Python 3.8+ and CUDA drivers are correctly installed in WSL2.

## Proposed Changes

### 1. Dependency Management
- Create `requirements.txt` containing:
    - `stable-baselines3[extra]`
    - `yfinance`
    - `pandas`
    - `pandas_ta`
    - `gymnasium`
    - `matplotlib`
    - `shimmy`
    - `torch` (with CUDA support instructions)

### 2. Code Adaptation (`ptrl_TW50_local_gpu.py`)
- **[DELETE]** Google Colab specific code:
    - `from google.colab import drive`
    - `drive.mount(...)`
    - `!pip install ...`
- **[MODIFY]** Path Management:
    - Replace hardcoded `/content/drive/...` with `os.getcwd()` or relative paths.
    - Create directories (`models`, `results`, `data`) automatically if they don't exist.
- **[NEW]** GPU Configuration:
    - Add explicit check for `torch.cuda.is_available()`.
    - Configure Stable Baselines3 `PPO` to use `device='cuda'`.
- **[MODIFY]** Execution Flow:
    - Wrap the main logic in `if __name__ == "__main__":` to prevent execution on import.
    - Add command-line arguments (optional) for training steps or mode (train/backtest).

## Verification Plan
### Automated Tests
- Run the script with a small number of steps (e.g., `--test_run`) to verify:
    1.  GPU is detected.
    2.  Data downloads successfully.
    3.  Training loop starts without errors.
    4.  Models are saved to the local disk.

### Manual Verification
- User to run `python ptrl_TW50_local_gpu.py` in their WSL2 terminal.
- Check `nvidia-smi` to ensure GPU usage during training.
