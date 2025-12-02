# Pro Trader RL (TW50) - 安裝與執行指南

本指南將引導您如何在全新的 Windows 電腦上設定環境，並開始執行 Pro Trader RL 專案。

## 1. 前置需求 (Prerequisites)

在開始之前，請確保您的電腦已安裝以下軟體：

1.  **Git**: 用於下載專案代碼。
    *   [下載連結](https://git-scm.com/download/win)
2.  **Python 3.8 ~ 3.10**: 建議使用 3.10 版本。
    *   [下載連結](https://www.python.org/downloads/)
    *   **注意**: 安裝時請務必勾選 **"Add Python to PATH"**。
3.  **Visual C++ Build Tools** (通常安裝某些 Python 套件需要):
    *   [下載連結](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    *   安裝時勾選 "Desktop development with C++"。

---

## 2. 下載專案 (Clone Repository)

開啟命令提示字元 (CMD) 或 PowerShell，執行以下指令：

```powershell
# 1. 移動到您想存放專案的資料夾
cd D:\Projects  # (範例)

# 2. 下載專案
git clone https://github.com/ROGabdp/ptrl-TW50-v01.git

# 3. 進入專案資料夾
cd ptrl-TW50-v01
```

---

## 3. 建立虛擬環境 (Create Virtual Environment)

為了避免套件衝突，強烈建議使用虛擬環境。

```powershell
# 1. 建立名為 venv_win 的虛擬環境
python -m venv venv_win

# 2. 啟動虛擬環境
.\venv_win\Scripts\activate

# 成功啟動後，您的命令列前方會出現 (venv_win) 字樣
```

---

## 4. 安裝依賴套件 (Install Dependencies)

```powershell
# 1. 更新 pip (建議)
python -m pip install --upgrade pip

# 2. 安裝 PyTorch (若有 NVIDIA 顯卡)
# 請先到 https://pytorch.org/get-started/locally/ 確認適合您顯卡的指令。
# 以下為 CUDA 11.8 的範例 (適用於大多數較新顯卡):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 若只有 CPU，則執行:
# pip install torch torchvision torchaudio

# 3. 安裝專案其他依賴
pip install -r requirements.txt
```

---

## 5. 執行訓練 (Training)

一切準備就緒！現在可以開始訓練 AI 了。

```powershell
# 執行論文版本主程式
python ptrl_TW50_paper_version.py
```

程式啟動後會出現選單：
1.  輸入 `1`：重新開始訓練 (Train from Scratch)。
2.  輸入 `2`：載入現有模型繼續訓練 (Resume Training)。
3.  輸入 `3`：**刪除舊模型並重新開始** (建議初次使用或重大修改後使用)。

**注意**: 訓練過程中，程式會自動下載台股資料並儲存在 `data/` 資料夾。

---

## 6. 執行回測 (Backtesting)

當訓練完成後 (或您想測試現有模型)，程式通常會自動進入回測階段。
若您想單獨執行回測，可以使用：

```powershell
# 執行回測腳本 (需確保 models_paper/ 資料夾內有模型檔案)
python ptrl_TW50_backtest.py
```

---

## 7. 常見問題排除 (Troubleshooting)

*   **Q: 出現 `ModuleNotFoundError: No module named '...'`**
    *   A: 請確認您是否已啟動虛擬環境 `(venv_win)`，並已執行 `pip install -r requirements.txt`。

*   **Q: 出現 `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`**
    *   A: 這是常見的 PyTorch 衝突。請在程式最上方加入：
        ```python
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        ```

*   **Q: 訓練速度很慢**
    *   A: 請確認 PyTorch 是否成功抓到 GPU。
        ```python
        import torch
        print(torch.cuda.is_available()) # 應顯示 True
        ```

---
Happy Trading! 🚀
