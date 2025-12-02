# Pro Trader RL 本地端執行指南 (WSL2 + Nvidia GPU)

本指南將協助您在 Windows WSL2 (Ubuntu) 環境下，使用 Nvidia RTX 4070 顯卡執行 `ptrl_TW50_local_gpu.py`。

## 1. 前置檢查 (Prerequisites)

在開始之前，請確保您的 WSL2 環境已正確設定：

1.  **確認 GPU 驅動**：
    在 WSL2 終端機中輸入以下指令，確認能看到您的 RTX 4070：
    ```bash
    nvidia-smi
    ```
    *如果看到顯卡資訊列表，代表驅動正常。*

2.  **確認 Python 版本**：
    建議使用 Python 3.8 或以上版本：
    ```bash
    python3 --version
    ```

## 2. 環境建置 (Setup Environment)

建議使用虛擬環境 (Virtual Environment) 以避免套件衝突。

### 步驟 A: 建立虛擬環境 (推薦)
```bash
# 1. 安裝 venv (如果尚未安裝)
sudo apt-get update
sudo apt-get install python3-venv

# 2. 建立名為 'venv' 的虛擬環境
python3 -m venv venv

# 3. 啟動虛擬環境
source venv/bin/activate
```
*啟動後，您的終端機提示字元前應該會出現 `(venv)`。*

### 步驟 B: 安裝依賴套件
請確保您位於 `ptrl_TW50_local_gpu.py` 所在的目錄，然後執行：

```bash
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 中包含 `torch`。如果 pip 沒有自動安裝到支援 CUDA 的版本，您可能需要手動安裝 (通常 pip 會自動處理，但若出錯請參考 [PyTorch 官網](https://pytorch.org/get-started/locally/))：
> `pip install torch --index-url https://download.pytorch.org/whl/cu118` (視您的 CUDA 版本而定)

## 3. 執行程式 (Execution)

確認環境設定完畢後，即可執行主程式：

```bash
python ptrl_TW50_local_gpu.py
```

### 程式執行流程：
1.  **檢查 GPU**：程式開頭會顯示 `✅ CUDA is available! Device: NVIDIA GeForce RTX 4070`。
2.  **下載資料**：自動從 Yahoo Finance 下載台股 50 成分股資料 (存於 `./data`)。
3.  **特徵工程**：計算技術指標與正規化。
4.  **訓練模型**：
    *   開始訓練 **Buy Agent** (PPO)。
    *   每 50,000 步會自動存檔至 `./models`。
    *   接著訓練 **Sell Agent**。
5.  **回測**：載入訓練好的模型進行回測。
6.  **結果輸出**：
    *   回測數據顯示於終端機。
    *   績效圖表存於 `./results/portfolio_performance.png`。

## 4. 常見問題排除 (Troubleshooting)

*   **錯誤：`CUDA not available`**
    *   請確認 `nvidia-smi` 在 WSL2 中能正常顯示。
    *   確認安裝的 PyTorch 版本包含 CUDA 支援：
        ```bash
        python -c "import torch; print(torch.cuda.is_available())"
        ```
        若為 `False`，請重新安裝 PyTorch GPU 版。

*   **錯誤：`Out of Memory` (OOM)**
    *   RTX 4070 記憶體應足夠，但若發生此錯誤，請在程式碼中將 `batch_size` 從 64 調小至 32。

*   **中斷後續練**
    *   程式已內建續練機制。若訓練中斷，只需重新執行 `python ptrl_TW50_local_gpu.py`，它會自動偵測 `./models` 中的存檔並繼續訓練。
