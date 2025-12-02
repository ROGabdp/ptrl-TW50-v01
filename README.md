# Pro Trader RL (TW50 Version)

這是一個基於深度強化學習 (Deep Reinforcement Learning, DRL) 的自動化交易策略專案，專門針對 **台灣 50 (TW50)** 成分股進行訓練與回測。

本專案參考了學術論文 "Pro Trader RL" 的架構，並針對台股市場特性進行了實作與優化。核心採用 **雙 Agent 架構**：一個負責「買入決策」，另一個負責「賣出決策」，旨在解決傳統單一模型難以兼顧進出場邏輯的問題。

## 🚀 功能特色 (Key Features)

*   **雙 Agent 協同架構 (Dual-Agent System)**
    *   **Buy Agent**: 專注於從市場中篩選具備上漲潛力的股票。
    *   **Sell Agent**: 專注於持倉管理，決定最佳獲利了結或停損時機。
*   **專業特徵工程 (Advanced Feature Engineering)**
    *   整合多種技術指標：Donchian Channel, SuperTrend, RSI, MFI, ATR。
    *   引入相對強弱指標 (Relative Strength, RS)：比較個股與大盤 (TWII) 的強弱關係。
    *   論文級正規化處理：確保數據適合神經網路訓練。
*   **多版本實作**
    *   **Paper Version**: 嚴格遵照論文邏輯與參數設定，適合學術驗證。
    *   **Optimized Version**: 針對台股特性優化獎勵函數 (Reward Function) 與風險控管機制。
*   **完整回測系統**
    *   內建資金管理模組 (Position Sizing)。
    *   自動化績效分析與圖表繪製 (vs 0050 Benchmark)。

## 📂 檔案結構 (File Structure)

| 檔案名稱 | 說明 |
| :--- | :--- |
| `ptrl_TW50_paper_version.py` | **[核心]** 論文復現版主程式。包含完整的訓練流程 (Training) 與回測 (Backtesting)。 |
| `ptrl_TW50_optimized.py` | **[進階]** 優化版主程式。改進了獎勵機制與狀態定義，更貼近實戰需求。 |
| `ptrl_TW50_backtest.py` | **[工具]** 獨立回測腳本。用於載入已訓練好的模型 (`models/`) 進行快速驗證，不需重新訓練。 |
| `ptrl_TW50_local_gpu.py` | 本地端 GPU 訓練腳本，支援中斷後接續訓練 (Resume Training)。 |
| `requirements.txt` | 專案依賴套件列表。 |
| `models/` | (自動產生) 存放訓練好的模型檔案 (`.zip`)。 |
| `results/` | (自動產生) 存放回測報告、績效圖表與 TensorBoard Log。 |

## 🛠️ 安裝與環境設定 (Installation)

本專案建議使用 Python 3.8 以上版本。

1.  **複製專案 (Clone Repository)**
    ```bash
    git clone https://github.com/ROGabdp/ptrl-TW50-v01.git
    cd ptrl-TW50-v01
    ```

2.  **建立虛擬環境 (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：若您有 NVIDIA 顯卡，請確保已安裝對應版本的 PyTorch (CUDA support) 以加速訓練。*

## 📖 使用教學 (Usage)

### 1. 訓練模型 (Training)

若要執行嚴格論文版本的訓練：
```bash
python ptrl_TW50_paper_version.py
```
程式會自動下載 TW50 股票資料，進行特徵工程，並開始訓練 Buy Agent 與 Sell Agent。訓練過程中會自動儲存模型至 `models_paper/`。

### 2. 執行回測 (Backtesting)

若已有訓練好的模型，可直接執行回測：
```bash
python ptrl_TW50_backtest.py
```
回測結束後，程式會輸出：
*   期初與期末資金
*   總報酬率 (ROI)
*   交易明細 (Trade Logs)
*   績效走勢圖 (儲存於 `results/` 資料夾)

## ⚠️ 免責聲明 (Disclaimer)

本專案僅供 **學術研究** 與 **程式交易演算法開發交流** 之用。
*   專案內的策略邏輯與回測結果僅供參考，**不保證未來獲利**。
*   使用者應自行承擔實際交易的風險，作者不對任何投資損失負責。

---
Developed by [Phil Liang](mailto:Phil_Liang@asus.com)
