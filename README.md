# 問卷模擬生成工具

基於已有的受測者數據（CSV或Excel格式），使用AI模型生成額外多元受測者的問卷回覆。

## 功能

- 讀取現有問卷數據（支援 .csv, .xlsx, .xls 格式）
- 分析回答模式和分佈
- 檢查數據質量和提供報告
- 生成符合原始數據特徵的模擬回覆
- 利用條件概率模型保持變量間相關性
- 生成數據可視化和比較報告
- 輸出易於整合的標準格式

## 使用方法

1. 安裝依賴：`pip install -r requirements.txt`
2. 準備數據：將50位受測者的CSV數據文件放在`data`目錄
3. 執行模擬：
   ```
   python simulator.py --input data/survey_data.csv --output results/simulated_data.csv --count 100
   ```
   或者使用Excel檔案：
   ```
   python simulator.py --input data/survey_data.xlsx --output results/simulated_data.xlsx --count 100
   ```
4. 進階選項：
   ```
   python simulator.py --input data/example_survey.csv --output data/simulated_data.csv --count 100 --seed 42 --no-visuals
   ```
5. 結果將存儲在指定的輸出文件中，視覺化報告保存在輸出目錄

## 視覺化與報告

執行模擬後，程式會在輸出目錄中自動生成以下內容：

- 每個變量的分佈比較圖
- 原始數據與模擬數據的相關性熱力圖
- 詳細的數據統計比較報告（markdown格式）

## 參數說明

- `--input`: 輸入數據文件路徑（.csv, .xlsx, .xls 格式，必須）
- `--output`: 輸出數據文件路徑（.csv 或 .xlsx 格式，必須）
- `--count`: 要生成的模擬樣本數量（默認100）
- `--seed`: 隨機種子，用於確保結果可重現
- `--no-visuals`: 添加此標誌以禁用視覺化報告生成
