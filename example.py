#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
問卷模擬工具使用示例
這個腳本展示了如何使用問卷模擬工具的主要功能
"""

import os
import pandas as pd
from simulator import SurveySimulator
from data_processor import DataProcessor
from visualizer import SurveyVisualizer

# 設定輸入和輸出路徑
INPUT_CSV_FILE = "data/example_survey.csv"
INPUT_EXCEL_FILE = "data/example_survey.xlsx" # 新增 Excel 輸入檔案路徑
OUTPUT_DIR = "results/example_run"
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, "simulated_survey_output.csv")
OUTPUT_EXCEL_FILE = os.path.join(OUTPUT_DIR, "simulated_survey_output.xlsx") # 新增 Excel 輸出檔案路徑
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 為了示例，我們將創建一個示例Excel文件 (如果它不存在)
if not os.path.exists(INPUT_EXCEL_FILE):
    if os.path.exists(INPUT_CSV_FILE):
        try:
            df_temp = pd.read_csv(INPUT_CSV_FILE)
            # 確保 data 目錄存在
            os.makedirs(os.path.dirname(INPUT_EXCEL_FILE), exist_ok=True)
            df_temp.to_excel(INPUT_EXCEL_FILE, index=False, engine='openpyxl')
            print(f"示例 Excel 文件已創建: {INPUT_EXCEL_FILE}")
        except Exception as e:
            print(f"創建示例 Excel 文件時出錯: {e}")
    else:
        print(f"警告: 找不到 {INPUT_CSV_FILE}，無法創建示例 Excel 文件。")

def main():
    print("===== 問卷模擬示例 (使用CSV) =====")
    
    # 創建模擬器（設置隨機種子以確保結果可重現）
    simulator_csv = SurveySimulator(random_seed=42)
    
    # 載入數據並分析
    print("\n步驟 1: 載入和分析原始問卷數據 (CSV)")
    try:
        simulator_csv.load_and_analyze(INPUT_CSV_FILE)
    except Exception as e:
        print(f"錯誤：載入或分析 {INPUT_CSV_FILE} 失敗: {e}")
        return # 如果CSV載入失敗，則提前退出或跳過CSV部分
    
    # 獲取數據處理器
    processor_csv = simulator_csv.processor
    
    # 查看數據分析建議
    print("\n步驟 2: 檢視數據分析建議 (CSV)")
    suggestions = processor_csv.get_column_suggestions()
    
    print("\n重要變量:")
    for var in suggestions['重要變量']:
        print(f"  * {var}")
    
    print("\n可能冗餘的變量:")
    for var in suggestions['可能冗餘的變量']:
        print(f"  * {var}")
    
    print("\n建議分組方式:")
    for group in suggestions['建議分組方式']:
        print(f"  * {group}")
    
    # 生成模擬數據並保存為CSV
    print("\n步驟 3: 生成模擬數據 (100份問卷) 並保存為 CSV")
    simulated_data_csv = simulator_csv.generate_samples(count=100, output_file=OUTPUT_CSV_FILE, create_visuals=True)
    
    # 顯示部分生成結果
    print(f"\n生成的 CSV 數據預覽 (前5筆):")
    print(simulated_data_csv.head(5))
    print(f"\nCSV 結果已保存至: {OUTPUT_CSV_FILE}")

    print("\n\n===== 問卷模擬示例 (使用Excel) =====")
    if not os.path.exists(INPUT_EXCEL_FILE):
        print(f"錯誤: 找不到輸入的 Excel 文件 {INPUT_EXCEL_FILE}，跳過 Excel 示例部分。")
        return
        
    simulator_excel = SurveySimulator(random_seed=123) # 使用不同的種子以獲得不同的結果
    
    print("\n步驟 1: 載入和分析原始問卷數據 (Excel)")
    try:
        simulator_excel.load_and_analyze(INPUT_EXCEL_FILE)
    except Exception as e:
        print(f"錯誤：載入或分析 {INPUT_EXCEL_FILE} 失敗: {e}")
        return

    # 生成模擬數據並保存為Excel
    print("\n步驟 2: 生成模擬數據 (50份問卷) 並保存為 XLSX")
    simulated_data_excel = simulator_excel.generate_samples(count=50, output_file=OUTPUT_EXCEL_FILE, create_visuals=False) # 為了簡潔，禁用視覺化
    
    print(f"\n生成的 Excel 數據預覽 (前5筆):")
    print(simulated_data_excel.head(5))
    print(f"\nExcel 結果已保存至: {OUTPUT_EXCEL_FILE}")
    
    # 展示如何單獨使用可視化器
    print("\n步驟 4: 單獨使用可視化器")
    original_data = pd.read_csv(INPUT_CSV_FILE)
    visualizer = SurveyVisualizer(OUTPUT_DIR)
    
    # 只為特定列創建比較圖
    columns_of_interest = ['年齡', '滿意度', '推薦指數']
    visualizer.compare_distributions(
        original_data, 
        simulated_data_csv, 
        columns=columns_of_interest
    )
    
    print(f"特定列的比較圖已保存")
    
    print("\n===== 示例完成 =====")


if __name__ == "__main__":
    main() 