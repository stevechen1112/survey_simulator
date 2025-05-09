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
INPUT_FILE = "data/example_survey.csv"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "simulated_survey.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("===== 問卷模擬示例 =====")
    
    # 創建模擬器（設置隨機種子以確保結果可重現）
    simulator = SurveySimulator(random_seed=42)
    
    # 載入數據並分析
    print("\n步驟 1: 載入和分析原始問卷數據")
    simulator.load_and_analyze(INPUT_FILE)
    
    # 獲取數據處理器
    processor = simulator.processor
    
    # 查看數據分析建議
    print("\n步驟 2: 檢視數據分析建議")
    suggestions = processor.get_column_suggestions()
    
    print("\n重要變量:")
    for var in suggestions['重要變量']:
        print(f"  * {var}")
    
    print("\n可能冗餘的變量:")
    for var in suggestions['可能冗餘的變量']:
        print(f"  * {var}")
    
    print("\n建議分組方式:")
    for group in suggestions['建議分組方式']:
        print(f"  * {group}")
    
    # 生成模擬數據
    print("\n步驟 3: 生成模擬數據 (100份問卷)")
    simulated_data = simulator.generate_samples(count=100, output_file=OUTPUT_FILE, create_visuals=True)
    
    # 顯示部分生成結果
    print(f"\n生成的數據預覽 (前5筆):")
    print(simulated_data.head(5))
    
    # 顯示結果保存位置
    print(f"\n完整結果已保存至: {OUTPUT_FILE}")
    print(f"可視化報告保存在: {OUTPUT_DIR}")
    
    # 展示如何單獨使用可視化器
    print("\n步驟 4: 單獨使用可視化器")
    original_data = pd.read_csv(INPUT_FILE)
    visualizer = SurveyVisualizer(OUTPUT_DIR)
    
    # 只為特定列創建比較圖
    columns_of_interest = ['年齡', '滿意度', '推薦指數']
    visualizer.compare_distributions(
        original_data, 
        simulated_data, 
        columns=columns_of_interest
    )
    
    print(f"特定列的比較圖已保存")
    
    print("\n===== 示例完成 =====")


if __name__ == "__main__":
    main() 