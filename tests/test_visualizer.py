#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

# 將主目錄添加到路徑中，以便可以導入模塊
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizer import SurveyVisualizer

class TestSurveyVisualizer(unittest.TestCase):
    """測試SurveyVisualizer類的功能"""
    
    def setUp(self):
        """每個測試用例執行前的設置"""
        # 創建臨時目錄作為輸出目錄
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建視覺化器
        self.visualizer = SurveyVisualizer(output_dir=self.output_dir)
        
        # 創建測試數據
        # 原始數據
        self.original_data = pd.DataFrame({
            'id': list(range(1, 11)),
            'age': [25, 32, 41, 29, 35, 22, 45, 38, 27, 33],
            'gender': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
            'education': ['大學', '碩士', '高中', '大學', '大學', '大學', '碩士', '博士', '大學', '大學'],
            'income': [35000, 48000, 28000, 40000, 42000, 25000, 60000, 75000, 32000, 45000],
            'satisfaction': [4, 5, 3, 4, 4, 3, 5, 4, 3, 4],
            'frequency': ['每天', '每天', '每週', '每天', '每天', '每週', '每天', '每天', '每月', '每週'],
            'recommend': [8, 9, 6, 7, 8, 5, 9, 8, 6, 7]
        })
        
        # 模擬數據
        self.simulated_data = pd.DataFrame({
            'id': list(range(1, 11)),
            'age': [28, 35, 40, 30, 33, 25, 42, 36, 29, 31],
            'gender': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
            'education': ['大學', '碩士', '高中', '大學', '碩士', '大學', '碩士', '大學', '大學', '碩士'],
            'income': [37000, 50000, 30000, 42000, 45000, 28000, 58000, 72000, 35000, 48000],
            'satisfaction': [4, 4, 3, 5, 4, 3, 5, 4, 3, 4],
            'frequency': ['每天', '每週', '每週', '每天', '每天', '每週', '每天', '每月', '每月', '每週'],
            'recommend': [8, 8, 7, 9, 8, 6, 9, 7, 6, 8]
        })
    
    def tearDown(self):
        """每個測試用例執行後的清理"""
        # 刪除臨時目錄
        shutil.rmtree(self.temp_dir)
    
    def test_compare_distributions(self):
        """測試分佈比較功能"""
        # 選擇要比較的列
        columns = ['age', 'gender', 'satisfaction']
        
        # 比較分佈
        self.visualizer.compare_distributions(
            self.original_data, 
            self.simulated_data, 
            columns=columns
        )
        
        # 驗證是否生成了可視化文件
        for column in columns:
            expected_file = os.path.join(self.output_dir, f'comparison_{column}.png')
            self.assertTrue(os.path.exists(expected_file), f"未找到文件: {expected_file}")
    
    def test_correlation_heatmap(self):
        """測試相關性熱力圖功能"""
        # 為原始數據創建相關性熱力圖
        self.visualizer.correlation_heatmap(self.original_data, '原始')
        
        # 驗證是否生成了熱力圖文件
        expected_file = os.path.join(self.output_dir, 'correlation_原始.png')
        self.assertTrue(os.path.exists(expected_file), f"未找到文件: {expected_file}")
        
        # 為模擬數據創建相關性熱力圖
        self.visualizer.correlation_heatmap(self.simulated_data, '模擬')
        
        # 驗證是否生成了熱力圖文件
        expected_file = os.path.join(self.output_dir, 'correlation_模擬.png')
        self.assertTrue(os.path.exists(expected_file), f"未找到文件: {expected_file}")
    
    def test_create_summary_report(self):
        """測試摘要報告創建功能"""
        # 創建摘要報告
        self.visualizer.create_summary_report(self.original_data, self.simulated_data)
        
        # 驗證是否生成了報告文件
        expected_file = os.path.join(self.output_dir, 'summary_report.md')
        self.assertTrue(os.path.exists(expected_file), f"未找到文件: {expected_file}")
        
        # 檢查報告內容是否包含必要的信息
        with open(expected_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 檢查必要的標題
            self.assertIn('# 數據模擬結果比較報告', content)
            
            # 檢查是否包含數值型變量的比較
            for column in ['age', 'income', 'satisfaction', 'recommend']:
                self.assertIn(f'## {column} 統計比較', content)
            
            # 檢查是否包含類別變量的比較
            for column in ['gender', 'education', 'frequency']:
                self.assertIn(f'## {column} 分佈比較', content)
    
    def test_auto_column_selection(self):
        """測試自動列選擇功能"""
        # 不指定要比較的列，讓視覺化器自動選擇
        self.visualizer.compare_distributions(self.original_data, self.simulated_data)
        
        # 檢查是否為適合的列生成了可視化
        for column in ['gender', 'education', 'satisfaction', 'frequency']:
            expected_file = os.path.join(self.output_dir, f'comparison_{column}.png')
            self.assertTrue(os.path.exists(expected_file), f"未找到文件: {expected_file}")
        
        # 確保沒有為id列生成可視化
        id_file = os.path.join(self.output_dir, 'comparison_id.png')
        self.assertFalse(os.path.exists(id_file), f"不應生成id列的可視化: {id_file}")

if __name__ == '__main__':
    unittest.main() 