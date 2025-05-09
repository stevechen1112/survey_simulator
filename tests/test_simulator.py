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
from simulator import SurveySimulator

class TestSurveySimulator(unittest.TestCase):
    """測試SurveySimulator類的功能"""
    
    def setUp(self):
        """每個測試用例執行前的設置"""
        # 創建臨時目錄
        self.temp_dir = tempfile.mkdtemp()
        
        # 創建測試數據
        self.test_data = pd.DataFrame({
            'id': list(range(1, 21)),
            '年齡': [25, 32, 41, 29, 35, 22, 45, 38, 27, 33, 29, 36, 43, 24, 39, 31, 26, 42, 28, 34],
            '性別': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女', '男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
            '教育程度': ['大學', '碩士', '高中', '大學', '大學', '大學', '碩士', '博士', '大學', '大學', '高中', '碩士', '大學', '大學', '高中', '碩士', '大學', '博士', '大學', '高中'],
            '月收入': [35000, 48000, 28000, 40000, 42000, 25000, 60000, 75000, 32000, 45000, 30000, 55000, 48000, 28000, 32000, 50000, 33000, 70000, 36000, 27000],
            '滿意度': [4, 5, 3, 4, 4, 3, 5, 4, 3, 4, 2, 5, 4, 3, 2, 4, 3, 5, 4, 2],
            '使用頻率': ['每天', '每天', '每週', '每天', '每天', '每週', '每天', '每天', '每月', '每週', '很少', '每天', '每週', '每週', '每月', '每天', '每週', '每天', '每週', '每月'],
            '推薦指數': [8, 9, 6, 7, 8, 5, 9, 8, 6, 7, 4, 9, 7, 6, 5, 8, 6, 9, 7, 4]
        })
        
        # 創建測試數據目錄
        os.makedirs(os.path.join(self.temp_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'output'), exist_ok=True)
        
        # 創建測試CSV文件
        self.test_csv_path = os.path.join(self.temp_dir, 'input', 'test_survey.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        # 設置輸出路徑
        self.output_csv_file = os.path.join(self.temp_dir, 'output', 'simulated_data.csv')
        self.output_xlsx_file = os.path.join(self.temp_dir, 'output', 'simulated_data.xlsx')
        
        # 初始化模擬器，使用固定的隨機種子以確保結果可重現
        self.simulator = SurveySimulator(random_seed=42)
    
    def tearDown(self):
        """每個測試用例執行後的清理"""
        # 刪除臨時目錄
        shutil.rmtree(self.temp_dir)
    
    def test_load_and_analyze(self):
        """測試數據加載和分析功能"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 驗證數據是否正確加載
        self.assertIsNotNone(self.simulator.original_data)
        self.assertEqual(len(self.simulator.original_data), 20)
        
        # 驗證處理後的數據和元數據
        self.assertIsNotNone(self.simulator.processed_data)
        self.assertIsNotNone(self.simulator.metadata)
        
        # 驗證條件概率計算
        self.assertIsNotNone(self.simulator.conditional_probs)
        self.assertTrue(len(self.simulator.conditional_probs) > 0)
    
    def test_generate_samples(self):
        """測試樣本生成功能"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 生成10個樣本（不創建可視化，以加快測試速度）
        samples = self.simulator.generate_samples(10, create_visuals=False)
        
        # 驗證生成的樣本數量
        self.assertEqual(len(samples), 10)
        
        # 驗證生成的樣本包含與原始數據相同的列
        for col in self.simulator.original_data.columns:
            self.assertIn(col, samples.columns)
        
        # 驗證生成的數值在合理範圍內
        if '年齡' in samples.columns:
            self.assertTrue(all(samples['年齡'] >= 18))  # 假設年齡至少為18
            self.assertTrue(all(samples['年齡'] <= 80))  # 假設年齡最大為80
        
        if '月收入' in samples.columns:
            self.assertTrue(all(samples['月收入'] >= 0))  # 收入不應為負
    
    def test_generate_samples_with_csv_output(self):
        """測試樣本生成並輸出到 CSV 文件"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 生成樣本並輸出到文件（不創建可視化）
        self.simulator.generate_samples(5, self.output_csv_file, create_visuals=False)
        
        # 驗證輸出文件是否存在
        self.assertTrue(os.path.exists(self.output_csv_file))
        
        # 驗證輸出文件內容
        output_data = pd.read_csv(self.output_csv_file)
        self.assertEqual(len(output_data), 5)
        for col in self.simulator.original_data.columns:
            self.assertIn(col, output_data.columns)

    def test_generate_samples_with_xlsx_output(self):
        """測試樣本生成並輸出到 XLSX 文件"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 生成樣本並輸出到 XLSX 文件（不創建可視化）
        self.simulator.generate_samples(5, self.output_xlsx_file, create_visuals=False)
        
        # 驗證輸出文件是否存在
        self.assertTrue(os.path.exists(self.output_xlsx_file))
        
        # 驗證輸出文件內容
        output_data = pd.read_excel(self.output_xlsx_file, engine='openpyxl')
        self.assertEqual(len(output_data), 5)
        for col in self.simulator.original_data.columns:
            self.assertIn(col, output_data.columns)
    
    def test_anomaly_detection(self):
        """測試異常檢測功能"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 獲取異常檢測模型
        model = self.simulator._train_anomaly_detector()
        
        # 驗證模型存在且可以進行預測
        self.assertIsNotNone(model)
        numeric_cols = [col for col, ctype in self.simulator.metadata['column_types'].items() 
                       if ctype in ['continuous'] and col in self.simulator.processed_data.columns]
        
        if numeric_cols and len(numeric_cols) >= 2:
            # 創建一個合理的樣本
            normal_sample = self.simulator.processed_data.iloc[0][numeric_cols].values.reshape(1, -1)
            pred = model.predict(normal_sample)
            self.assertIn(pred[0], [-1, 1])  # 預測結果應該是-1（異常）或1（正常）
    
    def test_generate_single_sample(self):
        """測試單個樣本生成功能"""
        self.simulator.load_and_analyze(self.test_csv_path)
        
        # 生成單個樣本
        sample = self.simulator._generate_single_sample_with_dependencies()
        
        # 驗證樣本是字典類型
        self.assertIsInstance(sample, dict)
        
        # 驗證樣本包含必要的鍵
        for col in self.simulator.processed_data.columns:
            if col != 'id':  # id可能在生成過程中不包括
                self.assertIn(col, sample)

if __name__ == '__main__':
    unittest.main() 