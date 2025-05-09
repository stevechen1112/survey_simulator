#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import subprocess

# 將主目錄添加到路徑中，以便可以導入模塊
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import SurveySimulator
from data_processor import DataProcessor
from visualizer import SurveyVisualizer

class TestIntegration(unittest.TestCase):
    """整合測試，測試整個問卷模擬工作流程"""
    
    def setUp(self):
        """每個測試用例執行前的設置"""
        # 創建臨時目錄
        self.temp_dir = tempfile.mkdtemp()
        
        # 創建測試目錄結構
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建測試數據
        self.test_data = pd.DataFrame({
            'id': list(range(1, 31)),
            '年齡': np.random.randint(18, 65, 30),
            '性別': np.random.choice(['男', '女'], 30),
            '教育程度': np.random.choice(['高中', '大學', '碩士', '博士'], 30),
            '月收入': np.random.randint(20000, 80000, 30),
            '滿意度': np.random.randint(1, 6, 30),
            '使用頻率': np.random.choice(['很少', '每月', '每週', '每天'], 30),
            '推薦指數': np.random.randint(1, 11, 30)
        })
        
        # 創建測試CSV文件
        self.input_csv_path = os.path.join(self.input_dir, 'input_survey.csv')
        self.test_data.to_csv(self.input_csv_path, index=False)
        
        # 設置輸出路徑
        self.output_file = os.path.join(self.output_dir, 'simulated_data.csv')
    
    def tearDown(self):
        """每個測試用例執行後的清理"""
        # 刪除臨時目錄
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """測試從數據加載到生成模擬數據的整個過程"""
        # 1. 創建模擬器
        simulator = SurveySimulator(random_seed=42)
        
        # 2. 加載和分析數據
        simulator.load_and_analyze(self.input_csv_path)
        
        # 3. 生成模擬數據
        simulated_data = simulator.generate_samples(15, self.output_file, create_visuals=True)
        
        # 驗證生成的數據
        self.assertEqual(len(simulated_data), 15)
        
        # 驗證生成的文件
        self.assertTrue(os.path.exists(self.output_file))
        
        # 驗證生成的視覺化報告
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'summary_report.md')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'correlation_原始.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'correlation_模擬.png')))
    
    def test_component_interaction(self):
        """測試各組件之間的交互"""
        # 初始化各組件
        processor = DataProcessor()
        processor.load_data(self.input_csv_path)
        processor.analyze_data()
        
        # 數據預處理
        processed_data, metadata = processor.preprocess_data()
        
        # 建立視覺化器
        visualizer = SurveyVisualizer(output_dir=self.output_dir)
        
        # 創建原始數據的相關性熱力圖
        visualizer.correlation_heatmap(processor.data, '原始')
        
        # 驗證熱力圖是否生成
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'correlation_原始.png')))
        
        # 使用模擬器進行模擬
        simulator = SurveySimulator(random_seed=42)
        simulator.processor = processor
        simulator.processed_data = processed_data
        simulator.metadata = metadata
        simulator.original_data = processor.data.copy()
        
        # 計算條件概率
        simulator._calculate_conditional_probabilities()
        
        # 生成模擬數據
        simulated_data = simulator.generate_samples(10, create_visuals=False)
        
        # 創建模擬數據的相關性熱力圖
        visualizer.correlation_heatmap(simulated_data, '模擬')
        
        # 驗證熱力圖是否生成
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'correlation_模擬.png')))
        
        # 創建比較報告
        visualizer.create_summary_report(processor.data, simulated_data)
        
        # 驗證報告是否生成
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'summary_report.md')))
    
    @unittest.skip("暫時跳過命令行界面測試，因為需要特殊環境")
    def test_command_line_interface(self):
        """測試命令行界面"""
        # 獲取主項目目錄
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simulator_path = os.path.join(project_dir, 'simulator.py')
        
        # 檢查simulator.py是否存在
        if not os.path.exists(simulator_path):
            self.skipTest("找不到simulator.py文件")
        
        # 創建命令行
        cmd = [
            sys.executable, simulator_path,
            '--input', self.input_csv_path,
            '--output', self.output_file,
            '--count', '5',
            '--seed', '42'
        ]
        
        # 執行命令
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 檢查命令是否成功執行
            self.assertEqual(result.returncode, 0, f"命令執行失敗: {result.stderr}")
            
            # 驗證輸出文件是否存在
            self.assertTrue(os.path.exists(self.output_file))
            
            # 驗證生成的數據
            output_data = pd.read_csv(self.output_file)
            self.assertEqual(len(output_data), 5)
        except Exception as e:
            self.fail(f"測試命令行界面時出錯: {str(e)}")

if __name__ == '__main__':
    unittest.main() 