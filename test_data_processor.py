#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
from io import StringIO

# 將主目錄添加到路徑中，以便可以導入模塊
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """測試DataProcessor類的功能"""
    
    def setUp(self):
        """每個測試用例執行前的設置"""
        # 獲取當前目錄
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 創建測試數據
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'age': [25, 32, 41, 29, 35],
            'gender': ['男', '女', '男', '女', '男'],
            'education': ['大學', '碩士', '高中', '大學', '大學'],
            'income': [35000, 48000, 28000, 40000, 42000],
            'satisfaction': [4, 5, 3, 4, 4],
            'frequency': ['每天', '每天', '每週', '每天', '每天'],
            'recommend': [8, 9, 6, 7, 8]
        })
        
        # 創建一個具有缺失值的數據集
        self.missing_data = self.test_data.copy()
        self.missing_data.loc[0, 'age'] = np.nan
        self.missing_data.loc[2, 'gender'] = np.nan
        self.missing_data.loc[4, 'income'] = np.nan
        
        # 創建測試數據目錄
        os.makedirs(os.path.join(self.current_dir, 'test_data'), exist_ok=True)
        
        # 創建測試CSV文件
        self.test_csv_path = os.path.join(self.current_dir, 'test_data', 'test_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        # 初始化處理器
        self.processor = DataProcessor()
    
    def tearDown(self):
        """每個測試用例執行後的清理"""
        # 刪除測試CSV文件
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        
        # 嘗試刪除測試數據目錄
        try:
            os.rmdir(os.path.join(self.current_dir, 'test_data'))
        except OSError:
            pass  # 如果目錄不為空，忽略錯誤
    
    def test_load_data(self):
        """測試數據加載功能"""
        self.processor.load_data(self.test_csv_path)
        
        # 驗證數據是否正確加載
        self.assertEqual(len(self.processor.data), 5)
        self.assertEqual(len(self.processor.data.columns), 8)
        self.assertEqual(self.processor.data['age'].iloc[0], 25)
        self.assertEqual(self.processor.data['gender'].iloc[1], '女')
    
    def test_analyze_data(self):
        """測試數據分析功能"""
        self.processor.data = self.test_data
        self.processor.analyze_data()
        
        # 驗證數據類型識別
        self.assertIn('age', self.processor.column_types)
        # 修正預期的類型：在我們的測試數據中，age只有5個唯一值，所以會被檢測為'ordinal'
        self.assertEqual(self.processor.column_types['age'], 'ordinal')
        self.assertEqual(self.processor.column_types['gender'], 'categorical')
        self.assertEqual(self.processor.column_types['satisfaction'], 'ordinal')
        
        # 驗證統計信息計算
        self.assertIn('age', self.processor.stats)
        # 對於ordinal類型，統計信息是以字典形式存儲的
        self.assertIsInstance(self.processor.stats['age'], dict)
        
        # 檢查類別變量統計
        self.assertIn('gender', self.processor.stats)
        self.assertIn('男', self.processor.stats['gender'])
        self.assertIn('女', self.processor.stats['gender'])
    
    def test_preprocess_data(self):
        """測試數據預處理功能"""
        self.processor.data = self.test_data
        self.processor.analyze_data()
        processed_data, metadata = self.processor.preprocess_data()
        
        # 驗證數據預處理結果
        self.assertEqual(len(processed_data), 5)
        self.assertEqual(len(processed_data.columns), 8)
        
        # 驗證元數據
        self.assertIn('column_types', metadata)
        self.assertIn('stats', metadata)
        self.assertIn('encoders', metadata)
        
        # 確認分類變量已編碼
        self.assertIn('gender', self.processor.encoders)
        self.assertIn('education', self.processor.encoders)
        self.assertIn('frequency', self.processor.encoders)
    
    def test_decode_data(self):
        """測試數據解碼功能"""
        self.processor.data = self.test_data
        self.processor.analyze_data()
        processed_data, _ = self.processor.preprocess_data()
        
        # 解碼處理後的數據
        decoded_data = self.processor.decode_data(processed_data)
        
        # 驗證解碼是否正確
        self.assertEqual(decoded_data['gender'].iloc[0], '男')
        self.assertEqual(decoded_data['gender'].iloc[1], '女')
        self.assertEqual(decoded_data['education'].iloc[0], '大學')
        self.assertEqual(decoded_data['education'].iloc[2], '高中')
    
    def test_missing_data_handling(self):
        """測試缺失值處理"""
        self.processor.data = self.missing_data
        self.processor.analyze_data()
        processed_data, _ = self.processor.preprocess_data()
        
        # 驗證缺失值已被處理
        self.assertFalse(processed_data.isnull().any().any())
    
    def test_get_column_suggestions(self):
        """測試列建議功能"""
        # 使用只包含數值列的數據創建相關性矩陣
        numeric_data = self.test_data[['age', 'income', 'satisfaction', 'recommend']]
        
        # 設置處理器數據和相關性矩陣
        self.processor.data = self.test_data
        self.processor.analyze_data()
        self.processor.correlations = numeric_data.corr()
        
        # 獲取建議
        suggestions = self.processor.get_column_suggestions()
        
        # 驗證建議結構
        self.assertIn('重要變量', suggestions)
        self.assertIn('可能冗餘的變量', suggestions)
        self.assertIn('建議分組方式', suggestions)

if __name__ == '__main__':
    unittest.main() 