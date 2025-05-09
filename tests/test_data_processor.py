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
        self.test_data_dir = os.path.join(self.current_dir, 'test_data_files')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 創建測試數據
        self.test_data_df = pd.DataFrame({
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
        self.missing_data_df = self.test_data_df.copy()
        self.missing_data_df.loc[0, 'age'] = np.nan
        self.missing_data_df.loc[2, 'gender'] = np.nan
        self.missing_data_df.loc[4, 'income'] = np.nan
        
        # 創建測試CSV文件
        self.test_csv_path = os.path.join(self.test_data_dir, 'test_data.csv')
        self.test_data_df.to_csv(self.test_csv_path, index=False)
        
        # 創建測試XLSX文件
        self.test_xlsx_path = os.path.join(self.test_data_dir, 'test_data.xlsx')
        self.test_data_df.to_excel(self.test_xlsx_path, index=False, engine='openpyxl')
        
        # 創建測試XLS文件 (需要 xlwt 套件來寫入 .xls, pandas >= 1.3 不再支援寫入 .xls)
        # 為了測試讀取，我們可以手動創建一個簡單的 .xls 或使用舊版 pandas 創建
        # 這裡我們假設有一個預先存在的 .xls 檔案用於測試讀取，或者跳過寫入測試
        # 或者，我們可以模擬一個xls文件來測試load_data能否正確調用pd.read_excel(engine='xlrd')
        # 此處為了簡化，我們主要測試 .xlsx 的讀寫和 .csv 的讀寫

        # 初始化處理器
        self.processor = DataProcessor()
    
    def tearDown(self):
        """每個測試用例執行後的清理"""
        for f_path in [self.test_csv_path, self.test_xlsx_path]:
            if os.path.exists(f_path):
                os.remove(f_path)
        
        try:
            # 刪除測試數據目錄，如果它是空的
            if os.path.exists(self.test_data_dir) and not os.listdir(self.test_data_dir):
                 os.rmdir(self.test_data_dir)
        except OSError:
            pass # 如果目錄不為空或無法刪除，則忽略

    def _assert_data_loaded_correctly(self, loaded_data):
        self.assertEqual(len(loaded_data), 5)
        self.assertEqual(len(loaded_data.columns), 8)
        self.assertEqual(loaded_data['age'].iloc[0], 25)
        self.assertEqual(loaded_data['gender'].iloc[1], '女')
        pd.testing.assert_frame_equal(loaded_data, self.test_data_df.astype(loaded_data.dtypes))

    def test_load_csv_data(self):
        """測試CSV數據加載功能"""
        self.processor.load_data(self.test_csv_path)
        self._assert_data_loaded_correctly(self.processor.data)
    
    def test_load_xlsx_data(self):
        """測試XLSX數據加載功能"""
        self.processor.load_data(self.test_xlsx_path)
        self._assert_data_loaded_correctly(self.processor.data)

    # 為了測試 .xls 讀取，您需要一個 .xls 檔案。
    # 如果您有 xlrd 和一個名為 test_data.xls 的測試檔案在 self.test_data_dir 中，則可以取消註解以下測試：
    # def test_load_xls_data(self):
    #     """測試XLS數據加載功能 (需要 test_data.xls 文件和 xlrd)"""
    #     test_xls_path = os.path.join(self.test_data_dir, 'test_data.xls')
    #     if not os.path.exists(test_xls_path):
    #         self.skipTest(f"測試文件 {test_xls_path} 不存在，跳過 .xls 載入測試。")
    #     self.processor.load_data(test_xls_path)
    #     self._assert_data_loaded_correctly(self.processor.data)

    def test_load_unsupported_format(self):
        """測試加載不支持的文件格式"""
        unsupported_file_path = os.path.join(self.test_data_dir, 'test_data.txt')
        with open(unsupported_file_path, 'w') as f:
            f.write("this is a text file")
        with self.assertRaisesRegex(Exception, "不支援的文件格式: .txt"):
            self.processor.load_data(unsupported_file_path)
        os.remove(unsupported_file_path) # 清理創建的臨時文件

    def test_analyze_data(self):
        """測試數據分析功能"""
        self.processor.data = self.test_data_df.copy()
        self.processor.analyze_data()
        
        self.assertIn('age', self.processor.column_types)
        self.assertEqual(self.processor.column_types['age'], 'ordinal') 
        self.assertEqual(self.processor.column_types['gender'], 'categorical')
        self.assertEqual(self.processor.column_types['satisfaction'], 'ordinal')
        
        self.assertIn('age', self.processor.stats)
        self.assertIsInstance(self.processor.stats['age'], dict)
        self.assertIn(25, self.processor.stats['age'])

    def test_preprocess_data(self):
        """測試數據預處理功能"""
        self.processor.data = self.test_data_df.copy()
        self.processor.analyze_data()
        processed_data, metadata = self.processor.preprocess_data()
        self.assertEqual(len(processed_data), 5)
        self.assertIn('gender', self.processor.encoders)

    def test_decode_data(self):
        """測試數據解碼功能"""
        self.processor.data = self.test_data_df.copy()
        self.processor.analyze_data()
        processed_data, _ = self.processor.preprocess_data()
        decoded_data = self.processor.decode_data(processed_data)
        self.assertEqual(decoded_data['gender'].iloc[0], '男')

    def test_missing_data_handling(self):
        """測試缺失值處理"""
        self.processor.data = self.missing_data_df.copy()
        self.processor.analyze_data()
        processed_data, _ = self.processor.preprocess_data()
        self.assertFalse(processed_data.isnull().any().any())

    def test_get_column_suggestions(self):
        """測試列建議功能"""
        self.processor.data = self.test_data_df.copy()
        self.processor.analyze_data()
        numeric_cols_for_corr = []
        for col, col_type in self.processor.column_types.items():
            if col_type in ['continuous', 'ordinal', 'binary'] and pd.api.types.is_numeric_dtype(self.test_data_df[col].dtype):
                 numeric_cols_for_corr.append(col)
        if numeric_cols_for_corr:
            self.processor.correlations = self.test_data_df[numeric_cols_for_corr].corr()
        else:
            self.processor.correlations = pd.DataFrame()
        suggestions = self.processor.get_column_suggestions()
        self.assertIn('重要變量', suggestions)

if __name__ == '__main__':
    unittest.main() 