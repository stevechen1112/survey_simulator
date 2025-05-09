import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import warnings


class DataProcessor:
    """處理問卷數據，包括讀取、分析和預處理"""
    
    def __init__(self):
        self.data = None
        self.column_types = {}
        self.encoders = {}
        self.stats = {}
        self.correlations = None
        self.data_quality_report = {}
    
    def load_data(self, file_path: str) -> None:
        """讀取CSV格式的問卷數據"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"成功載入數據：{len(self.data)}條記錄，{len(self.data.columns)}個問題")
            
            # 自動進行數據質量檢查
            self._check_data_quality()
        except Exception as e:
            raise Exception(f"載入數據時出錯：{str(e)}")
    
    def _check_data_quality(self) -> None:
        """檢查數據質量，包括缺失值、重複值和一致性問題"""
        if self.data is None:
            return
            
        quality_report = {
            'missing_values': {},
            'duplicates': 0,
            'inconsistencies': [],
            'warnings': []
        }
        
        # 檢查缺失值
        missing_counts = self.data.isnull().sum()
        for column, count in missing_counts.items():
            if count > 0:
                quality_report['missing_values'][column] = {
                    'count': int(count),
                    'percentage': float(count / len(self.data) * 100)
                }
        
        # 檢查重複行
        duplicates = self.data.duplicated().sum()
        quality_report['duplicates'] = int(duplicates)
        
        # 檢查數據一致性問題
        # 例如，檢查數值資料是否在合理範圍內
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                # 檢查是否存在極端值（超過3個標準差）
                mean = self.data[column].mean()
                std = self.data[column].std()
                if std > 0:  # 避免除以零
                    extreme_values = self.data[
                        (self.data[column] > mean + 3*std) | 
                        (self.data[column] < mean - 3*std)
                    ]
                    if len(extreme_values) > 0:
                        quality_report['warnings'].append(
                            f"列 '{column}' 有 {len(extreme_values)} 個極端值 (超過3個標準差)"
                        )
        
        # 檢查分類變量的一致性
        # 例如，假設年齡和教育程度應該有某種關係
        if 'age' in self.data.columns and 'education' in self.data.columns:
            # 這只是示例邏輯，實際檢查應根據數據特點調整
            pass
            
        # 存儲質量報告
        self.data_quality_report = quality_report
        
        # 顯示質量報告摘要
        print("\n數據質量報告摘要:")
        print(f"- 缺失值: 發現 {len(quality_report['missing_values'])} 列有缺失")
        if quality_report['missing_values']:
            for col, info in quality_report['missing_values'].items():
                print(f"  * {col}: {info['count']} 缺失 ({info['percentage']:.1f}%)")
        
        print(f"- 重複行: {quality_report['duplicates']} 行")
        
        if quality_report['warnings']:
            print("- 警告:")
            for warning in quality_report['warnings']:
                print(f"  * {warning}")
        
        print("\n")
    
    def analyze_data(self) -> None:
        """分析數據類型並計算統計信息"""
        if self.data is None:
            raise Exception("請先載入數據")
        
        print("開始數據分析...")
        
        # 識別每列的數據類型
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                # 文本/分類數據
                self.column_types[column] = 'categorical'
                # 計算各選項頻率
                self.stats[column] = self.data[column].value_counts(normalize=True).to_dict()
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                if set(self.data[column].dropna().unique()) <= {0, 1}:
                    # 二元問題
                    self.column_types[column] = 'binary'
                    self.stats[column] = self.data[column].value_counts(normalize=True).to_dict()
                elif len(self.data[column].dropna().unique()) <= 10:
                    # 有限級別問題（如李克特量表）
                    self.column_types[column] = 'ordinal'
                    self.stats[column] = self.data[column].value_counts(normalize=True).to_dict()
                else:
                    # 連續數值
                    self.column_types[column] = 'continuous'
                    self.stats[column] = {
                        'mean': self.data[column].mean(),
                        'std': self.data[column].std(),
                        'min': self.data[column].min(),
                        'max': self.data[column].max(),
                        'median': self.data[column].median(),
                        'skew': float(self.data[column].skew()),  # 偏度
                        'kurtosis': float(self.data[column].kurtosis()),  # 峰度
                        'q1': float(self.data[column].quantile(0.25)),  # 第一四分位數
                        'q3': float(self.data[column].quantile(0.75))   # 第三四分位數
                    }
        
        # 計算問題間相關性（對於數值類問題）
        numeric_columns = [col for col, ctype in self.column_types.items() 
                           if ctype in ['continuous', 'ordinal', 'binary']]
        if numeric_columns:
            self.correlations = self.data[numeric_columns].corr()
            
            # 輸出主要相關性
            strong_correlations = []
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # 避免重複
                        corr = self.correlations.loc[col1, col2]
                        if abs(corr) >= 0.5:  # 只考慮強相關
                            strong_correlations.append((col1, col2, corr))
            
            if strong_correlations:
                print("\n強相關性變量對:")
                for col1, col2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                    print(f"  * {col1} 與 {col2}: {corr:.2f}")
            
            print("\n")
        
        # 分析類別變量的分佈
        categorical_columns = [col for col, ctype in self.column_types.items() 
                              if ctype in ['categorical', 'binary', 'ordinal']]
        
        if categorical_columns:
            print("類別變量分佈:")
            for col in categorical_columns:
                dist = self.stats[col]
                print(f"  * {col}:")
                for category, freq in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]:  # 只顯示前5個類別
                    print(f"    - {category}: {freq:.1%}")
            print("\n")
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """將分類變量編碼為數值，準備建模"""
        if self.data is None or not self.column_types:
            raise Exception("請先載入並分析數據")
        
        processed_data = self.data.copy()
        
        # 對分類變量進行編碼
        for column, ctype in self.column_types.items():
            if ctype == 'categorical':
                encoder = LabelEncoder()
                non_null_mask = processed_data[column].notna()
                if non_null_mask.any():
                    processed_data.loc[non_null_mask, column] = encoder.fit_transform(
                        processed_data.loc[non_null_mask, column]
                    )
                    self.encoders[column] = encoder
        
        # 處理缺失值
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for column in processed_data.columns:
                if processed_data[column].isnull().any():
                    ctype = self.column_types.get(column)
                    
                    if ctype == 'continuous':
                        # 對連續變量使用中位數填充
                        median_value = processed_data[column].median()
                        processed_data[column].fillna(median_value, inplace=True)
                    else:
                        # 對分類變量使用眾數填充
                        mode_value = processed_data[column].mode()[0]
                        processed_data[column].fillna(mode_value, inplace=True)
        
        metadata = {
            'column_types': self.column_types,
            'stats': self.stats,
            'correlations': self.correlations,
            'encoders': self.encoders,
            'quality_report': self.data_quality_report
        }
        
        return processed_data, metadata
    
    def decode_data(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """將編碼後的數據轉換回原始格式"""
        decoded_data = encoded_data.copy()
        
        for column, encoder in self.encoders.items():
            non_null_mask = decoded_data[column].notna()
            if non_null_mask.any():
                # 確保數據是整數型的，安全地處理可能的浮點值和字符串值
                try:
                    # 嘗試先轉換為浮點數再轉為整數
                    values = pd.to_numeric(decoded_data.loc[non_null_mask, column], errors='coerce')
                    values = values.round().astype(int, errors='ignore')
                    decoded_data.loc[non_null_mask, column] = encoder.inverse_transform(values)
                except (ValueError, TypeError) as e:
                    # 如果轉換失敗，記錄警告但不中斷處理
                    print(f"警告: 轉換列 '{column}' 時出錯: {str(e)}")
                    print(f"此列的數據類型: {decoded_data[column].dtype}")
                    print(f"此列的值示例: {decoded_data[column].iloc[0] if len(decoded_data) > 0 else 'N/A'}")
                    # 跳過此列的轉換
                    continue
        
        return decoded_data
    
    def get_column_suggestions(self) -> Dict[str, List[str]]:
        """根據數據分析提供分類建議"""
        if self.data is None or not self.column_types:
            raise Exception("請先載入並分析數據")
        
        suggestions = {
            '重要變量': [],
            '可能冗餘的變量': [],
            '建議分組方式': []
        }
        
        # 識別重要變量（基於相關性和變異性）
        if self.correlations is not None:
            # 找出與多個其他變量相關的變量
            correlation_counts = {}
            for col in self.correlations.columns:
                # 計算與其他變量有強相關的數量
                strong_correlations = sum(1 for other_col in self.correlations.columns 
                                        if col != other_col and abs(self.correlations.loc[col, other_col]) >= 0.5)
                correlation_counts[col] = strong_correlations
            
            # 選擇相關性最強的變量
            important_vars = sorted(correlation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            suggestions['重要變量'] = [var for var, count in important_vars if count > 0]
        
        # 識別可能冗餘的變量（高度相關的對）
        redundant_pairs = []
        if self.correlations is not None:
            for i, col1 in enumerate(self.correlations.columns):
                for col2 in self.correlations.columns[i+1:]:
                    if abs(self.correlations.loc[col1, col2]) >= 0.9:  # 高度相關閾值
                        redundant_pairs.append((col1, col2))
        
        if redundant_pairs:
            # 對於每對高度相關的變量，選擇一個可能是冗餘的
            for col1, col2 in redundant_pairs:
                # 簡單策略：選擇名稱較長的或詞彙序靠後的
                if len(col1) > len(col2) or col1 > col2:
                    suggestions['可能冗餘的變量'].append(col1)
                else:
                    suggestions['可能冗餘的變量'].append(col2)
        
        # 提出變量分組方式（基於相關性聚類）
        # 這裡只是一個簡單的示例，實際上可能需要更複雜的聚類算法
        if self.correlations is not None and len(self.correlations) > 2:
            groups = []
            remaining_cols = list(self.correlations.columns)
            
            while remaining_cols:
                current_group = [remaining_cols.pop(0)]
                
                i = 0
                while i < len(remaining_cols):
                    col = remaining_cols[i]
                    # 檢查與當前組中任一變量的相關性
                    related = any(abs(self.correlations.loc[col, group_col]) >= 0.4 for group_col in current_group)
                    
                    if related:
                        current_group.append(col)
                        remaining_cols.pop(i)
                    else:
                        i += 1
                
                if len(current_group) > 1:
                    groups.append(current_group)
            
            for i, group in enumerate(groups, 1):
                if len(group) >= 2:  # 只考慮至少有2個變量的組
                    suggestions['建議分組方式'].append(f"組{i}: {', '.join(group)}")
        
        return suggestions
