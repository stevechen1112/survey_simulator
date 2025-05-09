#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from faker import Faker
from tqdm import tqdm
from data_processor import DataProcessor
from visualizer import SurveyVisualizer


class SurveySimulator:
    """基於現有問卷數據生成模擬受測者回覆"""
    
    def __init__(self, random_seed=None):
        self.faker = Faker('zh_TW')  # 使用繁體中文區域設置
        self.processor = DataProcessor()
        self.visualizer = None  # 視覺化器將按需初始化
        self.processed_data = None
        self.metadata = None
        self.original_data = None
        
        # 設置隨機種子以確保結果可重現
        if random_seed is not None:
            np.random.seed(random_seed)
            self.faker.seed_instance(random_seed)
    
    def load_and_analyze(self, input_file: str) -> None:
        """載入並分析現有問卷數據"""
        self.processor.load_data(input_file)
        self.processor.analyze_data()
        self.original_data = self.processor.data.copy()
        self.processed_data, self.metadata = self.processor.preprocess_data()
        
        # 計算條件概率
        self._calculate_conditional_probabilities()
    
    def _calculate_conditional_probabilities(self):
        """計算重要變量之間的條件概率關係"""
        self.conditional_probs = {}
        
        # 確定可能的條件關係
        numeric_columns = [col for col, ctype in self.metadata['column_types'].items() 
                          if ctype in ['continuous', 'ordinal', 'binary']]
        
        cat_columns = [col for col, ctype in self.metadata['column_types'].items() 
                      if ctype in ['categorical', 'ordinal', 'binary']]
        
        # 對於每對類別變量，計算條件概率
        for i, col1 in enumerate(cat_columns):
            for col2 in cat_columns[i+1:]:
                # 確保兩列有足夠的數據進行分析
                if self.original_data[col1].nunique() <= 10 and self.original_data[col2].nunique() <= 10:
                    try:
                        # 創建條件概率表
                        cond_prob = pd.crosstab(
                            self.original_data[col1], 
                            self.original_data[col2], 
                            normalize='index'
                        )
                        
                        # 存儲為字典格式，避免使用DataFrame的index相關操作
                        cond_prob_dict = {}
                        for idx in cond_prob.index:
                            row_dict = {}
                            for col in cond_prob.columns:
                                row_dict[col] = cond_prob.loc[idx, col]
                            cond_prob_dict[idx] = row_dict
                        
                        self.conditional_probs[(col1, col2)] = cond_prob_dict
                    except Exception as e:
                        print(f"警告: 計算 '{col1}' 和 '{col2}' 的條件概率時出錯: {str(e)}")
        
        # 查找連續變量和類別變量之間的關係
        for num_col in numeric_columns:
            for cat_col in cat_columns:
                try:
                    # 對於每個類別值，計算數值變量的統計特性
                    stats = {}
                    for category in self.original_data[cat_col].unique():
                        subset = self.original_data[self.original_data[cat_col] == category]
                        if len(subset) >= 5:  # 確保有足夠的數據
                            stats[category] = {
                                'mean': subset[num_col].mean(),
                                'std': max(subset[num_col].std(), 0.01),  # 防止標準差為零
                                'min': subset[num_col].min(),
                                'max': subset[num_col].max()
                            }
                    
                    if stats:
                        self.conditional_probs[(cat_col, num_col)] = stats
                except Exception as e:
                    print(f"警告: 計算 '{cat_col}' 和 '{num_col}' 的條件關係時出錯: {str(e)}")
    
    def generate_samples(self, count: int, output_file: str = None, create_visuals: bool = True) -> pd.DataFrame:
        """生成指定數量的模擬問卷回覆"""
        if self.processed_data is None or self.metadata is None:
            raise Exception("請先載入並分析數據")
        
        print(f"正在生成{count}位模擬受測者的回覆...")
        
        # 創建空的DataFrame來存儲結果
        simulated_data = pd.DataFrame(columns=self.processed_data.columns)
        
        # 使用隔離森林檢測異常值
        isolation_model = self._train_anomaly_detector()
        
        # 為每個待生成的樣本生成數據
        valid_samples = 0
        attempts = 0
        max_attempts = count * 3  # 最多嘗試次數是請求樣本數的3倍
        
        # 選擇用於異常檢測的數值列
        numeric_cols = [col for col, ctype in self.metadata['column_types'].items() 
                       if ctype in ['continuous'] and col in self.processed_data.columns]
        
        with tqdm(total=count) as pbar:
            while valid_samples < count and attempts < max_attempts:
                attempts += 1
                
                # 生成單個樣本
                sample = self._generate_single_sample_with_dependencies()
                
                # 將樣本轉換為DataFrame格式以進行異常檢測
                sample_df = pd.DataFrame([sample])
                
                # 對數值變量進行異常檢測
                is_normal = True
                if numeric_cols and len(numeric_cols) >= 2:
                    try:
                        # 確保樣本包含所有必要的數值列
                        missing_cols = [col for col in numeric_cols if col not in sample_df.columns]
                        if missing_cols:
                            for col in missing_cols:
                                sample_df[col] = self.metadata['stats'][col]['mean']
                        
                        # 獲取數值列的值
                        X = sample_df[numeric_cols].values.reshape(1, -1)
                        # 確保X的數據類型正確
                        X = X.astype(float)
                        
                        # 進行異常檢測
                        pred = isolation_model.predict(X)
                        is_normal = (pred[0] == 1)  # 1表示正常樣本，-1表示異常值
                    except Exception as e:
                        print(f"警告: 異常檢測時出錯: {str(e)}")
                        print(f"跳過此樣本的異常檢測")
                        is_normal = True  # 發生錯誤時，默認為正常
                
                # 如果樣本正常，加入到結果集
                if is_normal:
                    simulated_data = pd.concat([simulated_data, sample_df], ignore_index=True)
                    valid_samples += 1
                    pbar.update(1)
        
        print(f"生成了{valid_samples}個有效樣本，嘗試了{attempts}次")
        
        # 將編碼後的數據轉換回原始格式
        decoded_data = self.processor.decode_data(simulated_data)
        
        # 確保ID列是連續的
        if 'id' in decoded_data.columns:
            decoded_data['id'] = range(1, len(decoded_data) + 1)
        
        # 如果指定了輸出文件，保存結果
        if output_file:
            self.save_results(decoded_data, output_file)
            
            # 創建可視化和報告
            if create_visuals:
                self._generate_visualizations(decoded_data, os.path.dirname(output_file))
        
        return decoded_data
    
    def _train_anomaly_detector(self):
        """訓練隔離森林模型來檢測異常值"""
        # 選擇數值列用於異常檢測
        numeric_cols = [col for col, ctype in self.metadata['column_types'].items() 
                       if ctype in ['continuous'] and col in self.processed_data.columns]
        
        if numeric_cols and len(numeric_cols) >= 2:
            try:
                # 只對連續數值列進行異常檢測
                X = self.processed_data[numeric_cols].values.astype(float)
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X)
                return model
            except Exception as e:
                print(f"警告: 訓練異常檢測模型時出錯: {str(e)}")
                print("將使用默認通過所有樣本的模型")
        
        # 如果沒有足夠的數值列或訓練失敗，返回一個永遠預測為正常的假模型
        class DummyModel:
            def predict(self, X):
                return np.ones(X.shape[0])
        return DummyModel()
    
    def _generate_single_sample_with_dependencies(self) -> dict:
        """生成單個模擬樣本，考慮變量之間的相關性和條件概率"""
        sample = {}
        column_types = self.metadata['column_types']
        
        # 用於處理可能是統計標籤的值
        def is_stat_label(value):
            """檢查一個值是否為統計標籤"""
            stat_labels = ['min', 'max', 'mean', 'median', 'std', 'q1', 'q3', 'count', 'skew', 'kurtosis']
            return isinstance(value, str) and value in stat_labels
        
        def get_stat_value(col, label):
            """獲取指定列的統計值"""
            if col in self.metadata['stats']:
                stats = self.metadata['stats'][col]
                if isinstance(stats, dict) and label in stats:
                    return stats[label]
            # 如果無法獲取統計值，使用一個合理的默認值
            if label == 'min':
                return 0
            elif label == 'max':
                return 100
            elif label in ['mean', 'median']:
                return 50
            return 0
        
        # 1. 先處理分類變量（包括二元和序數）
        categorical_cols = [col for col, ctype in column_types.items() 
                           if ctype in ['categorical', 'binary', 'ordinal']]
        
        # 依據相關性強度的順序處理分類變量
        if categorical_cols:
            # 隨機選擇第一個要填充的變量
            first_col = np.random.choice(categorical_cols)
            choices = [k for k in self.metadata['stats'][first_col].keys() 
                      if not is_stat_label(k)]
            
            if choices:
                probabilities = [self.metadata['stats'][first_col][c] for c in choices]
                if sum(probabilities) > 0:
                    probabilities = [p/sum(probabilities) for p in probabilities]
                    sample[first_col] = np.random.choice(choices, p=probabilities)
            
            # 處理剩餘的分類變量，考慮條件概率
            remaining_cols = [col for col in categorical_cols if col != first_col]
            for col in remaining_cols:
                # 檢查是否有與已生成變量的條件關係
                has_cond_relation = False
                for filled_col in sample.keys():
                    if (filled_col, col) in self.conditional_probs:
                        # 使用條件概率
                        cond_probs = self.conditional_probs[(filled_col, col)]
                        filled_value = sample[filled_col]
                        
                        # 從字典中獲取條件概率
                        if filled_value in cond_probs:
                            row_dict = cond_probs[filled_value]
                            # 過濾掉統計標籤
                            valid_choices = [k for k in row_dict.keys() if not is_stat_label(k)]
                            if valid_choices:
                                valid_probs = [row_dict[k] for k in valid_choices]
                                if sum(valid_probs) > 0:
                                    # 歸一化概率
                                    valid_probs = [p/sum(valid_probs) for p in valid_probs]
                                    sample[col] = np.random.choice(valid_choices, p=valid_probs)
                                    has_cond_relation = True
                                    break
                
                # 如果沒有條件關係，使用邊緣概率
                if not has_cond_relation:
                    valid_choices = [k for k in self.metadata['stats'][col].keys() 
                                   if not is_stat_label(k)]
                    if valid_choices:
                        valid_probs = [self.metadata['stats'][col][c] for c in valid_choices]
                        if sum(valid_probs) > 0:
                            valid_probs = [p/sum(valid_probs) for p in valid_probs]
                            sample[col] = np.random.choice(valid_choices, p=valid_probs)
        
        # 2. 處理連續變量，考慮與分類變量的條件關係
        continuous_cols = [col for col, ctype in column_types.items() if ctype == 'continuous']
        
        for col in continuous_cols:
            # 檢查是否有與分類變量的條件關係
            has_cond_relation = False
            for cat_col in sample.keys():
                if (cat_col, col) in self.conditional_probs:
                    # 使用條件統計生成數值
                    cat_value = sample[cat_col]
                    stats_dict = self.conditional_probs[(cat_col, col)]
                    
                    if cat_value in stats_dict:
                        stats = stats_dict[cat_value]
                        # 確保獲取的是數值而非標籤
                        mean_val = stats['mean'] if not is_stat_label(stats['mean']) else get_stat_value(col, 'mean')
                        std_val = stats['std'] if not is_stat_label(stats['std']) else get_stat_value(col, 'std')
                        min_val = stats['min'] if not is_stat_label(stats['min']) else get_stat_value(col, 'min')
                        max_val = stats['max'] if not is_stat_label(stats['max']) else get_stat_value(col, 'max')
                        
                        # 從條件分布生成
                        value = np.random.normal(mean_val, std_val)
                        # 限制在原始範圍內
                        value = max(min_val, min(max_val, value))
                        sample[col] = value
                        has_cond_relation = True
                        break
            
            # 如果沒有條件關係，使用整體統計
            if not has_cond_relation:
                stats = self.metadata['stats'][col]
                mean_val = stats['mean'] if not is_stat_label(stats['mean']) else get_stat_value(col, 'mean')
                std_val = stats['std'] if not is_stat_label(stats['std']) else get_stat_value(col, 'std')
                min_val = stats['min'] if not is_stat_label(stats['min']) else get_stat_value(col, 'min')
                max_val = stats['max'] if not is_stat_label(stats['max']) else get_stat_value(col, 'max')
                
                value = np.random.normal(mean_val, std_val)
                value = max(min_val, min(max_val, value))
                sample[col] = value
        
        # 3. 添加一些隨機變異
        self._add_variation(sample)
        
        # 4. 最後檢查並替換任何統計標籤
        for col in sample:
            if is_stat_label(sample[col]):
                if column_types[col] == 'continuous':
                    sample[col] = get_stat_value(col, sample[col])
                else:
                    # 對於分類變量，如果發現統計標籤，隨機選擇一個有效值
                    valid_choices = [k for k in self.metadata['stats'][col].keys() 
                                   if not is_stat_label(k)]
                    if valid_choices:
                        sample[col] = np.random.choice(valid_choices)
                    else:
                        # 如果沒有有效選項，使用一個安全的默認值
                        sample[col] = "未知"
        
        return sample
    
    def _add_variation(self, sample: dict) -> None:
        """為樣本添加一些隨機變異以增加多樣性"""
        # 隨機選擇1-3個變量進行輕微調整
        num_vars_to_adjust = np.random.randint(1, 4)
        columns_to_adjust = np.random.choice(
            list(self.processed_data.columns), 
            size=min(num_vars_to_adjust, len(self.processed_data.columns)),
            replace=False
        )
        
        for column in columns_to_adjust:
            ctype = self.metadata['column_types'].get(column)
            
            if ctype == 'continuous':
                # 連續變量添加小幅度隨機波動
                variation = np.random.normal(0, self.metadata['stats'][column]['std'] * 0.1)
                sample[column] += variation
                
            elif ctype == 'ordinal' and np.random.random() < 0.2:
                # 序數變量有20%的可能性偏移一個單位
                current_value = sample[column]
                possible_values = list(self.metadata['stats'][column].keys())
                if len(possible_values) > 1:
                    current_idx = possible_values.index(current_value) if current_value in possible_values else 0
                    new_idx = current_idx + np.random.choice([-1, 1])
                    new_idx = max(0, min(len(possible_values) - 1, new_idx))
                    sample[column] = possible_values[new_idx]
    
    def _generate_visualizations(self, simulated_data: pd.DataFrame, output_dir: str = "results") -> None:
        """生成可視化圖表和統計報告"""
        if self.visualizer is None:
            self.visualizer = SurveyVisualizer(output_dir)
        
        print("正在生成數據可視化和報告...")
        
        # 比較原始數據和模擬數據的分佈
        self.visualizer.compare_distributions(self.original_data, simulated_data)
        
        # 繪製相關性熱力圖
        self.visualizer.correlation_heatmap(self.original_data, "原始")
        self.visualizer.correlation_heatmap(simulated_data, "模擬")
        
        # 創建摘要報告
        self.visualizer.create_summary_report(self.original_data, simulated_data)
        
        print(f"可視化和報告已保存至 {output_dir}")
    
    def save_results(self, simulated_data: pd.DataFrame, output_file: str) -> None:
        """將模擬數據保存到指定的輸出文件 (CSV 或 XLSX)"""
        try:
            _, file_extension = os.path.splitext(output_file)
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if file_extension.lower() == '.csv':
                simulated_data.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"模擬數據已保存至 CSV 文件: {output_file}")
            elif file_extension.lower() == '.xlsx':
                simulated_data.to_excel(output_file, index=False, engine='openpyxl')
                print(f"模擬數據已保存至 XLSX 文件: {output_file}")
            else:
                # 默認保存為 CSV，如果副檔名無法識別或不支援
                default_output_file = os.path.join(output_dir, os.path.basename(output_file).split('.')[0] + '_simulated.csv')
                simulated_data.to_csv(default_output_file, index=False, encoding='utf-8-sig')
                print(f"不支援的輸出文件格式。數據已默認保存至 CSV 文件: {default_output_file}")
                
        except Exception as e:
            print(f"保存結果時出錯: {str(e)}")


def main():
    """主函數，處理命令行參數並執行模擬"""
    parser = argparse.ArgumentParser(description='問卷模擬生成工具')
    parser.add_argument('--input', required=True, help='輸入CSV文件路徑')
    parser.add_argument('--output', required=True, help='輸出CSV文件路徑')
    parser.add_argument('--count', type=int, default=100, help='要生成的模擬樣本數量')
    parser.add_argument('--seed', type=int, help='隨機種子，用於確保結果可重現')
    parser.add_argument('--no-visuals', action='store_true', help='不生成可視化結果')
    args = parser.parse_args()
    
    simulator = SurveySimulator(random_seed=args.seed)
    simulator.load_and_analyze(args.input)
    simulator.generate_samples(args.count, args.output, not args.no_visuals)


if __name__ == "__main__":
    main()
