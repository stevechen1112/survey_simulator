#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class SurveyVisualizer:
    """問卷數據可視化工具"""
    
    def __init__(self, output_dir: str = "results"):
        """初始化可視化器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 設置中文顯示
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 設置風格
        sns.set_style("whitegrid")
    
    def compare_distributions(self, original_data: pd.DataFrame, simulated_data: pd.DataFrame, 
                             columns: Optional[List[str]] = None) -> None:
        """比較原始數據和模擬數據的分佈"""
        if columns is None:
            # 自動選擇適合可視化的列
            columns = []
            for col in original_data.columns:
                # 跳過ID列和超過20個唯一值的列
                if col.lower() == 'id' or original_data[col].nunique() > 20:
                    continue
                columns.append(col)
        
        for column in columns:
            plt.figure(figsize=(12, 6))
            
            # 處理分類數據
            if original_data[column].dtype == 'object' or original_data[column].nunique() <= 10:
                # 計算比例
                orig_counts = original_data[column].value_counts(normalize=True)
                sim_counts = simulated_data[column].value_counts(normalize=True)
                
                # 合併索引以確保兩個數據集有相同的類別
                all_categories = pd.Index(set(orig_counts.index) | set(sim_counts.index))
                orig_counts = orig_counts.reindex(all_categories, fill_value=0)
                sim_counts = sim_counts.reindex(all_categories, fill_value=0)
                
                # 創建數據框用於繪圖
                comparison_df = pd.DataFrame({
                    '原始數據': orig_counts,
                    '模擬數據': sim_counts
                })
                
                # 繪製柱狀圖
                comparison_df.plot(kind='bar', figsize=(12, 6))
                plt.title(f'{column} 分佈比較')
                plt.ylabel('比例')
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            # 處理數值數據
            else:
                # 繪製密度圖
                sns.kdeplot(original_data[column], label='原始數據', fill=True, alpha=0.5)
                sns.kdeplot(simulated_data[column], label='模擬數據', fill=True, alpha=0.5)
                plt.title(f'{column} 分佈比較')
                plt.xlabel(column)
                plt.ylabel('密度')
                plt.legend()
                plt.tight_layout()
            
            # 保存圖像
            plt.savefig(os.path.join(self.output_dir, f'comparison_{column}.png'), dpi=300)
            plt.close()
    
    def correlation_heatmap(self, data: pd.DataFrame, title: str) -> None:
        """繪製相關性熱力圖"""
        # 選擇數值型列
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = data[numeric_cols].corr()
            
            # 繪製熱力圖
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, 
                       fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            
            plt.title(f'{title}數據相關性')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'correlation_{title}.png'), dpi=300)
            plt.close()
    
    def create_summary_report(self, original_data: pd.DataFrame, simulated_data: pd.DataFrame) -> None:
        """創建摘要報告，比較原始數據和模擬數據的主要統計指標"""
        report = []
        
        # 遍歷所有列
        for column in original_data.columns:
            if column.lower() == 'id':
                continue
                
            # 數值型數據
            if pd.api.types.is_numeric_dtype(original_data[column]):
                try:
                    orig_stats = original_data[column].describe()
                    sim_stats = simulated_data[column].describe()
                    
                    report.append(f"## {column} 統計比較\n")
                    report.append("| 統計量 | 原始數據 | 模擬數據 |")
                    report.append("| ------ | -------- | -------- |")
                    
                    # 確保使用describe()返回的統計量名稱，避免使用可能不存在的自定義名稱
                    for stat in sim_stats.index:
                        if stat in orig_stats.index:
                            report.append(f"| {stat} | {orig_stats[stat]:.2f} | {sim_stats[stat]:.2f} |")
                    
                    report.append("\n")
                except Exception as e:
                    report.append(f"## {column} 統計比較 (處理時出錯: {str(e)})\n")
                    report.append("\n")
            
            # 分類數據
            else:
                try:
                    orig_counts = original_data[column].value_counts(normalize=True)
                    sim_counts = simulated_data[column].value_counts(normalize=True)
                    
                    # 合併索引以確保兩個數據集有相同的類別
                    all_categories = pd.Index(set(orig_counts.index) | set(sim_counts.index))
                    orig_counts = orig_counts.reindex(all_categories, fill_value=0)
                    sim_counts = sim_counts.reindex(all_categories, fill_value=0)
                    
                    report.append(f"## {column} 分佈比較\n")
                    report.append("| 類別 | 原始數據比例 | 模擬數據比例 |")
                    report.append("| ---- | ------------ | ------------ |")
                    
                    for category in all_categories:
                        orig_pct = orig_counts.get(category, 0)
                        sim_pct = sim_counts.get(category, 0)
                        report.append(f"| {category} | {orig_pct:.2%} | {sim_pct:.2%} |")
                    
                    report.append("\n")
                except Exception as e:
                    report.append(f"## {column} 分佈比較 (處理時出錯: {str(e)})\n")
                    report.append("\n")
        
        # 寫入報告文件
        with open(os.path.join(self.output_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
            f.write("# 數據模擬結果比較報告\n\n")
            f.write('\n'.join(report))
        
        print(f"摘要報告已生成: {os.path.join(self.output_dir, 'summary_report.md')}") 