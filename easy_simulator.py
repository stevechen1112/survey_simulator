#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
簡化版問卷模擬器
只需指定輸入文件名，自動處理輸入輸出路徑
"""

import os
import sys
import argparse
from simulator import SurveySimulator

def main():
    parser = argparse.ArgumentParser(description='簡化版問卷模擬生成工具')
    parser.add_argument('input_file', help='輸入CSV文件名稱（放在data目錄下）')
    parser.add_argument('--count', type=int, default=100, help='要生成的模擬樣本數量（默認100）')
    parser.add_argument('--seed', type=int, help='隨機種子，用於確保結果可重現')
    parser.add_argument('--no-visuals', action='store_true', help='不生成視覺化結果')
    args = parser.parse_args()
    
    # 自動生成輸入輸出路徑
    input_path = os.path.join('data', args.input_file)
    
    # 創建輸出目錄
    filename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = os.path.join('results', filename + '_simulated')
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置輸出文件路徑
    output_path = os.path.join(output_dir, 'simulated_data.csv')
    
    print(f"輸入文件: {input_path}")
    print(f"輸出目錄: {output_dir}")
    print(f"輸出文件: {output_path}")
    print(f"生成樣本數: {args.count}")
    
    # 檢查輸入文件是否存在
    if not os.path.exists(input_path):
        print(f"錯誤: 找不到輸入文件 {input_path}")
        print(f"請確保您的文件放在data目錄中")
        sys.exit(1)
    
    # 運行模擬器
    try:
        simulator = SurveySimulator(random_seed=args.seed)
        simulator.load_and_analyze(input_path)
        simulator.generate_samples(args.count, output_path, not args.no_visuals)
        
        print(f"\n模擬完成！")
        print(f"模擬數據已保存至: {output_path}")
        if not args.no_visuals:
            print(f"視覺化報告保存在: {output_dir}")
    except Exception as e:
        print(f"錯誤: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 