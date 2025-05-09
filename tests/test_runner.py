#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
測試運行器
運行所有的單元測試和整合測試，並生成覆蓋率報告
"""

import os
import sys
import unittest
import coverage
import argparse

def run_tests(with_coverage=False):
    """運行所有測試並生成覆蓋率報告（如果啟用）"""
    # 獲取當前目錄（tests目錄）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 獲取主目錄
    parent_dir = os.path.dirname(current_dir)
    
    # 添加主目錄到路徑中
    sys.path.append(parent_dir)
    
    if with_coverage:
        # 配置覆蓋率測量
        cov = coverage.Coverage(
            source=[
                os.path.join(parent_dir, "data_processor.py"),
                os.path.join(parent_dir, "simulator.py"),
                os.path.join(parent_dir, "visualizer.py"),
                os.path.join(parent_dir, "easy_simulator.py")
            ],
            omit=["*/__pycache__/*", "*/tests/*", "*/venv/*"]
        )
        cov.start()
    
    # 發現並運行所有測試
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(current_dir, pattern='test_*.py')
    
    # 運行測試
    unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    if with_coverage:
        # 停止覆蓋率測量
        cov.stop()
        
        # 生成覆蓋率報告
        print("\n生成測試覆蓋率報告...\n")
        
        # 輸出覆蓋率摘要到終端
        cov.report()
        
        # 生成 HTML 覆蓋率報告
        coverage_dir = os.path.join(current_dir, 'coverage')
        os.makedirs(coverage_dir, exist_ok=True)
        cov.html_report(directory=coverage_dir)
        
        print(f"\nHTML 覆蓋率報告已生成：{os.path.abspath(coverage_dir)}/index.html")
        
        # 也可以生成 XML 報告供 CI 工具使用
        cov.xml_report(outfile=os.path.join(coverage_dir, 'coverage.xml'))

def main():
    parser = argparse.ArgumentParser(description='運行測試並生成覆蓋率報告')
    parser.add_argument('--coverage', action='store_true', help='生成測試覆蓋率報告')
    args = parser.parse_args()
    
    run_tests(with_coverage=args.coverage)

if __name__ == "__main__":
    main() 