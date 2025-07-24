#!/usr/bin/env python3
"""
测试运行脚本
提供不同类型的测试运行选项
"""

import sys
import subprocess
import argparse
from pathlib import Path


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
    
    def run_unit_tests(self, verbose=True):
        """运行单元测试"""
        print("🧪 运行单元测试...")
        cmd = [
            "pytest", 
            str(self.tests_dir),
            "-m", "unit or not integration",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def run_integration_tests(self, verbose=True):
        """运行集成测试"""
        print("🔗 运行集成测试...")
        cmd = [
            "pytest",
            str(self.tests_dir),
            "-m", "integration",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def run_all_tests(self, verbose=True, coverage=True):
        """运行所有测试"""
        print("🚀 运行所有测试...")
        cmd = ["pytest", str(self.tests_dir)]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        else:
            cmd.append("-q")
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-exclude=tests/*"
            ])
        
        return subprocess.run(cmd)
    
    def run_specific_test(self, test_path, verbose=True):
        """运行特定测试"""
        print(f"🎯 运行特定测试: {test_path}")
        cmd = [
            "pytest",
            str(self.tests_dir / test_path),
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def run_by_keyword(self, keyword, verbose=True):
        """根据关键词运行测试"""
        print(f"🔍 运行包含关键词 '{keyword}' 的测试...")
        cmd = [
            "pytest",
            str(self.tests_dir),
            "-k", keyword,
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def run_failed_tests(self, verbose=True):
        """重新运行失败的测试"""
        print("🔄 重新运行失败的测试...")
        cmd = [
            "pytest",
            str(self.tests_dir),
            "--lf",  # last failed
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def run_performance_tests(self, verbose=True):
        """运行性能测试"""
        print("⚡ 运行性能测试...")
        cmd = [
            "pytest",
            str(self.tests_dir),
            "-m", "slow",
            "--durations=10",
            "-v" if verbose else "-q"
        ]
        return subprocess.run(cmd)
    
    def check_test_coverage(self):
        """检查测试覆盖率"""
        print("📊 生成测试覆盖率报告...")
        cmd = [
            "pytest",
            str(self.tests_dir),
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term",
            "--cov-exclude=tests/*",
            "--cov-exclude=main.py",
            "-q"
        ]
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✅ 覆盖率报告已生成到 htmlcov/ 目录")
            print("💡 打开 htmlcov/index.html 查看详细报告")
        
        return result
    
    def validate_code_quality(self):
        """验证代码质量"""
        print("🔍 检查代码质量...")
        
        # 检查代码格式（如果安装了black）
        try:
            print("📝 检查代码格式...")
            subprocess.run(["black", "--check", "."], check=True)
            print("✅ 代码格式检查通过")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ 未安装black或代码格式需要调整")
        
        # 检查代码风格（如果安装了flake8）
        try:
            print("📏 检查代码风格...")
            subprocess.run(["flake8", ".", "--max-line-length=100"], check=True)
            print("✅ 代码风格检查通过")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ 未安装flake8或代码风格需要调整")
    
    def install_test_dependencies(self):
        """安装测试依赖"""
        print("📦 安装测试依赖...")
        cmd = [
            "pip", "install", "-r", 
            str(self.tests_dir / "requirements.txt")
        ]
        return subprocess.run(cmd)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="心理咨询数据生成系统测试运行器")
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "all", "coverage", "failed", 
            "performance", "quality", "install-deps"
        ],
        help="要运行的测试类型"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        help="按关键词过滤测试"
    )
    
    parser.add_argument(
        "-t", "--test",
        type=str,
        help="运行特定测试文件"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="禁用覆盖率报告"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # 检查测试目录是否存在
    if not runner.tests_dir.exists():
        print(f"❌ 测试目录不存在: {runner.tests_dir}")
        sys.exit(1)
    
    # 执行相应的命令
    result = None
    
    try:
        if args.command == "unit":
            result = runner.run_unit_tests(args.verbose)
        
        elif args.command == "integration":
            result = runner.run_integration_tests(args.verbose)
        
        elif args.command == "all":
            if args.keyword:
                result = runner.run_by_keyword(args.keyword, args.verbose)
            elif args.test:
                result = runner.run_specific_test(args.test, args.verbose)
            else:
                result = runner.run_all_tests(args.verbose, not args.no_coverage)
        
        elif args.command == "coverage":
            result = runner.check_test_coverage()
        
        elif args.command == "failed":
            result = runner.run_failed_tests(args.verbose)
        
        elif args.command == "performance":
            result = runner.run_performance_tests(args.verbose)
        
        elif args.command == "quality":
            runner.validate_code_quality()
            result = runner.run_all_tests(args.verbose, True)
        
        elif args.command == "install-deps":
            result = runner.install_test_dependencies()
        
        # 输出结果
        if result and result.returncode == 0:
            print("\n✅ 测试执行成功!")
        elif result and result.returncode != 0:
            print(f"\n❌ 测试执行失败，退出代码: {result.returncode}")
            sys.exit(result.returncode)
    
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 测试执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 使用示例
    if len(sys.argv) == 1:
        print("🧪 心理咨询数据生成系统测试运行器")
        print("\n📖 使用示例:")
        print("  python run_tests.py all              # 运行所有测试")
        print("  python run_tests.py unit -v          # 运行单元测试（详细模式）")
        print("  python run_tests.py integration      # 运行集成测试")
        print("  python run_tests.py coverage         # 生成覆盖率报告")
        print("  python run_tests.py failed           # 重新运行失败的测试")
        print("  python run_tests.py performance      # 运行性能测试")
        print("  python run_tests.py quality          # 代码质量检查+测试")
        print("  python run_tests.py all -k student   # 运行包含'student'的测试")
        print("  python run_tests.py all -t test_models.py  # 运行特定测试文件")
        print("  python run_tests.py install-deps     # 安装测试依赖")
        print("\n🔧 选项:")
        print("  -v, --verbose    显示详细输出")
        print("  -k KEYWORD       按关键词过滤测试")
        print("  -t TEST          运行特定测试文件")
        print("  --no-coverage    禁用覆盖率报告")
        print("\n💡 提示:")
        print("  - 首次运行请先执行: python run_tests.py install-deps")
        print("  - 覆盖率报告会生成到 htmlcov/ 目录")
        print("  - 使用 -v 选项查看详细的测试输出")
        sys.exit(0)
    
    main()
