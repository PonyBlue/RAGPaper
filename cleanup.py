# -*- coding: utf-8 -*-
"""数据清理工具"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import shutil
from pathlib import Path
import os

# 项目目录
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CACHE_DIR = BASE_DIR / ".cache"


def get_directory_size(path: Path) -> float:
    """获取目录大小（MB）"""
    if not path.exists():
        return 0.0

    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)  # 转换为MB


def list_files(directory: Path):
    """列出目录中的文件"""
    if not directory.exists():
        return []

    files = []
    for file in directory.iterdir():
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            files.append(f"  - {file.name} ({size:.2f} KB)")
    return files


def show_status():
    """显示当前存储状态"""
    print("="*70)
    print("📊 当前存储状态")
    print("="*70)

    # PDF文件
    pdf_size = get_directory_size(DATA_DIR)
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    print(f"\n📄 PDF文件目录: data/")
    print(f"   文件数量: {len(pdf_files)}")
    print(f"   占用空间: {pdf_size:.2f} MB")
    if pdf_files:
        print(f"   文件列表:")
        for file in pdf_files:
            size = file.stat().st_size / (1024 * 1024)
            print(f"     - {file.name} ({size:.2f} MB)")

    # 向量数据库
    db_size = get_directory_size(CHROMA_DB_DIR)
    print(f"\n💾 向量数据库: chroma_db/")
    print(f"   占用空间: {db_size:.2f} MB")

    # 缓存
    cache_size = get_directory_size(CACHE_DIR)
    if cache_size > 0:
        print(f"\n🗂️  缓存目录: .cache/")
        print(f"   占用空间: {cache_size:.2f} MB")

    total_size = pdf_size + db_size + cache_size
    print(f"\n📦 总占用空间: {total_size:.2f} MB")
    print("="*70)


def clean_pdfs():
    """清理PDF文件"""
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.pdf")):
        print("✓ PDF目录已经是空的")
        return

    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    confirm = input(f"⚠️  将删除 {pdf_count} 个PDF文件，确认吗? (y/N): ")

    if confirm.lower() == 'y':
        for pdf in DATA_DIR.glob("*.pdf"):
            pdf.unlink()
            print(f"  ✓ 已删除: {pdf.name}")
        print(f"✓ 已清理 {pdf_count} 个PDF文件")
    else:
        print("❌ 已取消")


def clean_vectordb():
    """清理向量数据库"""
    if not CHROMA_DB_DIR.exists():
        print("✓ 向量数据库已经是空的")
        return

    db_size = get_directory_size(CHROMA_DB_DIR)
    confirm = input(f"⚠️  将删除向量数据库 ({db_size:.2f} MB)，确认吗? (y/N): ")

    if confirm.lower() == 'y':
        shutil.rmtree(CHROMA_DB_DIR)
        CHROMA_DB_DIR.mkdir()
        print("✓ 已清理向量数据库")
    else:
        print("❌ 已取消")


def clean_cache():
    """清理缓存"""
    if not CACHE_DIR.exists():
        print("✓ 缓存目录已经是空的")
        return

    cache_size = get_directory_size(CACHE_DIR)
    if cache_size == 0:
        print("✓ 缓存已经是空的")
        return

    confirm = input(f"⚠️  将删除缓存 ({cache_size:.2f} MB)，确认吗? (y/N): ")

    if confirm.lower() == 'y':
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        print("✓ 已清理缓存")
    else:
        print("❌ 已取消")


def clean_all():
    """清理所有数据"""
    print("\n⚠️  警告：这将删除所有数据！")
    show_status()

    confirm = input("\n确认清理所有数据吗? (yes/N): ")

    if confirm.lower() == 'yes':
        print("\n开始清理...")

        # 清理PDF
        if DATA_DIR.exists():
            for pdf in DATA_DIR.glob("*.pdf"):
                pdf.unlink()
            print("✓ 已清理PDF文件")

        # 清理向量库
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
            CHROMA_DB_DIR.mkdir()
            print("✓ 已清理向量数据库")

        # 清理缓存
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()
            print("✓ 已清理缓存")

        print("\n✓✓✓ 所有数据已清理完成！")
    else:
        print("❌ 已取消")


def clean_specific_pdf():
    """清理特定PDF"""
    if not DATA_DIR.exists():
        print("❌ data目录不存在")
        return

    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ 没有PDF文件")
        return

    print("\n📄 可用的PDF文件:")
    for i, pdf in enumerate(pdfs, 1):
        size = pdf.stat().st_size / (1024 * 1024)
        print(f"  {i}. {pdf.name} ({size:.2f} MB)")

    try:
        choice = input("\n输入要删除的文件编号 (多个用逗号分隔，例如 1,3): ")
        indices = [int(x.strip()) for x in choice.split(',')]

        for idx in indices:
            if 1 <= idx <= len(pdfs):
                pdf = pdfs[idx - 1]
                pdf.unlink()
                print(f"✓ 已删除: {pdf.name}")
            else:
                print(f"❌ 无效的编号: {idx}")

        print("\n⚠️  注意：删除PDF后，需要重置向量数据库才能完全清除相关数据")
        reset_db = input("是否同时重置向量数据库? (y/N): ")
        if reset_db.lower() == 'y':
            clean_vectordb()

    except ValueError:
        print("❌ 输入格式错误")


def main():
    """主菜单"""
    while True:
        print("\n" + "="*70)
        print("🗑️  RAG项目数据清理工具")
        print("="*70)
        print("\n选项:")
        print("  1. 查看存储状态")
        print("  2. 清理PDF文件（保留向量库）")
        print("  3. 清理向量数据库（保留PDF）")
        print("  4. 清理缓存")
        print("  5. 清理特定PDF文件")
        print("  6. 清理所有数据")
        print("  0. 退出")

        choice = input("\n请选择 (0-6): ").strip()

        if choice == '0':
            print("\n👋 再见！")
            break
        elif choice == '1':
            show_status()
        elif choice == '2':
            clean_pdfs()
        elif choice == '3':
            clean_vectordb()
        elif choice == '4':
            clean_cache()
        elif choice == '5':
            clean_specific_pdf()
        elif choice == '6':
            clean_all()
        else:
            print("❌ 无效的选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消，再见！")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
