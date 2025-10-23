import os
import shutil
import argparse
from typing import Iterable, Union

def copy_files_with_all_keywords(
    src_dir: str,
    keywords: list[str],
    dest_dir: str
) -> None:
    """
    从 src_dir 中递归搜索文件名同时包含 keywords 中所有关键词的文件，
    并复制到 dest_dir（不存在则自动创建）。

    :param src_dir: 要搜索的源目录路径
    :param keywords: 单个关键词或关键词列表
    :param dest_dir: 复制到的目标目录路径
    """
    # 创建目标目录（包括多级），已存在则忽略
    os.makedirs(dest_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            # 只有当 filename 同时包含所有关键词时，才复制
            if all(kw in filename for kw in keywords):
                src_file = os.path.join(root, filename)
                dest_file = os.path.join(dest_dir, filename)
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"已复制：{src_file} -> {dest_file}")
                except Exception as e:
                    print(f"复制失败：{src_file}，错误：{e}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--src-dir",
        type=str,
        required=True,
        help="源目录（包含所有待筛选文件）"
    )

    parser.add_argument(
        "-k", "--keywords",
        nargs='+',
        required=True,
        help="文件名中必须包含的关键词，可同时包含多个关键词"
    )

    args = parser.parse_args()
    if isinstance(args.keywords, str):
        args.keywords = [args.keywords]
    
    dest_dir = os.path.join(args.src_dir, '-'.join(args.keywords))
    copy_files_with_all_keywords(args.src_dir, args.keywords, dest_dir)

if __name__ == "__main__":
    main()