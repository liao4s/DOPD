import matplotlib.pyplot as plt
import re
import os
import argparse
import shutil

# extract the 1click metrics lines from str or file
def get_1click_metrics_lines(text: str=None, file_name: str=None):
    global MODEL_NAME
    global REQUEST_PROMPT
    REQUEST_PROMPT = None
    if text is None and file_name is not None:
        with open(file_name, 'r') as f:
            text = f.read()
    lines_text = ""
    # 按行遍历字符串
    in_block = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('[BeginMetrics]'):
            lines_text += line
            in_block = True
            continue
        if line.startswith('[EndMetrics]'):
            lines_text += "\n" + line + "\n" + "\n"
            in_block = False
            continue
        
        if not in_block or not line or ':' not in line:
            continue
        if line.startswith('model'):
            MODEL_NAME = line.split(':')[-1].strip()
        if line.startswith('sequence-length'):
            REQUEST_PROMPT = line.split(':')[-1].strip()
        lines_text += "\n" + line
        
    return lines_text

# extract 1click metrics lines and write them into 1 file
def merge_1click_metrics_from_dir(filedir):
    print(os.listdir(filedir))
    all_file_texts = ""
    TEXT_EXTS = {'.log', '.json'}
    for filepath in os.listdir(filedir):
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in TEXT_EXTS:
            continue
        if "1click_output" in filepath:
            with open(os.path.join(filedir, filepath), 'r', encoding="utf-8") as f:
                all_file_texts += f.read()
    all_metrics_lines_text = get_1click_metrics_lines(all_file_texts)
    filename = filedir.split('/')[-1].split("_")[0]+ "_" + MODEL_NAME + "_" + REQUEST_PROMPT + ".txt"
    with open(os.path.join(filedir, filename), "w") as f:
        f.write(all_metrics_lines_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filedir", type=str, required=True,
                        help="Path to 1click metrics log director")
    args = parser.parse_args()
    merge_1click_metrics_from_dir(args.filedir)
        

if __name__ == "__main__":
    main()