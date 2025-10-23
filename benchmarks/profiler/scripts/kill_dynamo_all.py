import argparse
import random
import signal
import threading
import subprocess
import time
import json
import pynvml
import time
import os
import re
from datetime import datetime
from pathlib import Path

def kill_process_by_keyword(keyword: str):
    try:
        # 获取所有匹配的进程信息
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split('\n')

        # 仅文件名
        filename = os.path.basename(__file__)
        for line in lines:
            if keyword in line and str(filename) not in line:
                parts = line.split()
                pid = int(parts[1])  # 第二列是 PID
                print(f"正在强制杀死进程 {pid}: {line}")
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    
    # 先删除dynamo之前的残余进程
    kill_process_by_keyword('dynamo')
    kill_process_by_keyword('llmctl')
    kill_process_by_keyword('http')
    kill_process_by_keyword('python3 -c from multiprocessing')