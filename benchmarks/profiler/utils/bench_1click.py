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

def get_mlops_1click_cmd(cc, request_input_len = 0, request_output_len = 0, 
                         num_requests = 1024, log_file:str = None, url:str = "localhost:8000", 
                         dataset_path = "/workspace/mlops-main-dynamo/test/ShareGPT_V3_unfiltered_cleaned_split.json",
                         max_sample_request_tokens: int = 16384, path_1click: str = ""):
    if path_1click == "":
        path_1click = "/workspace/mlops-main-dynamo/test/benchmark_client.py"
    mlops1click_CMD = [
        "python3", path_1click,
        "--endpoint", f"http://{url}/v1",
        "--model", f"{MODEL_NAME}",
        "--tokenizer", TOKENIZER_PATH,
        "--parallel", f"{cc}",
        "--num-requests", f"{num_requests}",
        "--sampling-policy", "normal",
        "--dataset", f"{dataset_path}",
        "--max-sample-request-tokens", f"{max_sample_request_tokens}"
    ]
    if request_input_len != 0:
        request_len_list = ["--prompt-len-mean", f"{request_input_len}",
         "--prompt-len-std", "10",
         "--output-len-mean", f"{request_output_len}",
         "--output-len-std", "6"]
        mlops1click_CMD += request_len_list
    if log_file:
        logfile_info_list = ["--log-file", f"{log_file}"]
        mlops1click_CMD += logfile_info_list
    return mlops1click_CMD


def parse_1click_metrics(text: str=None, file_name: str=None):
    if text is None and file_name is not None:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
    metrics = {}
    in_block = False
    # 用来匹配多值指标行，如 "e2e-latency(avg, P50, P90, P99): 5.64, 5.61, 6.22, 8.54"
    multi_val_pattern = re.compile(r'^(.*\))\s*:\s*(.*)$')
    all_metrics = []
    # 按行遍历字符串
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('[BeginMetrics]'):
            in_block = True
            continue
        if line.startswith('[EndMetrics]'):
            all_metrics.append(metrics)
            metrics = {}
            in_block = False
            continue
        if not in_block or not line or ':' not in line:
            continue

        # 先按第一个冒号拆 key/value
        key, raw = line.split(':', 1)
        key = key.strip()
        raw = raw.strip()

        # 检查是否匹配多值指标
        m = multi_val_pattern.match(line)
        if m:
            # m.group(1) 包括 "(avg, ...)" 部分
            k, vals = m.groups()
            k = k.strip()
            metrics[k] = [float(x) for x in vals.split(',')]
        else:
            # 单值：尝试转 float，失败则保留字符串
            try:
                metrics[key] = [float(raw) for raw in raw.split(',')]
            except ValueError:
                metrics[key] = [raw]
    sort_key = "batch-size"
    sort_index = 0
    # 方法二：就地排序
    all_metrics.sort(
        key=lambda m: m.get(sort_key, [float("inf")])[sort_index]
    )
    return all_metrics
def benchmark_1click_prefill(isl, osl, 
    genai_perf_artifact_dir, 
    model_name, port, model_path):
    logger.info(f"Running 1click profiling with isl {isl} and osl {osl}")
    mlops1click_cmd = get_mlops_1click_cmd(
        cc=1, request_input_len=isl, request_output_len=osl, num_requests=num_request,
        log_file=f"{genai_perf_artifact_dir}/mlops_1click.log", url=f"localhost:{port}",
        dataset_path=model_path
    )
    logger.info(f"Running command: {' '.join(mlops1click_cmd)}")
    mlops_process = subprocess.Popen(
        mlops1click_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = mlops_process.communicate()
    if mlops_process.returncode == 0:
        logger.info("1click profiling completed successfully")
        logger.info(stdout)
        metrics = parse_1click_metrics(file_name=f"{genai_perf_artifact_dir}/mlops_1click.log")
        return metrics
    else:
        logger.error(f"1click profiling failed with error code: {mlops_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None

def benchmark_1click_decode(isl, osl, num_request, 
    genai_perf_artifact_dir, 
    model_name, port, model_path):
    logger.info(f"Running 1click profiling with isl {isl} and osl {osl}")
    mlops1click_cmd = get_mlops_1click_cmd(
        cc=1, request_input_len=isl, request_output_len=osl, num_requests=num_request,
        log_file=f"{genai_perf_artifact_dir}/mlops_1click.log", url=f"localhost:{port}",
        dataset_path=model_path
    )
    logger.info(f"Running command: {' '.join(mlops1click_cmd)}")
    mlops_process = subprocess.Popen(
        mlops1click_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = mlops_process.communicate()
    if mlops_process.returncode == 0:
        logger.info("1click profiling completed successfully")
        logger.info(stdout)
        metrics = parse_1click_metrics(file_name=f"{genai_perf_artifact_dir}/mlops_1click.log")
        return metrics
    else:
        logger.error(f"1click profiling failed with error code: {mlops_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None
