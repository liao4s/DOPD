import threading
import time
import json
import pynvml
import os
import signal
import argparse
from datetime import datetime

# 全局存储 metrics 和输出路径
metrics = []
output_file = None

def write_metrics_and_exit(signum, frame):
    """在收到 SIGTERM/SIGINT 时调用，写文件并退出"""
    global metrics, output_file
    print(f"\n>>> Received signal {signum}, writing metrics to {output_file} ...", flush=True)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f">>> Successfully wrote {len(metrics)} records.", flush=True)
    except Exception as e:
        print(f"!!! Failed to write metrics: {e}", flush=True)
    finally:
        # 直接退出进程
        os._exit(0)

def gpu_monitor(interval: float):
    """定时采集 GPU 指标，追加到全局 metrics 列表"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    try:
        while True:
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = {"timestamp": timestamp, "gpus": []}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                entry["gpus"].append({
                    "index": i,
                    "mem_used_MB": mem.used // 1024**2,
                    "mem_total_MB": mem.total // 1024**2,
                    "util_gpu_pct": util.gpu,
                    "util_mem_pct": util.memory
                })
            metrics.append(entry)
            print(f"[{timestamp}] " +
                  " | ".join(
                      f"GPU{g['index']}: {g['mem_used_MB']}/{g['mem_total_MB']}MB "
                      f"(GPU {g['util_gpu_pct']}% / MEM {g['util_mem_pct']}%)"
                      for g in entry["gpus"]
                  ), flush=True)
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()

def main():
    global output_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True,
                        help="Path to output JSON file")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="采样间隔（秒）")
    args = parser.parse_args()

    output_file = args.log_file

    # 注册信号处理器
    signal.signal(signal.SIGTERM, write_metrics_and_exit)
    signal.signal(signal.SIGINT, write_metrics_and_exit)

    print(f">>> Starting GPU monitor, interval = {args.interval}s, logging to `{output_file}`")
    gpu_monitor(args.interval)
    # 这里永远不会自然返回，除非收到 SIGTERM/SIGINT

if __name__ == "__main__":
    main()
