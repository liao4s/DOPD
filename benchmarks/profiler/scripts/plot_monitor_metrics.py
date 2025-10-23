import json
from datetime import datetime
import matplotlib.pyplot as plt

def plot_monitor_metrics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # ===== 2. 解析时间序列 =====
    # JSON 中 timestamp 格式为 ISO8601（末尾带 Z 表示 UTC），去掉 Z 然后转换
    times = [
        datetime.fromisoformat(entry["timestamp"].replace("Z", ""))
        for entry in metrics_data
    ]

    num_gpus = len(metrics_data[0]["gpus"])
    gpu_util = {i: [] for i in range(num_gpus)}
    gpu_mem_util = {i: [] for i in range(num_gpus)}

    # 填充数据
    for entry in metrics_data:
        for gpu_info in entry["gpus"]:
            idx = gpu_info["index"]
            gpu_util[idx].append(gpu_info["util_gpu_pct"])
            gpu_mem_util[idx].append(gpu_info["util_mem_pct"])

    fig, axes = plt.subplots(num_gpus, 1, sharex=True, figsize=(12, 4 * num_gpus))
    if num_gpus == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(range(len(times)), gpu_util[i], label="GPU Util (%)")
        ax.bar(range(len(times)), gpu_mem_util[i], label="Mem Util (%)")
        ax.set_title(f"GPU {i}")
        ax.set_ylabel("utilization (%)")
        ax.legend(loc="upper left")
        ax.grid(True)

    plt.xlabel("timestamp")
    plt.tight_layout()
    plt.savefig(f"{json_path[:-5]}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    json_path = "/workspace/dynamo-eval/benchmark_logs/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_20250705-1029/dynamo-2x1p1x2d_request-len-3000/gpu_monitor_dynamo_2x1p1x2d_request-len-3000_cc-56.json"
    plot_monitor_metrics(json_path)