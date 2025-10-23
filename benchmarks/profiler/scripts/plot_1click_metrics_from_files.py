from typing import Union
import matplotlib.pyplot as plt
from matplotlib import font_manager
import re
import os

DEVICE_NAME = "H100"
MODEL_NAME = "Llama-3.3-70B-Instruct-FP8-dynamic"
def parse_metrics_from_string(text: str=None, file_name: str=None):
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


def collect_filepaths(dir_or_filepaths: Union[str, list[str]]) -> list[str]:
    """
    如果输入是目录，则收集该目录下所有 .txt 文件（或你需要的后缀）；
    如果已经是文件列表，则直接返回。
    """
    if isinstance(dir_or_filepaths, str) and os.path.isdir(dir_or_filepaths):
        # 遍历目录，收集所有 .txt 日志文件
        return [
            os.path.join(dir_or_filepaths, fn)
            for fn in os.listdir(dir_or_filepaths)
            if fn.endswith(".txt")
        ]
    elif isinstance(dir_or_filepaths, list):
        return dir_or_filepaths
    else:
        raise ValueError(f"传入参数既不是目录，也不是字符串列表：{dir_or_filepaths}")

def get_server_type(file_path: str):
    return "vllm" if "vllm" in file_path else "dynamo"

def get_server_config(file_path: str):
    server_type = get_server_type(file_path)
    if server_type == "dynamo":
        match_pd = re.search(r'(\d+)x(\d+)p*(\d+)x(\d+)d', file_path)
        num_prefill_worker = list(map(int, match_pd.groups()))[0]
        tp_prefill = list(map(int, match_pd.groups()))[1]
        num_decode_worker = list(map(int, match_pd.groups()))[2]
        tp_decode = list(map(int, match_pd.groups()))[3]
        return [num_prefill_worker, tp_prefill, num_decode_worker, tp_decode]
    elif server_type == "vllm":
        match_vllm = re.search(r'tp(\d+)', file_path)
        return [int(match_vllm.group(1))]

def get_num_gpu(file_path: str, num_gpu_type: str = "all_gpu"):
    '''
    num_gpu_type:
        all_gpu : num_all_gpu
        prefill_gpu: num_prefill_gpu
        decode_gpu: num_decode_gpu
    '''
    # print(file_path)
    server_type = "vllm" if "vllm" in file_path else "dynamo"
    server_config = get_server_config(file_path)
    num_prefill_gpu = 0 if server_type == "vllm"  else server_config[0] * server_config[1]
    num_decode_gpu = 0 if server_type == "vllm" else server_config[2] * server_config[3]
    # print(num_prefill_gpu, num_decode_gpu)
    num_all_gpu = sum(server_config) if server_type == "vllm"  else num_prefill_gpu + num_decode_gpu
    result_list = [num_all_gpu, num_prefill_gpu, num_decode_gpu]
    if num_gpu_type == "decode_gpu":
        print(f"{num_gpu_type} gpus is {result_list[2]}")
        return result_list[2]
    elif num_gpu_type == "prefill_gpu":
        print(f"{num_gpu_type} gpus is {result_list[1]}")
        return result_list[1]
    else: 
        print(f"{num_gpu_type} gpus is {result_list[0]}")
        return result_list[0]

def get_y_unit(y_key):
    y_unit = ""
    if "throughput" in y_key or "tps" in y_key:
        y_unit = "tokens/s"
    elif "ttft" in y_key or "tpot" in y_key:
        y_unit = "ms"
    return y_unit

def plot_metrics_from_files(dir_or_filepaths, x_key, y_keys, 
                            x_position=0, y_positions=[0], 
                            need_axhlines=False, y_axhlines=None, 
                            y_scale_strs=[None, None], title_str=None, x_scale_str:str=None,
                            y_labels=None, fig_path:str = None):
    """
    Plot metrics from given files. Supports one or two y-keys with dual y-axes.

    Args:
        dir_or_filepaths (Union [str, list[str]]): Directory or paths to log files containing metrics.
        x_key (str): The key for x-axis values.
        y_keys (list[str]): One or two keys for y-axis values.
        x_position (int): Index in the x_key array to use.
        y_positions (list[int]): Indices in the y_keys arrays to use.
        need_axhlines (bool): Whether to draw horizontal SLO lines.
        y_axhlines (tuple[float, float] or float): SLO values for y and y2 (if provided).
    """
    supplement_texts = ["", ""]
    for i, y_key in enumerate(y_keys):
        if '|' in y_key:
            y_keys[i] = y_key.split('|')[1]
            supplement_texts[i] = y_key.split('|')[0]
    y_key1 = y_keys[0]
    y_pos1 = y_positions[0]

    has_second = len(y_keys) > 1
    y_key2 = y_keys[1] if has_second else None
    y_pos2 = y_positions[1] if has_second and len(y_positions) > 1 else None
    font_manager.fontManager.addfont('/workspace/dynamo-eval/llm_benchmark/scripts/TimesNewRoman.ttf')

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx() if has_second else None


    handles1, labels1 = [], []
    handles2, labels2 = [], []

    filepaths = collect_filepaths(dir_or_filepaths)    

    # 根据filepaths个数生成多种颜色
    num_filepaths = len(filepaths)
    cmaps = ['tab20', 'tab20b', 'tab20c']
    colors = []
    for cmap_name in cmaps:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap.colors)
    colors = colors[:num_filepaths]

    for i, file_path in enumerate(filepaths):
        # Read and parse
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        all_metrics = parse_metrics_from_string(text)

        file_path_base = os.path.basename(file_path)
        server_config = get_server_config(file_path_base)
        server_type = get_server_type(file_path_base)
        if server_type == "vllm":
            server_config_str = f"vllm-tp{server_config[0]}" 
        elif server_type == "dynamo": 
            server_config_str = f"pd-disagg--{server_config[0]}x{server_config[1]}p{server_config[2]}x{server_config[3]}d"
        # Extract x
        x = [m[x_key][x_position] for m in all_metrics]

        if x_scale_str:
            x_scale = get_num_gpu(file_path_base, x_scale_str) if "gpu" in x_scale_str else float(x_scale_str)
            x = [_ / x_scale for _ in x]
        # Plot first y
        y1 = [m[y_key1][y_pos1] for m in all_metrics]
        if y_scale_strs[0] is not None:
            y_scale = get_num_gpu(file_path_base, y_scale_strs[0]) if "gpu" in y_scale_strs[0] else float(y_scale_strs[0])
            y1 = [_ / y_scale for _ in y1]
        
        label1 = y_labels[i][0] if y_labels else f"{server_config_str} - {supplement_texts[0]} {y_key1}" 
        h1, = ax1.plot(x, y1, color=colors[i % len(colors)], marker='o', label=label1)
        handles1.append(h1)
        labels1.append(label1)

        # Plot second y if present
        if has_second:
            y2 = [m[y_key2][y_pos2] for m in all_metrics]
            # Optionally scale or convert units
            y2 = [v for v in y2]
            if y_scale_strs[1] is not None:
                y_scale = get_num_gpu(file_path_base, y_scale_strs[1]) if "gpu" in y_scale_strs[1] else float(y_scale_strs[1])
                y2 = [_ / y_scale for _ in y2]
            label2 = y_labels[i][1] if y_labels else f"{server_config_str} - {supplement_texts[1]} {y_key2}"
            h2, = ax2.plot(x, y2, color=colors[(i) % len(colors)], marker='x', linestyle='--', label=label2)
            handles2.append(h2)
            labels2.append(label2)

    # Draw horizontal lines if requested
    if need_axhlines:
        if isinstance(y_axhlines, (list, tuple)) and has_second:
            ax1.axhline(y=y_axhlines[0], color='red', marker='o', label='baseline-output-throughput')
            ax2.axhline(y=y_axhlines[1], color='red', marker='x', linestyle='--', label='baseline-ttft')
        elif isinstance(y_axhlines, (int, float)):
            ax1.axhline(y=y_axhlines, color='red', marker='o', linestyle='--', label='SLO')

    # Set labels and title
    ax1.set_xlabel(x_key+f"{' per gpu' if x_scale_str and 'gpu' in x_scale_str else ''}")
    ax1.set_ylabel(f"{supplement_texts[0]} {y_key1} ({get_y_unit(y_key)}{'/gpu' if 'gpu' in y_scale_strs[0] else ''})")
    if has_second:
        ax2.set_ylabel(f"{supplement_texts[1]} {y_key2} ({get_y_unit(y_key2)}{'/gpu' if 'gpu' in y_scale_strs[1] else ''})")
    request_len = file_path_base[:-4].split('_')[-1].replace(', ', '/')
    device_name = DEVICE_NAME
    model_name = os.path.basename(filepaths[0]).split('_')[1]
    # plt.title(title_str if title_str else f"{device_name} {model_name} {supplement_texts[0]}-{y_key1} vs {x_key}")

    # Combine legends
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax1.legend(all_handles, all_labels, loc='best')

    ax1.grid(True)
    # Save and show
    save_dir = os.path.dirname(filepaths[0])
    base = os.path.basename(filepaths[0])[9:-4]
    img_path = fig_path if fig_path else os.path.join(save_dir, f"{base}_{supplement_texts[0]+y_key1}_vs_{x_key}.png")
    fig.savefig(img_path, dpi=300)
    plt.show()

def main():
    file_paths = ["/workspace/dynamo-eval/benchmark_logs/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_20250706-160_noqueue_burstgpt_1024/dynamo-2x1p1x2d_request-len-0/dynamo-2x1p1x2d_DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_1024.409756097561, 245.3151219512195.txt",
                  "/workspace/dynamo-eval/benchmark_logs/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_20250708-0311/dynamo-2x1p1x2d_request-len-0/dynamo-2x1p1x2d_DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_1024.409756097561, 245.3151219512195.txt"
                  ]

    file_dir = "/workspace/dynamo-eval/benchmark_logs/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_20250702-0459_nopbatch/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
    plot_metrics_from_files(file_paths, "batch-size", ["output|throughput", "ttft(avg, P50, P90, P99)"], 
        y_positions=[1,0], x_scale_str="1", y_scale_strs=["1", "0.001"], 
        need_axhlines=False, y_axhlines=[275, 1520],
        title_str="H100 DeepSeek-R1-Distill-Llama-70B-FP8-dynamic PD-disagg output-throughput vs batchsize per gpu ",
        y_labels=[["dynamo throughput", "dynamo ttft"], 
                  ["DOPD throughput", "DOPD ttft"], 
                  ["2p(tp1) 1d(tp2) throughput", "2p(tp1) 1d(tp2) ttft"], 
                  ["4x1p1x2d output throughput", "4x1p1x2d ttft"], 
                  ["5x1p1x2d output throughput", "5x1p1x2d ttft"], 
                  ["6x1p1x2d output throughput", "6x1p1x2d ttft"]],
        fig_path=os.path.join(os.path.dirname(file_paths[0]),"DeepSeek-R1-Distill-Llama-70B-FP8-dynamic_Azuredataset-1024_outputthroughput_vs_batch-size.png"))

if __name__ == "__main__":
    main()