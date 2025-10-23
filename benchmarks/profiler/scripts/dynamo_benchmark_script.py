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
from parse_metrics import parse_1click_metrics
from merge_metrics_files import merge_1click_metrics_from_dir
from pathlib import Path

MODEL_NAME = "DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
TOKENIZER_PATH = "/models/hub/models--neuralmagic--DeepSeek-R1-Distill-Llama-70B-FP8-dynamic/snapshots/fb3637a1165cec3832958bd72ebbe04021601489"
MODEL_PATH = "/models/hub/models--neuralmagic--DeepSeek-R1-Distill-Llama-70B-FP8-dynamic/snapshots/fb3637a1165cec3832958bd72ebbe04021601489"
MAX_MODEL_LEN = 16384
TP_DECODE = 2
TP_PREFILL = 1
NUM_DECODE = 1
GPU_UTIL_DECODE = 0.96
GPU_UTIL_PREFILL = 0.98
CONDITIONAL_DISAGG = False
MAX_LOCAL_PREFILL_LENGTH = 511
MAX_PREFILL_QUEUE_SIZE = 8
PRIORITY_QUEUE_TYPE = "short"
TYPE_BENCHMARK = "1click"
NUM_PREFILL_WORKERS = [2]
NUM_DECODE_WORKERS = [1]
REQUEST_LENS = [(0, 0)]
# API_URL = "10.200.1.2:5426"
# SERVER_TYPE = "vllm"
# USE_ROUTER = [False]
# USE_GPU_MONITOR = False
PORT = 5426
API_URL = f"10.200.1.13:{PORT}"
SERVER_TYPE = "dynamo"
USE_ROUTER = [False]
USE_GPU_MONITOR = True
SERVER_CONFIG_STR = "tp4-no-prefix-caching"
# DATASET = "/workspace/dynamo-eval/llama-70b-completions_online_dataset_requests_blocks_0626.json"
# DATASET = "/workspace/dynamo-eval/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET = "/workspace/dynamo-eval/AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv"
NUM_CONCURRENCY = [80, 88, 96, 104, 8, 16, 24, 32, 40, 48, 56, 64, 72, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,  208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392]   
NUM_REQUESTS = 1024
PATH_1CLICK = "/workspace/dynamo-eval/llm_benchmark/mlops-main-dynamo/test/benchmark_client.py"
MAX_SAMPLE_REQUEST_TOKENS = 16384
rand_seed = 42
random.seed(rand_seed)
# --- dynamo server ---
def create_dynamo_yamlfile(yaml_dir = "configs", num_decode_workers = 1, num_prefill_workers = 1, update_num_prefill: bool = True):
    yamlfile_content = f"""
Common:
  model: {MODEL_PATH}
  block-size: 128
  max-model-len: {MAX_MODEL_LEN}
  kv-transfer-config: '{{"kv_connector":"DynamoNixlConnector"}}'

Frontend:
  served_model_name: {MODEL_NAME}
  endpoint_chat: dynamo.Processor.chat/completions
  endpoint_completion: dynamo.Processor.completions
  port: {PORT}

Processor:
  router: round-robin
  common-configs: [model, block-size]

VllmWorker:
  remote-prefill: true
  conditional-disagg: {CONDITIONAL_DISAGG}
  max-local-prefill-length: {MAX_LOCAL_PREFILL_LENGTH}
  max-prefill-queue-size: {MAX_PREFILL_QUEUE_SIZE}
  tensor-parallel-size: {TP_DECODE}
  gpu-memory-utilization: {GPU_UTIL_DECODE}
  ServiceArgs:
    workers: {num_decode_workers}
    resources:
      gpu: {TP_DECODE}
  common-configs: [model, block-size, max-model-len, kv-transfer-config]

PrefillWorker:
  tensor-parallel-size: {TP_PREFILL}
  gpu-memory-utilization: {GPU_UTIL_PREFILL}
  priority-queue: {PRIORITY_QUEUE_TYPE}
  ServiceArgs:
    workers: {num_prefill_workers}
    resources:
      gpu: {TP_PREFILL}
  common-configs: [model, block-size, max-model-len, kv-transfer-config]
        """
    yamlfile_path = os.path.join(yaml_dir, f"{MODEL_NAME}_disagg_{num_prefill_workers}x{TP_PREFILL}p{num_decode_workers}x{TP_DECODE}d.yaml")
    with open(yamlfile_path, "w") as f:
        f.write(yamlfile_content)
    return yamlfile_path

# --- dynamo server ---
def create_dynamo_router_yamlfile(yaml_dir = "configs", num_decode_workers = 1, num_prefill_workers = 1, update_num_prefill: bool = True):
    yamlfile_content = f"""
Common:
  model: {MODEL_PATH}
  block-size: 128
  max-model-len: {MAX_MODEL_LEN}
  router: kv
  kv-transfer-config: '{{"kv_connector":"DynamoNixlConnector"}}'

Frontend:
  served_model_name: {MODEL_NAME}
  endpoint_chat: dynamo.Processor.chat/completions
  endpoint_completion: dynamo.Processor.completions
  port: {PORT}

Processor:
  common-configs: [model, block-size, max-model-len, router]

Router:
  min-workers: 1
  common-configs: [model]

VllmWorker:
  remote-prefill: true
  conditional-disagg: {CONDITIONAL_DISAGG}
  max-local-prefill-length: {MAX_LOCAL_PREFILL_LENGTH}
  max-prefill-queue-size: {MAX_PREFILL_QUEUE_SIZE}
  tensor-parallel-size: {TP_DECODE}
  swap-space: 32
  worker_stage: 'decode'
  enable-prefix-caching: true
  gpu-memory-utilization: {GPU_UTIL_DECODE}
  ServiceArgs:
    workers: {num_decode_workers}
    resources:
      gpu: {TP_DECODE}
  common-configs: [model, block-size, max-model-len, kv-transfer-config]

PrefillWorker:
  tensor-parallel-size: {TP_PREFILL}
  gpu-memory-utilization: {GPU_UTIL_PREFILL}
  worker_stage: 'prefill'
  ServiceArgs:
    workers: {num_prefill_workers}
    resources:
      gpu: {TP_PREFILL}
  common-configs: [model, block-size, max-model-len, kv-transfer-config]
        """
    yamlfile_path = os.path.join(yaml_dir, f"{MODEL_NAME}_disagg_{num_prefill_workers}x{TP_PREFILL}p{num_decode_workers}x{TP_DECODE}d_router.yaml")
    with open(yamlfile_path, "w") as f:
        f.write(yamlfile_content)
    return yamlfile_path

def kill_process_by_keyword(keyword: str):
    try:
        # 获取所有匹配的进程信息
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split('\n')

        for line in lines:
            if keyword in line and "dynamo_benchmark_script.py" not in line:
                parts = line.split()
                pid = int(parts[1])  # 第二列是 PID
                print(f"正在强制杀死进程 {pid}: {line}")
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"错误: {e}")
        
def start_dynamo_server(yamlfile_path, log_file, wait_time=60, use_router=False):
    """
    启动 dynamo server，返回 Popen 对象
    """
    # 先删除dynamo之前的残余进程
    kill_process_by_keyword('dynamo')
    kill_process_by_keyword('llmctl')
    kill_process_by_keyword('http')
    kill_process_by_keyword('python3 -c from multiprocessing')

    graphs_str = f"graphs.disagg{'_router' if use_router else ''}:Frontend"
    cmd = [
        "dynamo",
        "serve",
        graphs_str, 
        "-f", 
        yamlfile_path
    ]
    # stdout/stderr 重定向到日志文件
    log_fd = open(log_file, "a")
    try:
        dynamo_proc = subprocess.Popen(cmd, stdout=log_fd, stderr=log_fd)
        print(f"Waiting {wait_time} seconds for dynamo server to start...")
        time.sleep(wait_time)  # 等待一段时间让服务启动
        if dynamo_proc.poll() is not None:
            print("Dynamo server failed to start.")
            log_fd.close()
            raise RuntimeError("Dynamo server failed to start.")
        print(f"Started {cmd}")
    except KeyboardInterrupt:
        print("User interrupted; stopping Dynamo server…")
        stop_popen_subproc((dynamo_proc, log_fd))
    except Exception as e:
        print(f"start_dynamo_server raise exception: {e}")
        stop_popen_subproc((dynamo_proc, log_fd))

    return dynamo_proc, log_fd

def stop_popen_subproc(proc_item, timeout=5) -> str:
    """
    优雅地停止 subprocess.Popen，超时后强制 kill
    """
    proc, log_fd = proc_item
    print(f"Stopping {proc} subprocess")
    if proc and isinstance(proc, subprocess.Popen):
        proc.terminate()
    else:
        print(f"proc is {proc}.\nproc_type is {type(proc)}")
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  graceful shutdown timeout, killing ...")
        proc.kill()   # 再发 SIGKILL
        proc.wait()
    log_fd.close() if log_fd is not None else True
    print(f"Stopped {proc}")

def start_gpu_monitor(log_file, scripts_dir: str = "./", interval: float = 0.1):
    """
    start GPU monitor thread, collect GPU metrics
    """
    try:
        monitor_proc = subprocess.Popen(
            ["python3", os.path.join(scripts_dir, "gpu_monitor.py"), 
            "--log-file", f"{log_file}", "--interval", f"{interval}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if monitor_proc.poll() is not None:
            print("GPU monitor failed to start.")
            raise RuntimeError("GPU monitor failed to start.")
        print(f"Started GPU monitor")
    except KeyboardInterrupt:
        print("User interrupted; stopping Dynamo server…")
        stop_popen_subproc((monitor_proc, None))
    except Exception as e:
        print(f"start_gpu_monitor raise exception: {e}")
        stop_popen_subproc((monitor_proc, None))
    
    return monitor_proc, None

# --- benchmark ---
# TODO: Support more benchmark. Only support mlops1click benchmark for now
def start_benchmark_from_cmd(CMD, log_file:str = None) -> str:
    """
    启动基准测试，返回 Popen 对象
    """
    
    print(">>> beginning benchmark...")
    proc = subprocess.Popen(
        CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(">>> benchmark completed")
    all_lines = []
    for line in proc.stdout:
        print(line, end="")
        all_lines.append(line)
    
    out_str = "".join(all_lines)
    return_code = proc.wait()
    print(f">>> benchmark completed, return code = {return_code}")
    if log_file:
        with open(log_file, "w") as f:
            f.write(out_str)
    return out_str

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

def get_genai_perf_cmd(cc = 8, request_input_len = 1000, request_output_len = 100, num_requests = 1024, url="localhost:8000"):
    GENAI_PERF_CMD = [
        "genai-perf", "profile",
        "-m", f"{MODEL_NAME}",
        "--tokenizer", f"{MODEL_PATH}",
        "--service-kind", "openai",
        "--endpoint-type", "completions",
        "--endpoint", "v1/completions",
        "--url", f"{url}",
        "--streaming",
        "--concurrency", f"{cc}",
        "--num-dataset-entries", "256",
        "--warmup-request-count", "128",
        "--request-count", f"{num_requests}",
        "--synthetic-input-tokens-mean", f"{request_input_len}",
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", f"{request_output_len}",
        "--output-tokens-stddev", "0",
        "--extra-inputs", f"min_tokens:{request_output_len}",
        "--extra-inputs", f"max_tokens:{request_output_len}",
        "--extra-inputs", "ignore_eos:true",
        "--random-seed", "0",
        "--", "--max-threads", f"{cc}"
    ]
    return GENAI_PERF_CMD

def determine_ttft(metrics: dict, threshold=1.5, position=0):
    return metrics["ttft(avg, P50, P90, P99)"][position] >= threshold

def pressure_test():
    NUM_PREFILL_WORKERS = [1]
    NUM_DECODE_WORKERS = [1]
    REQUEST_LENS = [(3000,300)]
    server_type = "dynamo"
    dataset = DATASET
    num_ccs = [152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 300, 8, 32, 64, 128, 136, 144]   # num_ccs_lists must have 2 dim
    
    for j, num_decode_workers in enumerate(NUM_DECODE_WORKERS):
        for i, num_prefill_workers in enumerate(NUM_PREFILL_WORKERS):
            if (num_decode_workers*TP_DECODE) + (num_prefill_workers*TP_PREFILL) > 8:
                break
            server_config_str = f"{num_prefill_workers}x{TP_PREFILL}p{num_decode_workers}x{TP_DECODE}d"
            LOG_DIR = f"benchmark_logs/{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M')}/{server_type}_f'{server_config_str}'_request-len-{request_input_len}"
            os.makedirs(LOG_DIR, exist_ok=True)
            yamlfile_path = create_dynamo_yamlfile(num_decode_workers=num_decode_workers, num_prefill_workers=num_prefill_workers)
            start_dynamo_server(
                yamlfile_path=yamlfile_path,
                log_file=os.path.join(LOG_DIR, f"dynamo-server_{server_config_str}_request-len-{request_input_len}.log"),
                wait_time=90
            )
            while True:
                for request_input_len, request_output_len in REQUEST_LENS:
                    for cc in num_ccs:
                        benchmark_cmd = get_mlops_1click_cmd(cc, request_input_len, request_output_len) if TYPE_BENCHMARK == "1click" else get_genai_perf_cmd(cc, request_input_len, request_output_len)
                        start_benchmark_from_cmd(benchmark_cmd)

def main(args):
    scripts_dir = Path(__file__).resolve().parent
    log_dir = args.log_dir if args.log_dir else scripts_dir.parent.parent
    os.chdir(args.exec_path)
    print('pwd: ', os.getcwd())
    num_prefill_workers_list = NUM_PREFILL_WORKERS
    num_decode_workers_list = NUM_DECODE_WORKERS
    request_lens_list = REQUEST_LENS
    # server_type = "vllm"
    # use_routers = [False]
    # use_gpu_monitor = False
    # url = "10.200.1.10:6543"
    server_type = SERVER_TYPE
    use_routers_list = USE_ROUTER
    use_gpu_monitor = USE_GPU_MONITOR
    url = API_URL
    dataset = DATASET
    num_ccs_list = NUM_CONCURRENCY
    num_requests = NUM_REQUESTS
    path_1click = PATH_1CLICK
    max_sample_request_tokens = MAX_SAMPLE_REQUEST_TOKENS
    LOG_DIR_TEST = os.path.join(log_dir, f"benchmark_logs/{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M')}")
    for use_router in use_routers_list:
        for j, num_decode_workers in enumerate(num_decode_workers_list):
            for i, num_prefill_workers in enumerate(num_prefill_workers_list):
                for request_input_len, request_output_len in request_lens_list:
                    if (num_decode_workers*TP_DECODE) + (num_prefill_workers*TP_PREFILL) > 8:
                        break
                    for cc in num_ccs_list:
                        try:
                            # server_config_str examples: ['tp1', '6x1p1x2d']
                            server_config_str = f"{num_prefill_workers}x{TP_PREFILL}p{num_decode_workers}x{TP_DECODE}d" if "dynamo" in server_type else SERVER_CONFIG_STR
                                
                            LOG_DIR = os.path.join(LOG_DIR_TEST, f"{server_type}-{server_config_str}_request-len-{request_input_len}")
                            if use_router and server_type == "dynamo": 
                                LOG_DIR += "_router"
                            os.makedirs(LOG_DIR, exist_ok=True)
                            if server_type == "dynamo":
                                yamlfile_path = create_dynamo_router_yamlfile(num_decode_workers=num_decode_workers, num_prefill_workers=num_prefill_workers) \
                                    if use_router else create_dynamo_yamlfile(num_decode_workers=num_decode_workers, num_prefill_workers=num_prefill_workers)
                                dynamo_proc = start_dynamo_server(
                                    yamlfile_path=yamlfile_path,
                                    log_file=os.path.join(LOG_DIR, f"dynamo-server_{server_config_str}_request-len-{request_input_len}_cc-{cc}.log"),
                                    wait_time=90, use_router=use_router
                                )
                            result_text = None
                            if use_gpu_monitor:
                                gpu_monitor_proc = start_gpu_monitor(
                                    log_file=os.path.join(LOG_DIR, f"gpu_monitor_{server_type}_{server_config_str}_request-len-{request_input_len}_cc-{cc}.json"),
                                    scripts_dir=scripts_dir)
                            benchmark_cmd = get_mlops_1click_cmd(cc, request_input_len, request_output_len, url=url, num_requests=num_requests, 
                                                                    dataset_path=dataset, path_1click=path_1click, max_sample_request_tokens=max_sample_request_tokens) \
                                if TYPE_BENCHMARK == "1click" else get_genai_perf_cmd(cc, request_input_len, request_output_len)
                            result_text = start_benchmark_from_cmd(benchmark_cmd, os.path.join(LOG_DIR, f"{TYPE_BENCHMARK}_output_{server_type}_{server_config_str}_request-len-{request_input_len}_cc-{cc}.log"))
                            all_metrics = parse_1click_metrics(result_text)
                            if use_gpu_monitor:
                                stop_popen_subproc(gpu_monitor_proc)
                            if determine_ttft(all_metrics[0], 5):
                                break
                        except KeyboardInterrupt:
                            print("User interrupted; stopping Dynamo and Monitor server…")
                            # stop_popen_subproc(gpu_monitor_proc)
                            with open(os.path.join(LOG_DIR, f"{TYPE_BENCHMARK}_output_{server_type}_{server_config_str}_request-len-{request_input_len}_cc-{cc}.log"), "w") as f:
                                    f.write(result_text)
                            if server_type == "dynamo": stop_popen_subproc(dynamo_proc)
                        except Exception as e:
                            print(f"!!! raise exception: {e}")
                            merge_1click_metrics_from_dir(LOG_DIR)
                            print("waiting 5s")
                            time.sleep(5)
                            if use_gpu_monitor:
                                stop_popen_subproc(gpu_monitor_proc)
                            if server_type == "dynamo": stop_popen_subproc(dynamo_proc)
                        finally:
                            merge_1click_metrics_from_dir(LOG_DIR)
                            print("waiting 5s")
                            time.sleep(5)
                            if use_gpu_monitor:
                                stop_popen_subproc(gpu_monitor_proc)
                            if server_type == "dynamo": stop_popen_subproc(dynamo_proc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-type", type=str, default="dynamo", help="support server: ['vllm', 'dynamo']")
    parser.add_argument("--start-gpu-monitor", type=bool, default=False, help="whether turn on the gpu monitor")
    parser.add_argument("--benchmark-type", type=str, default="1click", help="support benchmark: ['1click', 'genai_perf']")
    # parser.add_argument("--request-lens", nargs='+', default=None, required=True, help="input and output lens, example: (1000, 100) (3000,300). If request-lens=None, sampling request in order from datasets")
    # parser.add_argument("--dataset", type=str, default=None, required=True, help="the datasets path or use default dataset")
    parser.add_argument("--test-type", type=str, default="benchmark", help="support test: ['benchmark', 'pressure']")
    parser.add_argument("--num-requests", type=int, default=1024, help="support test: ['benchmark', 'pressure']")
    parser.add_argument("--exec-path", type=str, default="/workspace/examples/llm", help="dynamo must execute in a specified directory")
    parser.add_argument("--log-dir", type=str, default=None, help="benchmark_logs dir path")
    parser.add_argument("--use-router", nargs='+', type=bool, default=True, help="whether dynamo use router")
    parser.add_argument("--num-prefillworkers", nargs='+', type=int, default=1, help="the number of prefillworkers")
    parser.add_argument("--num-decodeworkers", nargs='+', type=int, default=1, help="the number of decodeworkers")
    parser.add_argument("--tp-prefillworkers", nargs='+', type=int, default=1, help="the number of prefillworkers' tensor-parallel-size")
    parser.add_argument("--tp-decodeworkers", nargs='+', type=int, default=1, help="the number of prefillworkers' tensor-parallel-size")
    parser.add_argument("--concurrency", nargs='+', type=int, default=8, help="the number of benchmark concurrency")
    parser.add_argument("--url", type=str, default="localhost:8000", help="the number of benchmark concurrency")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")