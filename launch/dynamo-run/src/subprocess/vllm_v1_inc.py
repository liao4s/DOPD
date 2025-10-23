# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `dynamo-run out=vllm` runs this script
# Can also be used standalone: `python3 vllm_inc.py` - lots of optional cmd line params

# Setup checklist:
# - We are in a virtualenv with vllm installed. V1 is compatible with v0.9.0
# Steps:
# git clone https://github.com/vllm-project/vllm.git
# cd vllm && git checkout v0.9.0
# uv pip uninstall ai-dynamo-vllm
# VLLM_USE_PRECOMPILED=1 uv pip install --editable .

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Optional

import uvloop
from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.llm import (
    ModelType,
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import Component, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    model_path: str
    model_name: Optional[str]
    tensor_parallel_size: int
    kv_block_size: int
    context_length: int
    extra_engine_args: str


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, component: Component, dp_rank: int) -> None:
        self.inner = WorkerMetricsPublisher()
        self.inner.create_endpoint(component)
        self.dp_rank = dp_rank

    def record(
        self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]
    ):
        # request_total_slots and kv_total_blocks are properties of model + gpu
        # we should only publish them once, not every metric update
        # they should be part of some runtime metadata tied to MDC or put in etcd ?
        hit_rate = 0
        if scheduler_stats.prefix_cache_stats.queries > 0:
            hit_rate = (
                scheduler_stats.prefix_cache_stats.hits
                / scheduler_stats.prefix_cache_stats.queries
            )

        # TODO Manage DP Ranks in metrics aggregation.
        self.inner.publish(
            request_active_slots=scheduler_stats.num_running_reqs,
            request_total_slots=0,  # TODO - remove from metrics
            kv_active_blocks=0,  # TODO - need to calculate this
            kv_total_blocks=0,  # TODO - remove from metrics
            num_requests_waiting=scheduler_stats.num_waiting_reqs,  # used in current cost function
            gpu_cache_usage_perc=scheduler_stats.gpu_cache_usage,  # used in current cost function
            gpu_prefix_cache_hit_rate=hit_rate,
            data_parallel_rank=self.dp_rank,
        )

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, component: Component) -> None:
        self.component = component

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        return DynamoStatLoggerPublisher(self.component, dp_rank)

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)


class RequestHandler:
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, component, engine, default_sampling_params):
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def generate(self, request):
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        num_output_tokens_so_far = 0
        gen = self.engine_client.generate(prompt, sampling_params, request_id)
        async for res in gen:
            # res is vllm's RequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, cmd_line_args())


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    await register_llm(
        ModelType.Backend,
        generate_endpoint,
        config.model_path,
        config.model_name,
        kv_cache_block_size=config.kv_block_size,
    )

    arg_map = {
        "model": config.model_path,
        "task": "generate",
        "tensor_parallel_size": config.tensor_parallel_size,
        "skip_tokenizer_init": True,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
        "kv_events_config": KVEventsConfig(
            enable_kv_cache_events=True, publisher="zmq"
        ),
    }

    if config.context_length:
        # Usually we want it to default to the max (from tokenizer_config.json)
        arg_map["max_model_len"] = config.context_length

    if config.kv_block_size > 0:
        arg_map["block_size"] = config.kv_block_size

    if config.extra_engine_args != "":
        json_map = {}
        # extra_engine_args is a filename
        try:
            with open(config.extra_engine_args) as f:
                json_map = json.load(f)
        except FileNotFoundError:
            logging.error(f"File {config.extra_engine_args} not found.")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {config.extra_engine_args}: {e}")
        logging.debug(f"Adding extra engine arguments: {json_map}")
        arg_map = {**arg_map, **json_map}  # json_map gets precedence

    logger.info(f"VLLM config: {arg_map}")

    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ[
        "VLLM_WORKER_MULTIPROC_METHOD"
    ] = "spawn"  # Ensure our publisher makes it to the new process

    engine_args = AsyncEngineArgs(**arg_map)
    model_config = engine_args.create_model_config()
    # Load default sampling params from `generation_config.json`
    default_sampling_params = model_config.get_diff_sampling_param()

    # Taken from build_async_engine_client_from_engine_args()
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # Explicitly pass our custom stat logger for metrics
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=[StatLoggerFactory(component)],
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    logger.info("VllmWorker has been initialized")

    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=generate_endpoint.lease_id(), kv_block_size=engine_args.block_size
    )

    _ = ZmqKvEventPublisher(component=component, config=zmq_config)

    handler = RequestHandler(component, engine_client, default_sampling_params)

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="vLLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to disk model or HuggingFace model identifier to load. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Name to serve the model under. Defaults to deriving it from model path.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--kv-block-size", type=int, default=16, help="Size of a KV cache block."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Max model context length. Defaults to models max, usually model_max_length from tokenizer_config.json. Reducing this reduces VRAM requirements.",
    )
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a JSON file containing additional keyword arguments to pass to the vLLM AsyncLLMEngine.",
    )
    args = parser.parse_args()

    config = Config()
    config.model_path = args.model_path
    if args.model_name:
        config.model_name = args.model_name
    else:
        # This becomes an `Option` on the Rust side
        config.model_name = None

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.tensor_parallel_size = args.tensor_parallel_size
    config.kv_block_size = args.kv_block_size
    config.context_length = args.context_length
    config.extra_engine_args = args.extra_engine_args

    return config


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
