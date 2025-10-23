# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, List

import numpy as np
from rich.console import Console
from rich.table import Table
from tensorboardX import SummaryWriter
from utils.prefill_queue import PrefillQueue

from dynamo.llm import KvMetricsAggregator
from dynamo.planner import KubernetesConnector, LocalConnector
from dynamo.planner.defaults import LoadPlannerDefaults
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.planner import KubernetesConnector, LocalConnector
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES, SLAPlannerDefaults
from dynamo.planner.utils.load_predictor import LOAD_PREDICTORS
from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.utils.prometheus import PrometheusAPIClient
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# will not decrease decode worker number within 3 adjustment interval after a new decode worker
# is added. this is to leave time for the new decode worker to populate its kv cache.
NEW_DECODE_WORKER_GRACE_PERIOD = 3

# we do not scale up prefill worker if the prefill queue size is estimated to reduce within
# --prefill-queue-scale-up-threshold within the next NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD
# adjustment intervals following the trend observed in the current adjustment interval.
# this is to account for the time for prefill workers to start.
NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD = 3


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None
    p_load: Optional[float] = None
    d_load: Optional[float] = None

class DOPDPlanner:
    def __init__(self, runtime: DistributedRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self.namespace = args.namespace
        if args.environment == "local":
            self.connector = LocalConnector(args.namespace, runtime)
        elif args.environment == "kubernetes":
            self.connector = KubernetesConnector(args.namespace)
        else:
            raise ValueError(f"Invalid environment: {args.environment}")

        self.prometheus_api_client = PrometheusAPIClient(args.prometheus_endpoint)

        self.num_req_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )
        self.isl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )
        self.osl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )

        self._adjust_lock = asyncio.Lock()
        self.prefill_interpolator = PrefillInterpolator(args.profile_results_dir)
        self.decode_interpolator = DecodeInterpolator(args.profile_results_dir)  
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = f"{self.namespace}_prefill_queue"

        self.last_no_requests_time = None     
        self.prefill_client: Any | None = None
        self.workers_client: Any | None = None
        self.p_endpoints: List[int] = []
        self.d_endpoints: List[int] = []
        self.decode_worker_remaining_grace_period = 0

        if args.log_dir is None:
            args.log_dir = f"logs/{datetime.now().strftime('%m%d_%H%M%S')}"
        self.writer = SummaryWriter(args.log_dir)

        logger.info(f"Components present in namespace: {args.namespace}")

        self.last_adjustment_time = time.time()
        self.init_time = time.time()
        # Set the appropriate logger function for repeated metric logging
        self._repeating_log_func = logger.debug if args.no_operation else logger.info

        self.last_metrics = Metrics()
        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0

    async def set_metric_aggregator(self):
        # TODO: separate KV metrics and prefill metrics
        kv_listener = self.runtime.namespace(self.namespace).component("VllmWorker")
        await kv_listener.create_service()
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)

    async def get_workers_info(self):
        try:
            if self.prefill_client is None:
                self.prefill_client = (
                    await self.runtime.namespace(self.namespace)
                    .component("PrefillWorker")
                    .endpoint("mock")
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling instance_ids
            p_endpoints = self.prefill_client.instance_ids()
        except Exception:
            p_endpoints = []
            self._repeating_log_func(
                "No prefill workers found, operating in aggregated mode"
            )
        try:
            if self.workers_client is None:
                self.workers_client = (
                    await self.runtime.namespace(self.namespace)
                    .component("VllmWorker")
                    .endpoint("generate")
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling instance_ids
            d_endpoints = self.workers_client.instance_ids()
        except Exception as e:
            raise RuntimeError(f"Failed to get decode worker endpoints: {e}")
        return p_endpoints, d_endpoints

    async def reset_adjustment_interval(self):
        self._repeating_log_func(
            f"Reset metrics for new adjustment interval at t={time.time() - self.init_time:.1f}s"
        )

        self.p_endpoints, self.d_endpoints = await self.get_workers_info()

        self._repeating_log_func(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )

        self.metrics_collection_time = []
        self.prefill_queue_load = []
        self.kv_load = []

        self.last_adjustment_time = time.time()

    async def collect_metrics(self):

        self._repeating_log_func(
            f"Collecting metrics at t={time.time() - self.init_time:.1f}s"
        )

        # collect prefill queue load
        try:
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                prefill_queue_size = await prefill_queue.get_queue_size()
                measure_time = time.time() - self.init_time
            self.prefill_queue_load.append(prefill_queue_size)
            self._repeating_log_func(
                f"Collected prefill queue size at t={measure_time:.1f}s: {int(prefill_queue_size)}"
            )
            self.writer.add_scalar(
                "prefill_queue_size", prefill_queue_size, measure_time
            )
        except Exception as e:
            self._repeating_log_func(
                f"Failed to collect prefill queue size metrics: {e}"
            )

        # collect kv load
        total_active_requests: int = 0
        total_queued_requests: int = 0
        metrics = await self.metrics_aggregator.get_metrics()
        try:
            prev_kv_load_len = len(self.kv_load)
            for endpoint in metrics.endpoints:
                kv_load = getattr(endpoint, "gpu_cache_usage_perc", 0.0)
                num_requests_waiting = getattr(endpoint, "num_requests_waiting", 0)
                total_queued_requests += num_requests_waiting
                request_active_slots = getattr(endpoint, "request_active_slots", None)
                if request_active_slots:
                    total_active_requests += request_active_slots
                    if num_requests_waiting > 0:
                        # estimate kv load after waiting requests are scheduled based on current isl/osl
                        # TODO: use actual isl/osl estimation after the request_active_slot bug in disaggg is fixed
                        # Currently, we assume each request uses 0.02 kv cache
                        # kv_load = kv_load * (request_active_slots + num_requests_waiting) / request_active_slots
                        kv_load = kv_load + 0.02 * num_requests_waiting
                self.kv_load.append(kv_load)
            measure_time = time.time() - self.init_time
            self._repeating_log_func(
                f"Collected kv load at t={measure_time:.1f}s: {self.kv_load[prev_kv_load_len:]} (act/pnd req: {total_active_requests}/{total_queued_requests})"
            )
            average_kv_load = np.mean(self.kv_load[prev_kv_load_len:])
            self.writer.add_scalar("average_kv_load", average_kv_load, measure_time)
            self.writer.add_scalar(
                "total_queued_requests", total_queued_requests, measure_time
            )
        except Exception as e:
            self._repeating_log_func(f"Failed to collect kv load metrics: {e}")

        p_endpoints, d_endpoints = await self.get_workers_info()
        self.writer.add_scalar(
            "num_prefill_workers", len(p_endpoints), time.time() - self.init_time
        )
        self.writer.add_scalar(
            "num_decode_workers", len(d_endpoints), time.time() - self.init_time
        )
        curr_gpu_usage = (
            len(p_endpoints) * self.args.prefill_engine_num_gpu
            + len(d_endpoints) * self.args.decode_engine_num_gpu
        )
        self.writer.add_scalar("num_gpu", curr_gpu_usage, time.time() - self.init_time)

        self.metrics_collection_time.append(time.time())

    def calculate_optimal_pd(self, isl: float, osl: float, num_req: int) -> float:
        """
        Calculate the optimal rate of prefill and decode workers based on the input sequence length (isl) and output sequence length (osl).
        This is a placeholder function that should be implemented based on the specific requirements of the planner.
        """
        num_available_kv_blocks = 2040
        block_size = 128
        num_seq_blocks = math.ceil((isl+(osl/2))/block_size) 
        max_decode_concurrency = num_available_kv_blocks // num_seq_blocks
        next_ttft = self.prefill_interpolator.interpolate_ttft(int(isl)) 
        next_tpot = self.decode_interpolator.interpolate_itl(num_req, isl+(osl/2)) * self.d_correction_factor
        pd_rate_optimal = float(num_req / ((osl * next_tpot) / next_ttft))
        return math.ceil(pd_rate_optimal)

    async def observe_metrics(self):
        self._repeating_log_func(
            f"Reset metrics for new adjustment interval at t={time.time() - self.init_time:.1f}s"
        )

        self.p_endpoints, self.d_endpoints = await self.get_workers_info()

        self._repeating_log_func(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )

        self.metrics_collection_time = []
        self.prefill_queue_load = []
        self.kv_load = []

        self.last_adjustment_time = time.time()
        print(f"self.args.adjustment_interval: {self.args.adjustment_interval}")
        self.last_metrics.ttft = self.prometheus_api_client.get_avg_time_to_first_token(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.itl = self.prometheus_api_client.get_avg_inter_token_latency(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.num_req = self.prometheus_api_client.get_avg_request_count(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.request_duration = (
            self.prometheus_api_client.get_avg_request_duration(
                f"{self.args.adjustment_interval}s"
            )
        )
        self.last_metrics.isl = (
            self.prometheus_api_client.get_avg_input_sequence_tokens(
                f"{self.args.adjustment_interval}s"
            )
        )
        self.last_metrics.osl = (
            self.prometheus_api_client.get_avg_output_sequence_tokens(
                f"{self.args.adjustment_interval}s"
            )
        )

        logger.info(
            f"Observed num_req: {self.last_metrics.num_req:.2f} isl: {self.last_metrics.isl:.2f} osl: {self.last_metrics.osl:.2f}"
        )
        logger.info(
            f"Observed ttft: {self.last_metrics.ttft:.3f}s itl: {self.last_metrics.itl:.3f}s"
        )
        self.num_req_predictor.add_data_point(self.last_metrics.num_req)
        self.isl_predictor.add_data_point(self.last_metrics.isl)
        self.osl_predictor.add_data_point(self.last_metrics.osl)
        if self.last_metrics.osl and self.last_metrics.isl and self.last_metrics.request_duration and \
            self.last_metrics.ttft and self.last_metrics.itl and self.last_metrics.num_req:
            return None
        else:
            if self.last_no_requests_time is None:
                logger.warning("No requests observed, marking self.last_no_requests_time as current time")
                return time.time()
            else:
                return self.last_no_requests_time

    async def make_adjustments(self):
        # Note: all adjustments are blocking. Non-blocking adjustment and metric pulling
        # make the optimization problem too complex and should not be needed in most cases.
        logger.info(f"Making adjustments at t={time.time() - self.init_time:.1f}s")
        new_p_endpoints, new_d_endpoints = await self.get_workers_info()
        if len(new_p_endpoints) != len(self.p_endpoints) or len(new_d_endpoints) != len(
            self.d_endpoints
        ):
            logger.info("Decode/prefill workers changed, no adjustments will be made")
            return
        logger.info(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )
        # compute current gpu usage
        curr_gpu_usage = (
            len(self.p_endpoints) * self.args.prefill_engine_num_gpu
            + len(self.d_endpoints) * self.args.decode_engine_num_gpu
        )
        logger.info(f"Current engines use {curr_gpu_usage} GPUs")
        if self.last_no_requests_time and (time.time() - self.last_no_requests_time) > 50:
            logger.info(
                f"No requests observed for {time.time() - self.last_no_requests_time:.1f}s, setting next_num_p and next_num_d to {self.args.min_endpoint}"
            )
            next_num_p = self.args.min_endpoint
            next_num_d = self.args.min_endpoint
            self.last_no_requests_time = None
        # check if decode/prefill workers is still the same
        # note that we only check length as endpoint ids might change
        else:
            try:

                # first correct the prediction correction factor
                # for TTFT, we expect the correction factor to be << 1 due to queuing delay
                expect_ttft = self.prefill_interpolator.interpolate_ttft(
                    self.last_metrics.isl
                ) 
                self.p_correction_factor = self.last_metrics.ttft / expect_ttft
                # for ITL, we expect the correction factor to be close to 1
                expect_itl = self.decode_interpolator.interpolate_itl(
                    concurrency=self.last_metrics.num_req  # type: ignore
                    / len(self.d_endpoints)
                    * self.last_metrics.request_duration  # type: ignore
                    / self.args.adjustment_interval,
                    context_length=self.last_metrics.isl + self.last_metrics.osl / 2,  # type: ignore
                )
                self.d_correction_factor = self.last_metrics.itl / expect_itl
                logger.info(
                    f"self.last_metrics.isl: {self.last_metrics.isl}, self.last_metrics.ttft: {self.last_metrics.ttft:.3f}, expect_ttft: {expect_ttft}"
                )
                logger.info(
                    f"self.last_metrics.itl: {self.last_metrics.itl:.3f}, expect_itl: {expect_itl}"
                )
                logger.info(
                    f"Correction factors: TTFT: {self.p_correction_factor:.3f}, ITL: {self.d_correction_factor:.3f}"
                )
            except Exception as e:
                expect_ttft = 0
                expect_itl = 0
                logger.error(f"Failed to correct prediction factors: {e}")

            try:
                # predict the next load
                next_num_req = self.num_req_predictor.predict_next()
                next_isl = self.isl_predictor.predict_next()
                next_osl = self.osl_predictor.predict_next()
                logger.info(
                    f"Predicted load: num_req={next_num_req:.2f}, isl={next_isl:.2f}, osl={next_osl:.2f}"
                )
            except Exception as e:
                next_num_req = 0
                next_isl = 0
                next_osl = 0
                logger.error(f"Failed to predict load: {e}")
            
            try:
                # compute how many replicas are needed for prefill
                # here we assume the prefill bias is purely due to request queueing
                # and we increase the number of prefill replicas linearly to account for the queueing delay
                
                pred_prefill_load_per_gpu = (
                    next_num_req
                    * next_isl
                    / self.args.adjustment_interval
                    * min(1, self.p_correction_factor)
                )
                num_available_kv_blocks = 2040
                block_size = 128
                num_seq_blocks = math.ceil((next_isl+(next_osl/2))/block_size) 
                max_decode_concurrency = num_available_kv_blocks // num_seq_blocks
                # next_num_p = math.ceil(
                #     pred_prefill_load_per_gpu
                #     / self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
                #     / self.args.prefill_engine_num_gpu
                # )
                next_num_p = self.calculate_optimal_pd(next_isl, next_osl, max_decode_concurrency)
                # 3. compute number of decode replicas needed
                # compute how many replicas are needed for decode
                # 1. apply d_correction_factor to the ITL SLA
                corrected_itl = self.args.itl / self.d_correction_factor
                # 2. reversely find out what is best throughput/gpu that can achieve corrected_itl under the predicted context length
                pred_decode_thpt_per_gpu = (
                    self.decode_interpolator.find_best_throughput_per_gpu(
                        itl=corrected_itl, context_length=next_isl + next_osl / 2
                    )
                )
                # next_num_d = math.ceil(
                #     next_num_req
                #     * next_osl
                #     / self.args.adjustment_interval
                #     / pred_decode_thpt_per_gpu
                #     / self.args.decode_engine_num_gpu
                # )
                next_num_d = math.ceil(next_num_req / max_decode_concurrency)
                # correct num_p and num_d based on the gpu budget
                next_num_p = max(next_num_p, self.args.min_endpoint)
                next_num_d = max(next_num_d, self.args.min_endpoint)
                logger.info(
                    f"Predicted number of engine replicas: prefill={next_num_p}, decode={next_num_d}"
                )

                total_gpu_required = (
                    next_num_p * self.args.prefill_engine_num_gpu
                    + next_num_d * self.args.decode_engine_num_gpu
                )
                while total_gpu_required > self.args.max_gpu_budget:
                    scale = self.args.max_gpu_budget / total_gpu_required
                    next_num_p = max(self.args.min_endpoint, int(next_num_p * scale))
                    next_num_d = max(
                        self.args.min_endpoint,
                        int(
                            (
                                self.args.max_gpu_budget
                                - next_num_p * self.args.prefill_engine_num_gpu
                            )
                            / self.args.decode_engine_num_gpu
                        ),
                    )
                    logger.warning(
                        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({self.args.max_gpu_budget}), scaling down to {next_num_p} prefill and {next_num_d} decode replicas"
                    )
                    total_gpu_required = (
                    next_num_p * self.args.prefill_engine_num_gpu
                    + next_num_d * self.args.decode_engine_num_gpu
                    )
                print(f"[Aking info] next_num_p:{next_num_p}, next_num_d:{next_num_d}")
            except Exception as e:
                next_num_p = 1
                next_num_d = 1
                logger.error(f"Failed to compute number of replicas: {e}, setting to min_endpoint")

        # scale up/down the number of prefill/decode non-blockingly
        async with self._adjust_lock:
            if next_num_p > len(self.p_endpoints):
                for _ in range(next_num_p - len(self.p_endpoints)):
                    success = await self.connector.add_component("PrefillWorker", num_gpu=self.args.prefill_engine_num_gpu)
                    if success:
                        curr_gpu_usage += self.args.prefill_engine_num_gpu
                    else:
                        logger.info("Failed to scale up prefill worker")
            elif next_num_p < len(self.p_endpoints):
                for _ in range(len(self.p_endpoints) - next_num_p):
                    success = await self.connector.remove_component("PrefillWorker")
                    if success:
                        curr_gpu_usage -= self.args.prefill_engine_num_gpu
                    else:
                        logger.info("Failed to scale down prefill worker")

        async with self._adjust_lock:
            if next_num_d > len(self.d_endpoints):
                for _ in range(next_num_d - len(self.d_endpoints)):
                    success = await self.connector.add_component("VllmWorker", num_gpu=self.args.decode_engine_num_gpu)
                    if success:
                        curr_gpu_usage += self.args.decode_engine_num_gpu
                        self.decode_worker_remaining_grace_period = (
                            NEW_DECODE_WORKER_GRACE_PERIOD
                        )
                    else:
                        logger.info("Failed to scale up decode worker")

            elif next_num_d < len(self.d_endpoints):
                for _ in range(len(self.d_endpoints) - next_num_d):
                    success = await self.connector.remove_component("VllmWorker")
                    if success:
                        curr_gpu_usage -= self.args.decode_engine_num_gpu
                    else:
                        logger.info("Failed to scale down decode worker")

        logger.info(f"Engines after adjustment use {curr_gpu_usage} GPUs")

    async def run(self):
        """Main loop for the planner"""

        await self.set_metric_aggregator()

        if self._repeating_log_func == logger.debug:
            logger.info(
                "Running in no-operation mode - detailed metrics will be logged at DEBUG level"
            )

        self.last_no_requests_time = await self.observe_metrics()
        while True:
            block_time = time.time()
            async with self._adjust_lock:
                current_time = time.time()
                if current_time - block_time > 10:
                    # If we are still within the adjustment interval, wait until the interval is over
                    await asyncio.sleep(
                        self.args.adjustment_interval
                    )
                current_time = time.time()
                # Collect metrics at each metric pulling interval
                if (
                    len(self.metrics_collection_time) == 0
                    or current_time - self.metrics_collection_time[-1]
                    >= self.args.metric_pulling_interval
                ):
                    await self.collect_metrics()
                
            # Check if it's time for adjustment
            if (
                current_time - self.last_adjustment_time
                >= self.args.adjustment_interval
            ):
                if not self.args.no_operation:
                    # blockingly make adjustments to avoid overcompensation

                    await self.make_adjustments()
                self.last_no_requests_time = await self.observe_metrics()

            # Sleep to avoid busy waiting
            await asyncio.sleep(self.args.metric_pulling_interval / 10)


# @dynamo_worker()
# TODO: let's make it such that planner still works via CLI invokation
async def start_planner_dopd(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = DOPDPlanner(runtime, args)
    console = Console()
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Endpoint", style="green")

    components = await runtime.etcd_client().kv_get_prefix(args.namespace)
    for component in components:
        try:
            data = json.loads(component["value"].decode("utf-8"))
            if "component" in data:
                name = data["component"]
                endpoint = data["endpoint"]
                table.add_row(name, endpoint)
        except Exception:
            # Some entries may not be valid JSON or might be binary data
            pass

    console.print(table)

    await planner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Common planner arguments
    parser.add_argument(
        "--namespace",
        type=str,
        default=LoadPlannerDefaults.namespace,
        help="Namespace planner will look at",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=LoadPlannerDefaults.environment,
        help="Environment to run the planner in (local, kubernetes)",
    )
    parser.add_argument(
        "--no-operation",
        action="store_true",
        default=LoadPlannerDefaults.no_operation,
        help="Do not make any adjustments, just observe the metrics",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LoadPlannerDefaults.log_dir,
        help="Tensorboard logging directory",
    )
    parser.add_argument(
        "--adjustment-interval",
        type=int,
        default=LoadPlannerDefaults.adjustment_interval,
        help="Interval in seconds between scaling adjustments",
    )
    parser.add_argument(
        "--max-gpu-budget",
        type=int,
        default=LoadPlannerDefaults.max_gpu_budget,
        help="Maximum number of GPUs to use",
    )
    parser.add_argument(
        "--min-endpoint",
        type=int,
        default=LoadPlannerDefaults.min_endpoint,
        help="Minimum number of endpoints to keep for prefill/decode workers",
    )
    parser.add_argument(
        "--metric-pulling-interval",
        type=int,
        default=LoadPlannerDefaults.metric_pulling_interval,
        help="Interval in seconds between metric pulls",
    )
    parser.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=LoadPlannerDefaults.decode_engine_num_gpu,
        help="Number of GPUs per decode engine",
    )
    parser.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=LoadPlannerDefaults.prefill_engine_num_gpu,
        help="Number of GPUs per prefill engine",
    )
    # Load-planner specific arguments
    parser.add_argument(
        "--decode-kv-scale-up-threshold",
        type=float,
        default=LoadPlannerDefaults.decode_kv_scale_up_threshold,
        help="KV cache utilization threshold to scale up decode workers",
    )
    parser.add_argument(
        "--decode-kv-scale-down-threshold",
        type=float,
        default=LoadPlannerDefaults.decode_kv_scale_down_threshold,
        help="KV cache utilization threshold to scale down decode workers",
    )
    parser.add_argument(
        "--prefill-queue-scale-up-threshold",
        type=float,
        default=LoadPlannerDefaults.prefill_queue_scale_up_threshold,
        help="Queue utilization threshold to scale up prefill workers, this threshold is per prefill worker",
    )
    parser.add_argument(
        "--prefill-queue-scale-down-threshold",
        type=float,
        default=LoadPlannerDefaults.prefill_queue_scale_down_threshold,
        help="Queue utilization threshold to scale down prefill workers, this threshold is per prefill worker",
    )
    
    args = parser.parse_args()
    asyncio.run(dynamo_worker()(start_planner)(args))
