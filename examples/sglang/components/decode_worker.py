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

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import msgspec
import sglang as sgl
import uvloop
from sglang.srt.server_args import ServerArgs
from utils.sgl_utils import parse_sglang_args_inc

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class DecodeRequestHandler:
    def __init__(self, engine: sgl.Engine):
        self.engine = engine
        logging.info("Decode request handler initialized")

    async def generate(self, request: str):
        req = msgspec.json.decode(request, type=dict)

        results = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"]
            if req["request"]["batch_token_ids"] is None
            else req["request"]["batch_token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=req["bootstrap_host"],
            bootstrap_port=req["bootstrap_port"],
            bootstrap_room=req["bootstrap_room"],
        )

        async for result in results:
            yield result


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    server_args = parse_sglang_args_inc(sys.argv[1:])
    await init(runtime, server_args)


async def init(runtime: DistributedRuntime, server_args: ServerArgs):
    """Initialize decode worker"""

    engine = sgl.Engine(server_args=server_args)

    handler = DecodeRequestHandler(engine)

    component = runtime.namespace("dynamo").component("decode")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
