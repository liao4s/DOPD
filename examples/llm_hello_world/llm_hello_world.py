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

import asyncio
import logging
import random

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dynamo.sdk import (
    DYNAMO_IMAGE,
    AbstractService,
    abstract_endpoint,
    api,
    depends,
    endpoint,
    service,
)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    text: str


"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/v1/chat/completions)
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Router    │  Routes requests to appropriate worker
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Worker    │  Generates text using LLM
└─────────────┘
"""


class WorkerInterface(AbstractService):
    """Interface for LLM workers."""

    @abstract_endpoint  # enforces that the service implements the method, but also that it is properly decorated
    async def generate(self, request: ChatRequest):
        pass


class RouterInterface(AbstractService):
    """Interface for request routers."""

    @abstract_endpoint
    async def generate(self, request: ChatRequest):
        pass


@service(
    dynamo={"namespace": "llm-hello-world"},
    image=DYNAMO_IMAGE,
)
class VllmWorker(WorkerInterface):
    @endpoint()
    async def generate(self, request: ChatRequest):
        # Convert to Spongebob case (randomly capitalize letters)
        for token in request.text.split():
            spongebob_token = "".join(
                c.upper() if random.random() < 0.5 else c.lower() for c in token
            )
            yield spongebob_token


@service(
    dynamo={"namespace": "llm-hello-world"},
    image=DYNAMO_IMAGE,
)
class TRTLLMWorker(WorkerInterface):
    @endpoint()
    async def generate(self, request: ChatRequest):
        # Convert to SHOUTING case
        for token in request.text.split():
            yield token.upper()


@service(
    dynamo={"namespace": "llm-hello-world"},
    image=DYNAMO_IMAGE,
)
class SlowRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Will be overridden by link()

    @endpoint()
    async def generate(self, request: ChatRequest):
        print("Routing slow")
        async for response in self.worker.generate(request.model_dump_json()):
            await asyncio.sleep(1)  # Simulate slow routing with a 1-second delay
            yield response


@service(
    dynamo={"namespace": "llm-hello-world"},
    image=DYNAMO_IMAGE,
)
class FastRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Will be overridden by link()

    @endpoint()
    async def generate(self, request: ChatRequest):
        print("Routing fast")
        async for response in self.worker.generate(request.model_dump_json()):
            await asyncio.sleep(0.1)  # Simulate fast routing with a 0.1-second delay
            yield response


app = FastAPI()


@service(
    dynamo={"namespace": "llm-hello-world"},
    image=DYNAMO_IMAGE,
    app=app,
)
class Frontend:
    router = depends(RouterInterface)  # Will be overridden by link()

    @api()
    async def generate(self, request: ChatRequest):
        print(f"Received request: {request}")

        async def content_generator():
            async for response in self.router.generate(request.model_dump_json()):
                print(f"Received response: {response}")
                # Format as SSE
                yield f"data: {response}\n\n"

        return StreamingResponse(
            content_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


# Mix and match pipelines (Tests)
# Frontend.link(SlowRouter).link(TRTLLMWorker) # type: ignore[attr-defined]
# slow_pipeline = Frontend.link(SlowRouter).link(VllmWorker) # type: ignore[attr-defined]
Frontend.link(FastRouter).link(VllmWorker)  # type: ignore[attr-defined]

"""
Example usage:

fast_pipeline = Frontend.link(FastRouter).link(TRTLLMWorker)
# slow_pipeline = Frontend.link(SlowRouter).link(VllmWorker)
# mixed_pipeline = Frontend.link(FastRouter).link(VllmWorker)


# Basic setup with VLLM worker and slow router
The interface-based design allows for:
1. Easy swapping of implementations (VLLM vs TRT-LLM)
2. Different routing strategies (slow vs fast)
3. Type safety through interface contracts
"""
