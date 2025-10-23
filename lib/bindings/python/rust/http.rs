// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use pyo3::{exceptions::PyException, prelude::*};

use crate::{engine::*, to_pyerr, CancellationToken};

pub use dynamo_llm::http::service::{error as http_error, service_v2};

pub use dynamo_runtime::{
    error,
    pipeline::{async_trait, AsyncEngine, Data, ManyOut, SingleIn},
    protocols::annotated::Annotated,
    Error, Result,
};

#[pyclass]
pub struct HttpService {
    inner: service_v2::HttpService,
}

#[pymethods]
impl HttpService {
    #[new]
    #[pyo3(signature = (port=None))]
    pub fn new(port: Option<u16>) -> PyResult<Self> {
        let builder = service_v2::HttpService::builder().port(port.unwrap_or(8080));
        let inner = builder.build().map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    pub fn add_completions_model(&self, model: String, engine: HttpAsyncEngine) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_completions_model(&model, engine)
            .map_err(to_pyerr)
    }

    pub fn add_chat_completions_model(
        &self,
        model: String,
        engine: HttpAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_chat_completions_model(&model, engine)
            .map_err(to_pyerr)
    }

    pub fn remove_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn remove_chat_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_chat_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn list_chat_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_chat_completions_models())
    }

    pub fn list_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_completions_models())
    }

    fn run<'p>(&self, py: Python<'p>, token: CancellationToken) -> PyResult<Bound<'p, PyAny>> {
        let service = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token.inner).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

/// Python Exception for HTTP errors
#[pyclass(extends=PyException)]
pub struct HttpError {
    code: u16,
    message: String,
}

#[pymethods]
impl HttpError {
    #[new]
    pub fn new(code: u16, message: String) -> Self {
        HttpError { code, message }
    }

    #[getter]
    fn code(&self) -> u16 {
        self.code
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HttpAsyncEngine(pub PythonAsyncEngine);

impl From<PythonAsyncEngine> for HttpAsyncEngine {
    fn from(engine: PythonAsyncEngine) -> Self {
        Self(engine)
    }
}

#[pymethods]
impl HttpAsyncEngine {
    /// Create a new instance of the HttpAsyncEngine
    /// This is a simple extension of the PythonAsyncEngine that handles HttpError
    /// exceptions from Python and converts them to the Rust version of HttpError
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response are not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        Ok(PythonAsyncEngine::new(generator, event_loop)?.into())
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for HttpAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        match self.0.generate(request).await {
            Ok(res) => Ok(res),

            // Inspect the error - if it was an HttpError from Python, extract the code and message
            // and return the rust version of HttpError
            Err(e) => {
                if let Some(py_err) = e.downcast_ref::<PyErr>() {
                    Python::with_gil(|py| {
                        if let Ok(http_error_instance) = py_err
                            .clone_ref(py)
                            .into_value(py)
                            .extract::<PyRef<HttpError>>(py)
                        {
                            Err(http_error::HttpError {
                                code: http_error_instance.code,
                                message: http_error_instance.message.clone(),
                            })?
                        } else {
                            Err(error!("Python Error: {}", py_err.to_string()))
                        }
                    })
                } else {
                    Err(e)
                }
            }
        }
    }
}
