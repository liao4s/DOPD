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

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    common::{self, SamplingOptionsProvider, StopConditionsProvider},
    ContentProvider,
};

pub mod chat_completions;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod nvext;
pub mod responses;
pub mod validate;

use validate::{
    validate_range, FREQUENCY_PENALTY_RANGE, PRESENCE_PENALTY_RANGE, TEMPERATURE_RANGE, TOP_P_RANGE,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct AnnotatedDelta<R> {
    pub delta: R,
    pub id: Option<String>,
    pub event: Option<String>,
    pub comment: Option<String>,
}

trait OpenAISamplingOptionsProvider {
    fn get_temperature(&self) -> Option<f32>;

    fn get_top_p(&self) -> Option<f32>;

    fn get_frequency_penalty(&self) -> Option<f32>;

    fn get_presence_penalty(&self) -> Option<f32>;

    fn nvext(&self) -> Option<&nvext::NvExt>;
}

trait OpenAIStopConditionsProvider {
    fn get_max_tokens(&self) -> Option<u32>;

    fn get_min_tokens(&self) -> Option<u32>;

    fn get_stop(&self) -> Option<Vec<String>>;

    fn nvext(&self) -> Option<&nvext::NvExt>;
}

impl<T: OpenAISamplingOptionsProvider> SamplingOptionsProvider for T {
    fn extract_sampling_options(&self) -> Result<common::SamplingOptions> {
        // let result = self.validate();
        // if let Err(e) = result {
        //     return Err(format!("Error validating sampling options: {}", e));
        // }

        let mut temperature = validate_range(self.get_temperature(), &TEMPERATURE_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating temperature: {}", e))?;
        let mut top_p = validate_range(self.get_top_p(), &TOP_P_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating top_p: {}", e))?;
        let frequency_penalty =
            validate_range(self.get_frequency_penalty(), &FREQUENCY_PENALTY_RANGE)
                .map_err(|e| anyhow::anyhow!("Error validating frequency_penalty: {}", e))?;
        let presence_penalty = validate_range(self.get_presence_penalty(), &PRESENCE_PENALTY_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating presence_penalty: {}", e))?;

        if let Some(nvext) = self.nvext() {
            let greedy = nvext.greed_sampling.unwrap_or(false);
            if greedy {
                top_p = None;
                temperature = None;
            }
        }

        Ok(common::SamplingOptions {
            n: None,
            best_of: None,
            frequency_penalty,
            presence_penalty,
            repetition_penalty: None,
            temperature,
            top_p,
            top_k: None,
            min_p: None,
            seed: None,
            use_beam_search: None,
            length_penalty: None,
        })
    }
}

impl<T: OpenAIStopConditionsProvider> StopConditionsProvider for T {
    fn extract_stop_conditions(&self) -> Result<common::StopConditions> {
        let max_tokens = self.get_max_tokens();
        let min_tokens = self.get_min_tokens();
        let stop = self.get_stop();

        if let Some(stop) = &stop {
            if stop.len() > 4 {
                anyhow::bail!("stop conditions must be less than 4")
            }
        }

        let mut ignore_eos = None;

        if let Some(nvext) = self.nvext() {
            ignore_eos = nvext.ignore_eos;
        }

        Ok(common::StopConditions {
            max_tokens,
            min_tokens,
            stop,
            stop_token_ids_hidden: None,
            ignore_eos,
        })
    }
}

pub trait DeltaGeneratorExt<ResponseType: Send + Sync + 'static + std::fmt::Debug>:
    Send + Sync + 'static
{
    fn choice_from_postprocessor(
        &mut self,
        response: common::llm_backend::BackendOutput,
    ) -> Result<ResponseType>;

    /// Gets the current prompt token count (Input Sequence Length).
    fn get_isl(&self) -> Option<u32>;
}
