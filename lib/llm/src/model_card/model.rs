// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Model Deployment Card
//!
//! The ModelDeploymentCard (MDC) is the primary model configuration structure that will be available to any
//! component that needs to interact with the model or its dependent artifacts.
//!
//! The ModelDeploymentCard contains LLM model deployment configuration information:
//! - Display name and service name for the model
//! - Model information (ModelInfoType)
//! - Tokenizer configuration (TokenizerKind)
//! - Prompt formatter settings (PromptFormatterArtifact)
//! - Various metadata like revision, publish time, etc.

use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_runtime::{slug::Slug, storage::key_value_store::Versioned, transports::nats};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;
use url::Url;

use crate::gguf::{Content, ContentConfig, ModelConfigLike};
use crate::protocols::TokenIdType;

/// If a model deployment card hasn't been refreshed in this much time the worker is likely gone
const CARD_MAX_AGE: chrono::TimeDelta = chrono::TimeDelta::minutes(5);

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ModelInfoType {
    HfConfigJson(String),
    GGUF(PathBuf),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HfTokenizerJson(String),
    GGUF(Box<HfTokenizer>),
}

/// Supported types of prompt formatters.
///
/// We need a way to associate the prompt formatter template definition with an associated
/// data model which is expected for rendering.
///
/// All current prompt formatters are Jinja2 templates which use the OpenAI ChatCompletionRequest
/// format. However, we currently do not have a discovery path to know if the model supports tool use
/// unless we inspect the template.
///
/// TODO(): Add an enum for the PromptFormatDataModel with at minimum arms for:
/// - OaiChat
/// - OaiChatToolUse
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PromptFormatterArtifact {
    HfTokenizerConfigJson(String),
    GGUF(PathBuf),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConfig {
    HfGenerationConfigJson(String),
    GGUF(PathBuf),
}

#[derive(Serialize, Deserialize, Clone, Debug, Builder, Default)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    /// Identifier to expect in OpenAI compatible HTTP request, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    /// This will get slugified for use in NATS.
    pub service_name: String,

    /// Model information
    pub model_info: Option<ModelInfoType>,

    /// Tokenizer configuration
    pub tokenizer: Option<TokenizerKind>,

    /// Prompt Formatter configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_formatter: Option<PromptFormatterArtifact>,

    /// Generation config - default sampling params
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_config: Option<GenerationConfig>,

    /// Prompt Formatter Config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_context: Option<Vec<PromptContextMixin>>,

    /// When this card was last advertised by a worker. None if not yet published.
    pub last_published: Option<chrono::DateTime<chrono::Utc>>,

    /// Incrementing count of how many times we published this card
    #[serde(default, skip_serializing)]
    pub revision: u64,

    /// Max context (in number of tokens) this model can handle
    pub context_length: u32,

    /// Size of a KV cache block - vllm only currently
    /// Passed to the engine and the KV router.
    pub kv_cache_block_size: u32,
}

impl ModelDeploymentCard {
    pub fn builder() -> ModelDeploymentCardBuilder {
        ModelDeploymentCardBuilder::default()
    }

    /// Create a ModelDeploymentCard where only the name is filled in.
    ///
    /// Single-process setups don't need an MDC to communicate model details, but it
    /// simplifies the code to assume we always have one. This is how you get one in those
    /// cases. A quasi-null object: <https://en.wikipedia.org/wiki/Null_object_pattern>
    pub fn with_name_only(name: &str) -> ModelDeploymentCard {
        ModelDeploymentCard {
            display_name: name.to_string(),
            service_name: Slug::slugify(name).to_string(),
            ..Default::default()
        }
    }

    /// How often we should check if a model deployment card expired because it's workers are gone
    pub fn expiry_check_period() -> Duration {
        match CARD_MAX_AGE.to_std() {
            Ok(duration) => duration / 3,
            Err(_) => {
                // Only happens if CARD_MAX_AGE is negative, which it isn't
                unreachable!("Cannot run card expiry watcher, invalid CARD_MAX_AGE");
            }
        }
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(file)?)?)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(json: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(json)?)
    }

    //
    // Methods
    //

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
    }

    pub fn set_service_name(&mut self, service_name: &str) {
        self.service_name = service_name.to_string();
    }

    pub fn slug(&self) -> Slug {
        Slug::from_string(&self.display_name)
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn mdcsum(&self) -> String {
        let json = self.to_json().unwrap();
        format!("{}", blake3::hash(json.as_bytes()))
    }

    /// Was this card last published a long time ago, suggesting the worker is gone?
    pub fn is_expired(&self) -> bool {
        if let Some(last_published) = self.last_published.as_ref() {
            chrono::Utc::now() - last_published > CARD_MAX_AGE
        } else {
            false
        }
    }

    /// Is this a full model card with tokenizer?
    /// There are cases where we have a placeholder card (see `with_name_only`).
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn tokenizer_hf(&self) -> anyhow::Result<HfTokenizer> {
        match &self.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => {
                HfTokenizer::from_file(file).map_err(anyhow::Error::msg)
            }
            Some(TokenizerKind::GGUF(t)) => Ok(*t.clone()),
            None => {
                anyhow::bail!("Blank ModelDeploymentCard does not have a tokenizer");
            }
        }
    }

    pub fn is_gguf(&self) -> bool {
        match &self.model_info {
            Some(info) => info.is_gguf(),
            None => false,
        }
    }

    /// Move the files this MDC uses into the NATS object store.
    /// Updates the URI's to point to NATS.
    pub async fn move_to_nats(&mut self, nats_client: nats::Client) -> Result<()> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        tracing::debug!(
            nats_addr,
            %bucket_name,
            "Uploading model deployment card fields to NATS"
        );

        macro_rules! nats_upload {
            ($field:expr, $enum_variant:path, $filename:literal) => {
                if let Some($enum_variant(src_file)) = $field.take() {
                    if !nats::is_nats_url(&src_file) {
                        let target = format!("nats://{nats_addr}/{bucket_name}/{}", $filename);
                        nats_client
                            .object_store_upload(
                                &std::path::PathBuf::from(&src_file),
                                url::Url::parse(&target)?,
                            )
                            .await?;
                        $field = Some($enum_variant(target));
                    }
                }
            };
        }

        nats_upload!(self.model_info, ModelInfoType::HfConfigJson, "config.json");
        nats_upload!(
            self.prompt_formatter,
            PromptFormatterArtifact::HfTokenizerConfigJson,
            "tokenizer_config.json"
        );
        nats_upload!(
            self.tokenizer,
            TokenizerKind::HfTokenizerJson,
            "tokenizer.json"
        );
        nats_upload!(
            self.gen_config,
            GenerationConfig::HfGenerationConfigJson,
            "generation_config.json"
        );

        Ok(())
    }

    /// Move the files this MDC uses from the NATS object store to local disk.
    /// Updates the URI's to point to the created files.
    ///
    /// The returned TempDir must be kept alive, it cleans up on drop.
    pub async fn move_from_nats(&mut self, nats_client: nats::Client) -> Result<tempfile::TempDir> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        let target_dir = tempfile::TempDir::with_prefix(bucket_name.to_string())?;
        tracing::debug!(
            nats_addr,
            %bucket_name,
            target_dir = %target_dir.path().display(),
            "Downloading model deployment card fields from NATS"
        );

        macro_rules! nats_download {
            ($field:expr, $enum_variant:path, $filename:literal) => {
                if let Some($enum_variant(src_url)) = $field.take() {
                    if nats::is_nats_url(&src_url) {
                        let target = target_dir.path().join($filename);
                        nats_client
                            .object_store_download(Url::parse(&src_url)?, &target)
                            .await?;
                        $field = Some($enum_variant(target.display().to_string()));
                    }
                }
            };
        }

        nats_download!(self.model_info, ModelInfoType::HfConfigJson, "config.json");
        nats_download!(
            self.prompt_formatter,
            PromptFormatterArtifact::HfTokenizerConfigJson,
            "tokenizer_config.json"
        );
        nats_download!(
            self.tokenizer,
            TokenizerKind::HfTokenizerJson,
            "tokenizer.json"
        );
        nats_download!(
            self.gen_config,
            GenerationConfig::HfGenerationConfigJson,
            "generation_config.json"
        );

        Ok(target_dir)
    }

    /// Delete this card from the key-value store and it's URLs from the object store
    pub async fn delete_from_nats(&mut self, nats_client: nats::Client) -> Result<()> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        tracing::trace!(
            nats_addr,
            %bucket_name,
            "Delete model deployment card from NATS"
        );
        nats_client
            .object_store_delete_bucket(bucket_name.as_ref())
            .await
    }
}

impl Versioned for ModelDeploymentCard {
    fn revision(&self) -> u64 {
        self.revision
    }

    fn set_revision(&mut self, revision: u64) {
        self.last_published = Some(chrono::Utc::now());
        self.revision = revision;
    }
}

impl fmt::Display for ModelDeploymentCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.slug())
    }
}
pub trait ModelInfo: Send + Sync {
    /// Model type
    fn model_type(&self) -> String;

    /// Token ID for the beginning of sequence
    fn bos_token_id(&self) -> TokenIdType;

    /// Token ID for the end of sequence
    fn eos_token_ids(&self) -> Vec<TokenIdType>;

    /// Maximum position embeddings / max sequence length
    /// TODO: This is only used in a single test, no other code. Remove?
    fn max_position_embeddings(&self) -> Option<usize>;

    /// Vocabulary size
    /// TODO: This is only used in a single test, no other code. Remove?
    fn vocab_size(&self) -> Option<usize>;
}

impl ModelInfoType {
    pub async fn get_model_info(&self) -> Result<Arc<dyn ModelInfo>> {
        match self {
            Self::HfConfigJson(info) => HFConfig::from_json_file(info).await,
            Self::GGUF(path) => HFConfig::from_gguf(path),
        }
    }
    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::GGUF(_))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFConfig {
    /// denotes the mixin to the flattened data model which can be present
    /// in the config.json file
    architectures: Vec<String>,

    /// general model type
    model_type: String,

    text_config: Option<HFTextConfig>,

    // Sometimes it's inside HFTextConfig, sometimes it's here
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFTextConfig {
    // It can take multiple attempts to load this, so Option
    bos_token_id: Option<TokenIdType>,

    // We set this once bos_token_id is loaded so we don't have to deal with Option
    #[serde(default)]
    final_bos_token_id: TokenIdType,

    eos_token_id: Option<serde_json::Value>,

    #[serde(default)]
    final_eos_token_ids: Vec<TokenIdType>,

    /// max sequence length
    max_position_embeddings: Option<usize>,

    /// number of layers in the model
    num_hidden_layers: usize,

    /// number of attention heads in the model
    num_attention_heads: Option<usize>,

    /// Vocabulary size
    vocab_size: Option<usize>,
}

impl HFConfig {
    async fn from_json_file(file: &str) -> Result<Arc<dyn ModelInfo>> {
        let file_pathbuf = PathBuf::from(file);
        let contents = std::fs::read_to_string(file)?;
        let mut config: Self = serde_json::from_str(&contents)?;
        if config.text_config.is_none() {
            let text_config: HFTextConfig = serde_json::from_str(&contents)?;
            config.text_config = Some(text_config);
        }
        // Sometimes bos_token_id is in generation_config.json not config.json
        let Some(text_config) = config.text_config.as_mut() else {
            anyhow::bail!(
                "Missing text config fields (model_type, eos_token_ids, etc) in config.json"
            );
        };

        if text_config.bos_token_id.is_none() {
            let bos_token_id = crate::file_json_field::<TokenIdType>(
                &Path::join(
                    file_pathbuf.parent().unwrap_or(&PathBuf::from("")),
                    "generation_config.json",
                ),
                "bos_token_id",
            )
            .context(
                "missing bos_token_id in generation_config.json and config.json, cannot load",
            )?;
            text_config.bos_token_id = Some(bos_token_id);
        }
        // Now that we have it for sure, set it in the non-Option field
        let final_bos_token_id = text_config.bos_token_id.take().unwrap();
        text_config.final_bos_token_id = final_bos_token_id;

        // TODO: refactor this when we switch to per-architecture tokenization
        let final_eos_token_ids: Vec<TokenIdType> = config
            .eos_token_id
            .as_ref()
            .or(text_config.eos_token_id.as_ref())
            .and_then(|v| {
                if v.is_number() {
                    v.as_number()
                        .and_then(|n| n.as_u64())
                        .map(|n| vec![n as TokenIdType])
                } else if v.is_array() {
                    let arr = v.as_array().unwrap(); // Safety: We just checked
                    Some(
                        arr.iter()
                            .filter_map(|inner_v| {
                                inner_v
                                    .as_number()
                                    .and_then(|n| n.as_u64())
                                    .map(|n| n as TokenIdType)
                            })
                            .collect(),
                    )
                } else {
                    tracing::error!(
                        ?v,
                        file,
                        "eos_token_id is not a number or an array, cannot use"
                    );
                    None
                }
            })
            .or_else(|| {
                // Maybe it's in generation_config.json
                crate::file_json_field(
                    &Path::join(
                        file_pathbuf.parent().unwrap_or(&PathBuf::from("")),
                        "generation_config.json",
                    ),
                    "eos_token_id",
                )
                .inspect_err(
                    |err| tracing::warn!(%err, "Missing eos_token_id in generation_config.json"),
                )
                .ok()
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing eos_token_id in config.json and generation_config.json, cannot load"
                )
            })?;
        text_config.final_eos_token_ids = final_eos_token_ids;

        Ok(Arc::new(config))
    }
    fn from_gguf(gguf_file: &Path) -> Result<Arc<dyn ModelInfo>> {
        let content = load_gguf(gguf_file)?;
        let model_config_metadata: ContentConfig = (&content).into();
        let num_hidden_layers =
            content.get_metadata()[&format!("{}.block_count", content.arch())].to_u32()? as usize;

        let bos_token_id = content.get_metadata()["tokenizer.ggml.bos_token_id"].to_u32()?;
        let eos_token_id = content.get_metadata()["tokenizer.ggml.eos_token_id"].to_u32()?;

        // to_vec returns a Vec that's already there, so it's cheap
        let vocab_size = content.get_metadata()["tokenizer.ggml.tokens"]
            .to_vec()?
            .len();

        let arch = content.arch().to_string();
        Ok(Arc::new(HFConfig {
            architectures: vec![format!("{}ForCausalLM", capitalize(&arch))],
            // "general.architecture"
            model_type: arch,
            text_config: Some(HFTextConfig {
                bos_token_id: None,
                final_bos_token_id: bos_token_id,

                eos_token_id: None,
                final_eos_token_ids: vec![eos_token_id],

                // "llama.context_length"
                max_position_embeddings: Some(model_config_metadata.max_seq_len()),
                // "llama.block_count"
                num_hidden_layers,
                // "llama.attention.head_count"
                num_attention_heads: Some(model_config_metadata.num_attn_heads()),
                // "tokenizer.ggml.tokens".len()
                vocab_size: Some(vocab_size),
            }),
            eos_token_id: None,
        }))
    }
}

impl ModelInfo for HFConfig {
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn bos_token_id(&self) -> TokenIdType {
        self.text_config.as_ref().unwrap().final_bos_token_id
    }

    fn eos_token_ids(&self) -> Vec<TokenIdType> {
        self.text_config
            .as_ref()
            .unwrap()
            .final_eos_token_ids
            .clone()
    }

    fn max_position_embeddings(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().max_position_embeddings
    }

    fn vocab_size(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().vocab_size
    }
}

impl TokenizerKind {
    pub fn from_gguf(gguf_file: &Path) -> anyhow::Result<Self> {
        let content = load_gguf(gguf_file)?;
        let out = crate::gguf::convert_gguf_to_hf_tokenizer(&content)
            .with_context(|| gguf_file.display().to_string())?;
        Ok(TokenizerKind::GGUF(Box::new(out.tokenizer)))
    }
}

pub(crate) fn load_gguf(gguf_file: &Path) -> anyhow::Result<Content> {
    let filename = gguf_file.display().to_string();
    let mut f = File::open(gguf_file).with_context(|| filename.clone())?;
    // vec because GGUF can be split into multiple files (shards)
    let mut readers = vec![&mut f];
    crate::gguf::Content::from_readers(&mut readers).with_context(|| filename.clone())
}

fn capitalize(s: &str) -> String {
    s.chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 0 {
                c.to_uppercase().to_string()
            } else {
                c.to_lowercase().to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::HFConfig;
    use std::path::Path;

    #[tokio::test]
    pub async fn test_config_json_llama3() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct/config.json");
        let config = HFConfig::from_json_file(&config_file.display().to_string()).await?;
        assert_eq!(config.bos_token_id(), 128000);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_config_json_llama4() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/Llama-4-Scout-17B-16E-Instruct/config.json");
        let config = HFConfig::from_json_file(&config_file.display().to_string()).await?;
        assert_eq!(config.bos_token_id(), 200000);
        Ok(())
    }
}
