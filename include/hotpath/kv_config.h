#pragma once

#include <cstdint>
#include <string>

namespace hotpath {

// Parses a HuggingFace config.json string and returns the number of KV cache
// bytes consumed per input token by this model.
//
// Formula: 2 (K+V) * num_kv_heads * head_dim * dtype_bytes * num_hidden_layers
//
// Returns 0 if the config lacks enough information to compute the value.
// Accepts any model architecture that stores standard transformer fields
// (Llama, Qwen, Mistral, GPT-2, etc.).
int64_t parse_kv_bytes_per_token_from_config(const std::string& config_json);

// Looks up the local HuggingFace hub cache (~/.cache/huggingface/hub) for the
// given model name (e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct") and reads
// its config.json.
//
// Returns bytes-per-token, or 0 if the model is not cached locally.
int64_t detect_kv_bytes_per_token(const std::string& model_name);

}  // namespace hotpath
