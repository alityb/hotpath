#include "hotpath/kv_config.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

namespace hotpath {
namespace {

// Extracts the first integer value for "key": <int> from a JSON string.
// Returns 0 if the key is not found.
int64_t json_int(const std::string& json, const std::string& key) {
  const std::regex re("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch m;
  if (std::regex_search(json, m, re)) {
    return std::stoll(m[1].str());
  }
  return 0;
}

// Extracts the first string value for "key": "value" from a JSON string.
// Returns "" if the key is not found.
std::string json_str(const std::string& json, const std::string& key) {
  const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch m;
  if (std::regex_search(json, m, re)) {
    return m[1].str();
  }
  return "";
}

int dtype_bytes(const std::string& dtype) {
  if (dtype == "float32" || dtype == "tf32") return 4;
  if (dtype == "float8_e4m3fn" || dtype == "float8_e5m2") return 1;
  // float16, bfloat16, half, and anything unrecognised default to 2
  return 2;
}

// Converts "org/model" → "models--org--model" for HF hub directory lookup.
std::string model_to_hf_dir(const std::string& model_name) {
  std::string dir = "models--";
  for (char c : model_name) {
    if (c == '/') {
      dir += "--";
    } else {
      dir += c;
    }
  }
  return dir;
}

}  // namespace

int64_t parse_kv_bytes_per_token_from_config(const std::string& config_json) {
  const int64_t num_layers = json_int(config_json, "num_hidden_layers")
                           ?: json_int(config_json, "n_layer")
                           ?: json_int(config_json, "num_layers");
  if (num_layers <= 0) return 0;

  const int64_t num_attention_heads = json_int(config_json, "num_attention_heads")
                                    ?: json_int(config_json, "n_head");
  if (num_attention_heads <= 0) return 0;

  // GQA: num_key_value_heads may be smaller than num_attention_heads.
  // Fall back to full MHA if the field is absent.
  int64_t num_kv_heads = json_int(config_json, "num_key_value_heads");
  if (num_kv_heads <= 0) num_kv_heads = num_attention_heads;

  // head_dim may be explicit (Llama-3, Qwen-2) or implied by hidden_size.
  int64_t head_dim = json_int(config_json, "head_dim");
  if (head_dim <= 0) {
    const int64_t hidden_size = json_int(config_json, "hidden_size")
                              ?: json_int(config_json, "n_embd");
    if (hidden_size <= 0 || num_attention_heads <= 0) return 0;
    head_dim = hidden_size / num_attention_heads;
  }
  if (head_dim <= 0) return 0;

  const std::string dtype = json_str(config_json, "torch_dtype");
  const int bytes_per_elem = dtype_bytes(dtype);

  // 2 = K + V
  return 2LL * num_kv_heads * head_dim * bytes_per_elem * num_layers;
}

int64_t detect_kv_bytes_per_token(const std::string& model_name) {
  if (model_name.empty()) return 0;

  const char* home = std::getenv("HOME");
  if (!home) return 0;

  const std::filesystem::path snapshots_dir =
      std::filesystem::path(home) / ".cache" / "huggingface" / "hub" /
      model_to_hf_dir(model_name) / "snapshots";

  std::error_code ec;
  if (!std::filesystem::exists(snapshots_dir, ec)) return 0;

  for (const auto& entry : std::filesystem::directory_iterator(snapshots_dir, ec)) {
    if (!entry.is_directory()) continue;
    const std::filesystem::path config_path = entry.path() / "config.json";
    if (!std::filesystem::exists(config_path)) continue;

    std::ifstream f(config_path);
    if (!f.is_open()) continue;
    const std::string content(std::istreambuf_iterator<char>(f),
                              std::istreambuf_iterator<char>{});
    const int64_t result = parse_kv_bytes_per_token_from_config(content);
    if (result > 0) return result;
  }

  return 0;
}

}  // namespace hotpath
