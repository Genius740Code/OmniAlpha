// NNEvaluator<G> : Evaluator<G> loading a TorchScript model via libtorch.
// Contract: evaluate(const State*, EvalResult*, count)
//   - priors[i] = softmax over legal moves in G::legal_moves order
//   - value from player-to-move view, in [-1, 1]
// Compiled only when GAI_LIBTORCH=ON (see tools/common.hpp + CMakeLists.txt).
#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include "engine/core/api.hpp"
#include "engine/inference/evaluator.hpp"

namespace gai {

static inline void softmax_inplace(float* x, int n) {
  if (n <= 0) return;
  float m = x[0];
  for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
  float s = 0.f;
  for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); s += x[i]; }
  if (s > 0.f) { for (int i = 0; i < n; i++) x[i] /= s; }
  else { for (int i = 0; i < n; i++) x[i] = 1.f / (float)n; }
}

template <GameLike G>
struct NNEvaluator : Evaluator<G> {
  explicit NNEvaluator(std::string path, bool strict = false)
      : model_path_(std::move(path)), strict_(strict) {}

  // One instance per thread (tools build one per worker; CachingEvaluator has
  // no mutex), so the reused member batch below is not shared across threads.
  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    if (!loaded_) try_load();
    if (!module_) { silent_fallback(states, out, count, "load"); return; }
    const int P = G::kInputPlanes, H = G::kInputH, W = G::kInputW;
    const int PW = P * H * W;
    try {
      torch::NoGradGuard ng;
      // Reuse the CPU staging buffer across calls (per-call alloc was the
      // dominant fixed cost at small batch sizes). Resized+re-zeroed only when
      // needed. On CUDA this is then copied to the device in one shot; raw
      // memcpy into device memory would be invalid.
      if (!staging_.defined() || staging_.size(0) != count || staging_.size(1) != P
          || staging_.size(2) != H || staging_.size(3) != W)
        staging_ = torch::zeros({count, P, H, W}, torch::kFloat32);
      else
        staging_.zero_();
      float* bptr = staging_.data_ptr<float>();
      std::vector<float> enc(PW);
      Move mv[G::kMaxMoves];
      for (int i = 0; i < count; i++) {
        G::encode(states[i], enc.data());
        std::memcpy(bptr + (size_t)i * PW, enc.data(), sizeof(float) * PW);
      }
      std::vector<torch::jit::IValue> inputs{device_ == torch::kCPU ? staging_ : staging_.to(device_)};
      auto outv = module_->forward(inputs);
      at::Tensor logits, val;
      if (outv.isTuple()) {
        auto t = outv.toTuple()->elements();
        logits = t[0].toTensor();
        val = t[1].toTensor();
      } else {
        silent_fallback(states, out, count, "non-tuple");
        return;
      }
      // Host-side reads need the outputs on CPU (no-op on the CPU path).
      if (device_ != torch::kCPU) {
        logits = logits.to(torch::kCPU).contiguous();
        val = val.to(torch::kCPU).contiguous();
      }
      const float* lp = logits.data_ptr<float>();
      const float* vp = val.data_ptr<float>();
      std::vector<float> legal;
      legal.resize(G::kMaxMoves);
      for (int i = 0; i < count; i++) {
        int n = G::legal_moves(states[i], mv);
        out[i].n = n;
        if (n <= 0) { out[i].value = 0; continue; }
        int base = i * G::kActionCount;
        for (int j = 0; j < n; j++) legal[j] = lp[base + (int)mv[j]];
        softmax_inplace(legal.data(), n);
        for (int j = 0; j < n; j++) out[i].priors[j] = legal[j];
        float vv = vp[i];
        if (!std::isfinite(vv)) { silent_fallback(states, out, count, "non-finite-value"); return; }
        for (int j = 0; j < n; j++) {
          if (!std::isfinite(legal[j])) { silent_fallback(states, out, count, "non-finite-logit"); return; }
        }
        out[i].value = vv > 1.f ? 1.f : vv < -1.f ? -1.f : vv;
      }
    } catch (const std::exception& e) {
      if (fallbacks_ < 5)
        std::fprintf(stderr, "NNEvaluator: evaluate exception: %s\n", e.what());
      silent_fallback(states, out, count, "exception");
    }
  }
  long fallbacks() const { return fallbacks_; }

 private:
  void try_load() {
    loaded_ = true;
    try {
      // Device selection: GAI_NN_DEVICE=cuda loads the module on GPU
      // (audit B1). Default stays CPU so existing runs are unchanged; if CUDA
      // is requested but unavailable we warn and fall back to CPU.
      torch::Device dev = torch::kCPU;
      if (const char* d = std::getenv("GAI_NN_DEVICE"); d && std::strcmp(d, "cuda") == 0) {
        if (torch::cuda::is_available()) dev = torch::kCUDA;
        else std::fprintf(stderr, "NNEvaluator: GAI_NN_DEVICE=cuda but CUDA is unavailable; using CPU\n");
      }
      device_ = dev;
      module_ = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(model_path_, dev));
      module_->eval();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "NNEvaluator: cannot load '%s' (%s); falling back to uniform\n",
                   model_path_.c_str(), e.what());
      module_.reset();
    }
  }
  void uniform_fallback(const typename G::State* states, EvalResult<G>* out, int count) {
    Move mv[G::kMaxMoves];
    for (int i = 0; i < count; i++) {
      int n = G::legal_moves(states[i], mv);
      out[i].n = n;
      out[i].value = 0;
      for (int j = 0; j < n; j++) out[i].priors[j] = n ? 1.f / (float)n : 0.f;
    }
  }
  // Strict mode: any silent fallback is fatal (exit 4) instead of uniform.
  // A corrupt .ts or NaN weights must never poison a run with exit code 0.
  [[noreturn]] void die(const char* why) {
    std::fprintf(stderr, "NNEvaluator(strict): fallback '%s' for '%s'; aborting (exit 4)\n",
                 why, model_path_.c_str());
    std::exit(4);
  }
  void silent_fallback(const typename G::State* states, EvalResult<G>* out, int count, const char* why) {
    fallbacks_++;
    if (strict_) die(why);
    std::fprintf(stderr, "NNEvaluator: fallback '%s' for '%s' (#%ld); using uniform\n",
                 why, model_path_.c_str(), fallbacks_);
    uniform_fallback(states, out, count);
  }
  std::string model_path_;
  std::shared_ptr<torch::jit::script::Module> module_;
  torch::Device device_ = torch::kCPU;
  at::Tensor staging_;  // reused CPU input buffer (instance is per-thread)
  bool loaded_ = false, strict_ = false;
  long fallbacks_ = 0;
};

}  // namespace gai
