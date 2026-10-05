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

  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    if (!loaded_) try_load();
    if (!module_) { silent_fallback(states, out, count, "load"); return; }
    const int P = G::kInputPlanes, H = G::kInputH, W = G::kInputW;
    const int PW = P * H * W;
    try {
      torch::NoGradGuard ng;
      at::Tensor batch = torch::zeros({count, P, H, W}, torch::kFloat32);
      float* bptr = batch.data_ptr<float>();
      std::vector<float> enc(PW);
      Move mv[G::kMaxMoves];
      for (int i = 0; i < count; i++) {
        G::encode(states[i], enc.data());
        std::memcpy(bptr + (size_t)i * PW, enc.data(), sizeof(float) * PW);
      }
      std::vector<torch::jit::IValue> inputs{batch};
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
    } catch (const std::exception&) {
      silent_fallback(states, out, count, "exception");
    }
  }
  long fallbacks() const { return fallbacks_; }

 private:
  void try_load() {
    loaded_ = true;
    try {
      module_ = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(model_path_, torch::kCPU));
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
  bool loaded_ = false, strict_ = false;
  long fallbacks_ = 0;
};

}  // namespace gai
