# Engineering Tradeoffs

## 1. Quantization Level (INT8 vs. FP16)
- **Decision:** Used INT8 quantization for weights.
- **Tradeoff:** Accepted a ~2% drop in intent classification accuracy to achieve a 4x reduction in model size.
- **Reasoning:** On 2GB edge devices, loading a full FP16 model would cause an Out-Of-Memory (OOM) error. INT8 is the "sweet spot" for performance on CPUs without specialized hardware.

## 2. Framework Choice (OpenVINO vs. Native PyTorch)
- **Decision:** Compiled the model using OpenVINO Runtime.
- **Tradeoff:** Added a compilation step to the pipeline.
- **Reasoning:** OpenVINO provides hardware-specific graph optimizations (like operator fusion) that reduced inference latency from ~2.5s to <900ms on standard Intel/AMD mobile CPUs.

## 3. Sequence Length Limitation
- **Decision:** Capped input sequence length to 128 tokens.
- **Tradeoff:** Reduced the model's ability to handle very long "paragraph-style" user queries.
- **Reasoning:** KV-cache growth is the primary driver of memory spikes. Limiting the sequence length ensures the peak memory never exceeds the 2GB hardware threshold.