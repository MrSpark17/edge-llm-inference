# Failure Modes & Mitigations

## 1. Out-Of-Memory (OOM) on Long Contexts
- **Failure:** If a user provides a prompt exceeding 128 tokens, the KV-cache may spike beyond the 2GB RAM limit.
- **Mitigation:** Implemented a hard truncation at the Tokenizer level. Future work includes implementing "Sliding Window Attention" to handle longer context without memory linear growth.

## 2. Intent Drift due to Quantization
- **Failure:** Aggressive INT8 quantization can cause "stochastic parrot" behavior where the model loses nuance in intent classification (e.g., confusing "Refund" with "Status Update").
- **Mitigation:** Currently using a high-precision fallback for low-confidence scores. We plan to implement "Quantization-Aware Training" (QAT) to recover lost accuracy.

## 3. Cold Start Latency
- **Failure:** The first inference request takes ~2-3 seconds as the OpenVINO model is loaded into memory.
- **Mitigation:** Implemented a "Lazy Loading" strategy with a progress indicator in the UI. For production, we would use a model warm-up script on system boot.