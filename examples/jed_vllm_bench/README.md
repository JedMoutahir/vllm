# Jed’s vLLM Throughput & Latency Benchmark (Docker + Python client)

This drop-in folder lets you **spin up vLLM** for any HF model and **benchmark throughput/latency** with a simple async client.
- Works locally or on a GPU server (A100/H100/g6e.xlarge etc.).
- Produces a CSV with per-request latency and an aggregate summary JSON.
- Optional **matplotlib** plot script.

> You supply a model id (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

---

## Contents
```
examples/jed_vllm_bench/
├── README.md
├── Dockerfile
├── compose.yaml
├── client_requirements.txt
├── bench_client.py
├── analyze_results.py
└── sample_prompts.txt
```

---

## Quick start (Docker Compose)

1) Edit `compose.yaml` env vars if needed:
```yaml
environment:
  - HF_TOKEN=your_hf_token_if_needed
  - MODEL=meta-llama/Llama-3.1-8B-Instruct
  - MAX_MODEL_LEN=8192
  - TP_SIZE=1
  - PORT=8000
```

2) Launch server:
```bash
docker compose -f examples/jed_vllm_bench/compose.yaml up -d
# server will listen on http://localhost:8000
```

3) Create a client venv and run the load:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r examples/jed_vllm_bench/client_requirements.txt

python examples/jed_vllm_bench/bench_client.py   --endpoint http://localhost:8000/v1   --prompts-file examples/jed_vllm_bench/sample_prompts.txt   --concurrency 16 --requests 200   --max-new-tokens 256 --temperature 0.2   --out runs/vllm_bench
```

4) Analyze:
```bash
python examples/jed_vllm_bench/analyze_results.py   --results runs/vllm_bench/results.jsonl   --out runs/vllm_bench
# Produces: summary.json, latency_hist.png
```

---

## Tuning tips

- Increase `--concurrency` to approach saturation; monitor GPU util with `nvidia-smi`.
- Adjust `--max-new-tokens` to simulate longer generations.
- For multi-GPU: set `TP_SIZE>1` in `compose.yaml` (tensor parallel), and ensure the container sees multiple GPUs.
- For streaming latency curves, set `--stream` on the client (not default here).

---

## Notes

- The server uses the official `vllm/vllm-openai` image. You can pass `--dtype auto` or `--gpu-memory-utilization` via `VLLM_ARGS` in compose if desired.
- If your model requires gated access or a token, set `HF_TOKEN`.
- The client uses the **OpenAI-compatible** `/v1/chat/completions` endpoint.
