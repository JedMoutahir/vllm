# Simple wrapper to include any extra utilities; vLLM image already has the server.
FROM vllm/vllm-openai:latest

# Optionally install extra packages (e.g., curl, jq) for debugging
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Default command is provided by the base image; compose.yaml sets args.
