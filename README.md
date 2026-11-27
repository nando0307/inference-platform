# Inference Platform

A learning project for running and measuring local LLM inference. It includes a CPU FastAPI service, streaming and bounded request handling, and opt-in experiments for batching, vLLM, routing, and GPU placement. Experimental results are evidence for specific runs—not production SLOs or general performance claims.

## Quick start

Requires Python 3.12. Install the pinned direct dependencies and run one local worker:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

The app loads `Qwen/Qwen2.5-0.5B-Instruct` on CPU in float32 at startup. Model files must be available in the Hugging Face cache or downloadable; startup fails if loading fails. Keep one worker and do not use reload mode when checking model lifetime.

Check liveness and readiness:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

Chat completions use `POST /v1/chat/completions` (OpenAI-compatible request/streaming format). The local service is loopback-bound and unauthenticated by default. For a less-trusted network, set `INFERENCE_API_TOKEN`; never expose credentials in source control or command history. See [API implementation](src/api.py) for request details.

Run the CPU test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Project guides

- [Request routing](docs/request-routing.md): static worker policies, admission behavior, and CPU-only simulation.
- [Adaptive placement](docs/adaptive-placement.md): placement planning and its constraints.
- [Workload generator](docs/workload-generator.md): reproducible workload generation.
- [Modal T4 lab operator guide](docs/modal-t4-lab.md): pinned one-T4 validation workflow.
- [Results summary](logs/results-summary.md): scoped experiment results and caveats.
- [Learning log](logs/practice.md) and [roadmap](plan.md): detailed history and remaining learning work.

## Optional GPU and multi-worker experiments

GPU, Modal, and multi-GPU commands can consume paid resources. Do not run them without explicit approval for the specific workload, hardware, and expected cost; check current provider pricing and budget first. The repo's CPU tests do not validate GPU memory, hardware performance, or co-resident placement. The static multi-GPU configurations contain placeholder device UUIDs and require operator review and real device UUIDs before use. Their memory-utilization settings are not physical memory partitions; placement, recovery, and failover are limited and operator-controlled. Follow the [operator guide](docs/modal-t4-lab.md) and [routing guide](docs/request-routing.md), not historical measurements in isolation.
