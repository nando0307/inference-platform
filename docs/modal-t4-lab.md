# One-T4 Modal lab

This lab is an opt-in, one-function experiment. It does not deploy a public API, open a tunnel, build a model-download layer, or modify the A-D serving configurations. The pinned image is `vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3` (vLLM 0.33.0). The upstream vLLM image does not include model snapshots; model modes require the Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` and Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` to already be present in `/opt/huggingface/hub`.
The repository `requirements.txt` is the local application stack, not the serving image's dependency lock. This lab uses the existing digest-pinned vLLM image and does not install or update dependencies.

## Before launch

Install Modal CLI 1.6.3 or compatible, select the intended profile, and authenticate using `modal setup` if needed. From the repository root, run these local-only checks:

```bash
scripts/modal-t4-lab.sh preflight
modal profile current
modal token info >/dev/null 2>&1 && echo 'Modal authentication accepted (details suppressed).'
```

The script suppresses token details. The model cache check is local and does not download weights. The lab container is strictly offline and fails if its pinned snapshots are absent from `/opt/huggingface/hub`. Do not launch until the required model snapshot is present in the pinned image's cache; this lab intentionally has no build-time or runtime model download.

Check remaining credits, billing configuration, workspace usage budget, and out-of-pocket spend limit in the Modal dashboard's [Usage & Billing](https://modal.com/settings/usage) page; the official [budget guide](https://modal.com/docs/guide/budgets) explains that usage budgets apply before credits, while spend limits cap net charges after credits. Verify the intended workspace, current-cycle credit balance, usage limit, spend limit, and payment method yourself before launch. Modal CLI authentication does not reveal these values, and this task did not inspect them.

## Launch and inspect

The first launch runs diagnostics only; it still allocates a T4. The exact one-T4 launch command is:

```bash
modal run modal_t4_lab.py --mode diagnostic
```

The shell wrapper offers the same launch and validates the mode before calling Modal:

```bash
scripts/modal-t4-lab.sh launch diagnostic
```

After the diagnostic passes, later validation modes are `single-small`, `single-large`, and `shared`. Each invocation requests exactly `gpu="T4"`, one container, no retries, 4 physical CPU cores, and 8192 MiB RAM. Modal enforces a 958-second Function timeout and a 2-second idle scaledown window, bounding the function plus GPU container lifetime to 16 minutes. The runner uses a 900-second deadline to reserve cleanup time; a forced Function timeout is the final bound if normal cleanup fails.

The remote diagnostics report `nvidia-smi -L` GPU UUID mapping, CUDA availability/version and device capability/name/memory, vLLM version compatibility, and offline cache presence. Model runs check `/v1/models` identity, non-streaming and streaming chat completion, and GPU memory. Worker HTTP servers bind only to `127.0.0.1`; no endpoints or metrics are exposed externally. The worker process groups receive SIGTERM and are waited on, then SIGKILL only if needed.

List running and recently stopped apps/containers without changing them:

```bash
modal app list
modal container list
```

Do not stop an app by name. If an invocation remains active, identify its exact app ID in the CLI output and verify its identity before stopping only that ID. Inspect an app with:

```bash
modal app info ap-EXACT_ID
```

Then stop it with:

```bash
scripts/modal-t4-lab.sh stop ap-EXACT_ID
```

Cleanup verification: wait at least 2 seconds after the function returns or stop the exact app, then run `modal container list` and `modal app list`; verify no running container belongs to this experiment. The Modal dashboard's Apps/Containers view is the second check. Do not stop unrelated apps. No persistent volume is created. Normal function completion releases the ephemeral GPU; CLI interruption alone is not proof of release, so inspect the lists.

## Stage 1: one model at a time

Do not run these commands during environment preparation. After the diagnostic passes and snapshots are available, exercise each model in a separate invocation:

```bash
modal run modal_t4_lab.py --mode single-small
modal container list
modal run modal_t4_lab.py --mode single-large
modal container list
```

Each mode checks the served model identity, readiness, one basic non-streaming request, one streaming request, GPU memory, and cleanup. Capture the printed JSON and CLI output. Confirm no lab container remains before the next invocation.

## Stage 2: two models sharing one T4

Only after Stage 1 passes, use:

```bash
modal run modal_t4_lab.py --mode shared
modal container list
```

This starts both pinned model processes concurrently, each with `CUDA_VISIBLE_DEVICES=0`, 0.5 GPU memory utilization, and `max-num-seqs=2`. It verifies each process's model identity and basic streaming/non-streaming request, records GPU memory, and shuts down both process groups. Confirm the function completes within its 16-minute ceiling and verify no lab container remains. Do not use `T4:2`.

## Cost estimate

Modal's [official pricing](https://modal.com/pricing) lists T4 at $0.000164/GPU-second, CPU at $0.0000131 per physical core-second, and memory at $0.00000222 per GiB-second. At the configured 4 cores, 8 GiB, one T4, and 960 seconds, the estimate at requested resources is:

- GPU: 960 × $0.000164 = $0.15744.
- CPU: 960 × 4 × $0.0000131 = $0.050304.
- RAM: 960 × 8 × $0.00000222 = $0.0170496.
- Total: **$0.2247936 (about $0.225) per full 16-minute invocation at requested resources**.

Modal bills CPU and memory by the higher of request or actual usage, so this is not a hard dollar cap; actual CPU/RAM use can increase the amount. The function timeout bounds duration, not spend. These estimates exclude image transfer/build costs, model download or storage (none is done by this lab), network egress, taxes, plan charges, discounts/credits, and any pricing or region changes. Credits may cover compute charges but do not reduce the listed resource rate. A workspace spend limit was not configured or verified.
