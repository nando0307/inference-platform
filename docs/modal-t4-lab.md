# One-T4 Modal lab

The lab uses the digest-pinned `vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3` image (vLLM 0.33.0) and one stable Modal Volume v2 named `inference-huggingface-cache`, mounted at `/opt/huggingface`. On an empty or invalid cache, the approved T4 execution downloads and verifies both pinned Hugging Face snapshots, commits the manifest, then switches back to offline mode before diagnostics or inference. Later executions reuse a verified cache without downloading or committing. Imports and image builds do not download models or run remote work.

Modal 1.6.3 could not detect Python in the pinned image before the Function ran. The image recipe uses `setup_dockerfile_commands` to expose the existing `python3` as `python` without replacing the image's Python packages; the next T4 diagnostic will confirm the interpreter, pip, and torch/vLLM imports. One T4 validation is authorized but deferred until after PR merge; no hardware result exists yet, so both the root cause and the alias's effectiveness remain unverified. `add_python="3.12"` could provide Modal a detectable interpreter, but adds another Python environment whose packages may differ from the pinned vLLM/PyTorch/CUDA stack. The alias avoids that compatibility risk, pending the T4 result.

## Populate and verify the cache on T4

From the repository root, with the intended Modal workspace selected and authenticated, run a diagnostic only after explicit spending approval:

```bash
modal run modal_t4_lab.py --mode diagnostic
```

The T4 first validates the mounted Volume v2 manifest and snapshot files locally. If missing or invalid, it obtains each complete revision file list from the Hub, downloads both exact revisions into the Hub cache under `/opt/huggingface`, and requires exact snapshot path equality, required config/tokenizer metadata and model weight files, and symlink targets inside the mounted Volume. After fully verifying both snapshots, it writes their verified path lists to `/opt/huggingface/verified-snapshots.json` and commits the Volume before any diagnostic or inference. A valid cache skips network downloads and commits and stays offline. Any download or verification error stops before inference. The existing T4 function timeout is unchanged; whether both snapshots fit within that limit is unverified, and no timeout or resource limit is increased to accommodate downloads.

Pinned snapshots:

- `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`: `/opt/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`: `/opt/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

Provisioning and verification run on the T4 and require network access to Hugging Face; downloads incur GPU/CPU time, cache storage, and potentially network transfer charges. Do not run any paid T4 command until the workspace, budget, spending limits, and explicit per-command approval are confirmed.

## Preflight and first paid tests

Install Modal CLI 1.6.3 or compatible, select the intended profile, and authenticate using `modal setup` if needed. Then run:

```bash
scripts/modal-t4-lab.sh preflight
modal profile current
modal token info >/dev/null 2>&1 && echo 'Modal authentication accepted (details suppressed).'
```

Review credits, billing configuration, workspace usage budget and out-of-pocket spend limit in Modal [Usage & Billing](https://modal.com/settings/usage) and the official [budget guide](https://modal.com/docs/guide/budgets). Authentication does not establish the budget or spend-limit configuration. Run the diagnostic above only after explicit spending approval; it populates the cache on T4 only when missing or invalid.

After the diagnostic succeeds, run each single-model test one at a time and confirm cleanup between runs:

```bash
modal run modal_t4_lab.py --mode single-small
modal container list
modal run modal_t4_lab.py --mode single-large
modal container list
```

Each function requests one T4, four physical CPU cores, 8192 MiB memory, max one container, no retries, a 958-second function timeout and a 2-second idle scaledown. The runner deadline is 900 seconds and worker startup limit is 360 seconds. Worker servers bind only to `127.0.0.1`; no public endpoint, tunnel, external metrics, or token output is produced. Cleanup sends SIGTERM, then SIGKILL to the process group if needed. After a run, wait at least 2 seconds, list containers/apps and confirm the lab container stopped. Stop only the exact app ID created by this run using `scripts/modal-t4-lab.sh stop ap-EXACT_ID` if cleanup is needed; verify again with `modal container list` and `modal app list`.

## Charges and limits

Official rates and allowances can change; check Modal's [pricing](https://modal.com/pricing), [Volume guide](https://modal.com/docs/guide/volumes), [networking guide](https://modal.com/docs/guide/networking), and workspace billing before use. Current published pricing includes 10 GiB of Volume storage and 10 GiB of network egress per workspace per month; usage above included allowances is billed at the applicable published rate. Deleted Volume data may take up to 7 days to be purged, so deletion does not necessarily stop storage billing immediately. Actual cache transfer and storage totals depend on model sizes and workspace usage. Estimates are not spend caps.

The official pricing lists T4 at $0.000164/GPU-second, CPU at $0.0000131 per physical core-second, and memory at $0.00000222 per GiB-second. If the full 958-second function timeout elapses at all requested resources, the conditional estimate is:

- GPU: 958 × $0.000164 = $0.157112.
- CPU: 958 × 4 × $0.0000131 = $0.0501992.
- RAM: 958 × 8 × $0.00000222 = $0.01701408.
- Total: **$0.22432528** for the full timeout at requested resources.

Modal bills CPU and memory by the higher of request or actual usage, so actual CPU/RAM use can increase this estimate. It excludes Volume storage, cache download/egress, image transfer/build, taxes, plan charges, credits/discounts and pricing or region changes. The timeout bounds duration, not dollars; this is not a spend cap.