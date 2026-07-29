# Disagg-DFlash (branch `sd_disagg`)

Remote DFlash draft on a dedicated GPU; verify keeps target only. HS via **NIXL**; token ids on the control plane. Wave scheduling manufactures decode overlap.

## Paper-style run matrix (fair benches, profile OFF)

Same client every time. Change only the server launch. Between configs: stop previous servers (`pkill -f 'vllm serve|dflash_draft_server'` or Ctrl-C the launcher), wait until GPUs are free, then start the next.

```bash
cd /home/shiladitya/vllm_2307/vllm/specdec_benchmarking
# Shared client knobs (tune rates to your machine; keep identical across rows)
RATES=32,64,128
DUR=30
```

| # | What it shows | Launch | Bench tag |
|---|---------------|--------|-----------|
| 0 | **Base** — target only, no SD | `PD1_b8192` on GPU0 | `paper_base_PD1_b8192` |
| 1 | **Colocated SD** — draft steals SMs on same GPU | `PD1SD1_b8192` on GPU0 | `paper_coloc_PD1SD1_b8192` |
| 2 | **Disagg sync** — isolation only (pay full Tad on critical path) | `V1SD1` + `--no-disagg-async --no-wave-schedule` | `paper_disagg_sync_V1SD1_b8192` |
| 3 | **Disagg async, no wave** — park all; little decode-only overlap | `V1SD1` + `--no-wave-schedule` | `paper_disagg_async_nowave_V1SD1_b8192` |
| 4 | **Disagg async + wave (default)** — primary claim | `V1SD1` defaults | `paper_disagg_async_wave_V1SD1_b8192` |
| 5 | **Wave size fixed 50** — sensitivity | `V1SD1` + `--wave-size 50` | `paper_disagg_wave50_V1SD1_b8192` |

Expected story: **0 → 1** (SD helps but contention); **1 → 2** (separate draft GPU helps some); **2 → 3** (small); **3 → 4** (decode-heavy overlap win); **5** confirms wave knob.

### 0 — Base (no SD)

```bash
./launch_bench_server.sh PD1_b8192 --devices 0 --port 8000
# other terminal:
./benchmark_random.sh --tag paper_base_PD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR"
```

### 1 — Colocated DFlash

```bash
./launch_bench_server.sh PD1SD1_b8192 --devices 0 --port 8000
./benchmark_random.sh --tag paper_coloc_PD1SD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR"
```

### 2 — Disagg sync (ablation)

```bash
./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000 \
  --no-disagg-async --no-wave-schedule
./benchmark_random.sh --tag paper_disagg_sync_V1SD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR" \
  --draft-metrics-url http://localhost:9101
```

### 3 — Disagg async, no wave

```bash
./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000 \
  --no-wave-schedule
./benchmark_random.sh --tag paper_disagg_async_nowave_V1SD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR" \
  --draft-metrics-url http://localhost:9101
```

### 4 — Disagg async + wave (main result)

```bash
./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000
./benchmark_random.sh --tag paper_disagg_async_wave_V1SD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR" \
  --draft-metrics-url http://localhost:9101
```

### 5 — Wave size = 50

```bash
./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000 \
  --wave-size 50
./benchmark_random.sh --tag paper_disagg_wave50_V1SD1_b8192 --port 8000 \
  --request-rates "$RATES" --duration-sec "$DUR" \
  --draft-metrics-url http://localhost:9101
```

### Optional — overlap diagnostics (NOT for the TPOT table)

```bash
./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000 \
  --wave-size 50 --enable-disagg-profile
./benchmark_random.sh --tag paper_disagg_profile_V1SD1_b8192 --port 8000 \
  --request-rates 64 --duration-sec 20 \
  --draft-metrics-url http://localhost:9101
# grep startup_logs for [DisaggDFlash][timing] / [wave] / [profile]
```

Results land in `bench_results/<tag>/r<RATE>/`. Report request throughput, output tok/s, mean TPOT / TTFT vs `#` above.

### Fairness notes

- Rows **0–1** use **1 GPU**; rows **2–5** use **2 GPUs** (verify + draft). That matches the GreenLLM “extra draft GPU” story — say so in the paper.
- Optional iso-2-GPU colocated control: `PD2SD1_b8192 --devices 0,1` → tag `paper_coloc_PD2SD1_b8192`.
- Keep rates / duration / model lengths identical across rows.
- **V\*SD\*** and **PD\*SD\*** both use `rejection_sample_method=synthetic` with the same `SYNTH_RATES` in `launch_bench_server.sh` so acceptance length is controlled and comparable.

## Profiling vs benchmarks

- **Default: timing/profile off.** Opt-in with `--enable-disagg-profile` only for overlap debug (`slack_ms`, `await_ms`, `nixl_await_ms`). Those paths CUDA-sync and skew latency.
- **NIXL transfer logs (cheap):** `--nixl-log-every 0` (every xfer) or `--nixl-log-every N`. Wall-clock + NIXL telemetry on verify; draft logs nbytes. No extra CUDA sync for logging.
- NIXL WRITE is posted in `speculate_begin` and joined in `speculate_wait` so HS DMA can overlap verify post-sample; ZMQ meta is sent only after NIXL DONE.
- Draft token ids: TP0 only talks to the draft server. Real ids go `DraftTokenIds` → scheduler `request.spec_token_ids` → next `SchedulerOutput.scheduled_spec_decode_tokens` → **all TP ranks hydrate** `req_states.draft_tokens` (no draft-token NCCL broadcast).
