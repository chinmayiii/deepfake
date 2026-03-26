# Resume Proof Pack

Use this checklist to verify project claims in under 3 minutes.

## 1) API probes and service health

```bash
curl http://127.0.0.1:7860/live
curl http://127.0.0.1:7860/ready
curl http://127.0.0.1:7860/health
```

## 2) Operational visibility

```bash
curl http://127.0.0.1:7860/config
curl http://127.0.0.1:7860/metrics
curl http://127.0.0.1:7860/capabilities
```

## 3) Evaluation evidence

```bash
curl http://127.0.0.1:7860/eval/summary
```

Generate/update evaluation summary artifact:

```bash
python realeval.py --data path/to/dataset_root --out docs/eval_summary.json
```

Note: if `num_samples` is `0`, the evaluation file is still a placeholder and should be regenerated on a real held-out dataset.

## 4) Performance benchmark evidence

```bash
python tools/benchmark_api.py --base-url http://127.0.0.1:7860 --iterations 20 --output docs/perf_summary.json
```

Use `docs/perf_summary.json` as the source of latency/throughput claims.

## 5) Automated test proof

```bash
run_smoke_tests.bat
```

or

```powershell
./run_smoke_tests.ps1
```

## 6) What this demonstrates

- Production API design with readiness/liveness probes
- Defensive controls (upload limits, rate limiting, request timeouts)
- Runtime observability and structured metrics
- Reproducible quality reporting via evaluation artifact
- CI-validatable endpoint contracts via smoke tests
