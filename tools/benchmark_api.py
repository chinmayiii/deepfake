import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


def fetch(url: str):
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8") if exc.fp else ""
    elapsed_ms = (time.perf_counter() - start) * 1000
    return status, elapsed_ms, body


def main():
    parser = argparse.ArgumentParser(description="Simple API latency benchmark")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:7860", help="API base URL"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of requests per endpoint"
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=[
            "/live",
            "/ready",
            "/health",
            "/model-info",
            "/config",
            "/capabilities",
            "/eval/summary",
            "/metrics",
        ],
        help="Endpoints to benchmark",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write benchmark JSON report",
    )
    args = parser.parse_args()

    endpoints = args.endpoints
    results = {}

    for endpoint in endpoints:
        latencies = []
        status_codes = []
        payload_example = None
        for _ in range(args.iterations):
            status, elapsed_ms, body = fetch(f"{args.base_url}{endpoint}")
            status_codes.append(status)
            latencies.append(elapsed_ms)
            payload_example = body

        results[endpoint] = {
            "status_codes": sorted(set(status_codes)),
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "p95_ms": round(
                sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2
            ),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "rps_estimate": round(
                (1000.0 / (sum(latencies) / len(latencies)))
                if latencies and (sum(latencies) / len(latencies)) > 0
                else 0.0,
                2,
            ),
            "sample_payload": json.loads(payload_example) if payload_example else None,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "iterations": args.iterations,
        "endpoints": endpoints,
        "results": results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
