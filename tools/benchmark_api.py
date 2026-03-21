import argparse
import json
import time
import urllib.request


def fetch(url: str):
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=15) as response:
        body = response.read().decode("utf-8")
        status = response.status
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
    args = parser.parse_args()

    endpoints = ["/health", "/model-info"]
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
            "sample_payload": json.loads(payload_example) if payload_example else None,
        }

    print(
        json.dumps(
            {
                "base_url": args.base_url,
                "iterations": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
