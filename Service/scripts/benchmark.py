"""
Benchmark script for measuring single-request and batch latency, throughput (RPS),
and concurrent RTB performance.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from fastapi.testclient import TestClient

from src.main import app


SAMPLE_IMPRESSION = {
    "auction_id": "auc_bench_1",
    "ts": "2024-04-13 12:00:00",
    "exchange_id": "14",
    "dimension_id": "250",
    "rtb_url": "https://news.yahoo.co.jp/articles/sample",
    "ab_fold": "a",
    "ias_viewability": 65.0,
    "google_ctr": 0.087,
    "google_viewability": 77.0,
    "bid_value": 78.0,
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
    "ip": "hashed_ip_bench",
    "device_type": "1",
    "biz": "home",
    "connection_type": "4g",
    "country": "jp",
    "city": "Tokyo",
    "latitude": 35.69,
    "longitude": 139.63,
    "postal_code": "100-0001",
}


def benchmark_single_requests(client, n_requests: int = 500) -> Dict[str, float]:
    print(f"\n[1/3] Benchmarking {n_requests} sequential single RTB requests...")

    # Warmup
    for _ in range(20):
        client.post("/api/v1/predict", json=SAMPLE_IMPRESSION)

    latencies = []
    t_start = time.perf_counter()
    for i in range(n_requests):
        payload = {**SAMPLE_IMPRESSION, "auction_id": f"auc_seq_{i}"}
        t0 = time.perf_counter()
        resp = client.post("/api/v1/predict", json=payload)
        t1 = time.perf_counter()
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
        latencies.append((t1 - t0) * 1000.0)

    t_total = time.perf_counter() - t_start
    latencies.sort()

    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean_lat = statistics.mean(latencies)
    rps = n_requests / t_total

    print(f"  Completed in: {t_total:.2f} s")
    print(f"  Throughput:   {rps:.1f} req/s")
    print(f"  Latency p50:  {p50:.2f} ms")
    print(f"  Latency p90:  {p90:.2f} ms")
    print(f"  Latency p95:  {p95:.2f} ms")
    print(f"  Latency p99:  {p99:.2f} ms")
    print(f"  Latency mean: {mean_lat:.2f} ms")

    return {
        "n_requests": n_requests,
        "rps": round(rps, 1),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean_lat, 2),
    }


def benchmark_batches(
    client, batch_sizes: List[int] = [10, 50, 100, 250]
) -> List[Dict[str, Any]]:
    print("\n[2/3] Benchmarking batch inference throughput...")
    results = []

    for size in batch_sizes:
        batch_items = [
            {**SAMPLE_IMPRESSION, "auction_id": f"auc_b_{i}"} for i in range(size)
        ]
        payload = {"items": batch_items}

        # 5 runs
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp = client.post("/api/v1/predict/batch", json=payload)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            runs.append((t1 - t0) * 1000.0)

        avg_lat = statistics.mean(runs)
        items_per_sec = size / (avg_lat / 1000.0)
        per_item_ms = avg_lat / size

        print(
            f"  Batch size {size:3d}: total {avg_lat:6.2f} ms | per-item {per_item_ms:5.2f} ms | {items_per_sec:7.1f} impressions/sec"
        )
        results.append(
            {
                "batch_size": size,
                "total_ms": round(avg_lat, 2),
                "per_item_ms": round(per_item_ms, 3),
                "throughput_ips": round(items_per_sec, 1),
            }
        )

    return results


def benchmark_concurrency(
    client_or_url: str, n_workers: int = 8, n_requests_per_worker: int = 50
):
    print(
        f"\n[3/3] Benchmarking concurrent RTB bidding ({n_workers} concurrent workers)..."
    )

    def worker_task(worker_id: int):
        with TestClient(app) as test_c:
            w_latencies = []
            for i in range(n_requests_per_worker):
                p = {**SAMPLE_IMPRESSION, "auction_id": f"w_{worker_id}_{i}"}
                t0 = time.perf_counter()
                r = test_c.post("/api/v1/predict", json=p)
                t1 = time.perf_counter()
                if r.status_code == 200:
                    w_latencies.append((t1 - t0) * 1000.0)
            return w_latencies

    t0 = time.perf_counter()
    all_latencies = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_task, w) for w in range(n_workers)]
        for f in as_completed(futures):
            all_latencies.extend(f.result())
    t1 = time.perf_counter()

    total_reqs = len(all_latencies)
    total_time = t1 - t0
    rps = total_reqs / total_time
    all_latencies.sort()

    p50 = statistics.median(all_latencies)
    p95 = all_latencies[int(len(all_latencies) * 0.95)]
    p99 = all_latencies[int(len(all_latencies) * 0.99)]

    print(f"  Completed {total_reqs} requests in {total_time:.2f} s")
    print(f"  Concurrent Throughput: {rps:.1f} req/s")
    print(f"  Concurrent p50:        {p50:.2f} ms")
    print(f"  Concurrent p95:        {p95:.2f} ms")
    print(f"  Concurrent p99:        {p99:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CTR inference service.")
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="Optional base URL of remote/running service (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--requests", type=int, default=500, help="Number of sequential requests"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CTR RTB INFERENCE SERVICE BENCHMARK")
    print("=" * 60)

    if args.url:
        print(f"Testing remote service at: {args.url}")
        with httpx.Client(base_url=args.url, timeout=30.0) as client:
            benchmark_single_requests(client, n_requests=args.requests)
            benchmark_batches(client)
    else:
        print("Testing via in-process ASGI TestClient (zero network overhead)...")
        with TestClient(app) as client:
            benchmark_single_requests(client, n_requests=args.requests)
            benchmark_batches(client)
            benchmark_concurrency(client)

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
