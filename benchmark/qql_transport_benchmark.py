from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from qql import AsyncConnection, Connection


URL = os.getenv("QQL_BENCH_URL", "http://localhost:6333")
SECRET = os.getenv("QQL_BENCH_SECRET") or None
ITERATIONS = int(os.getenv("QQL_BENCH_ITERATIONS", "50"))
WARMUP = int(os.getenv("QQL_BENCH_WARMUP", "5"))
CONCURRENCY = int(os.getenv("QQL_BENCH_CONCURRENCY", "10"))

DOCS = [
    "Qdrant stores vectors and payloads for semantic search workloads",
    "FastEmbed generates local dense embeddings for short text queries",
    "gRPC can reduce transport overhead for high volume vector database calls",
    "REST remains simple and reliable for operational database workflows",
    "Async clients help Python applications keep network requests in flight",
    "Local embedding models can dominate latency before the database is called",
    "Hybrid search combines dense vectors with sparse lexical retrieval",
    "Payload filters narrow search results by metadata fields and values",
    "Collection topology determines named dense and sparse vector behavior",
    "Benchmark results should separate setup cost from measured query latency",
]
QUERY_TEXT = "local embedding vector database transport benchmark"


@dataclass(frozen=True)
class Result:
    mode: str
    total_ms: float
    avg_ms: float
    qps: float


def search_query(collection: str) -> str:
    return f"SEARCH {collection} SIMILAR TO '{QUERY_TEXT}' LIMIT 5"


def insert_query(collection: str, idx: int, text: str) -> str:
    return (
        f"INSERT INTO COLLECTION {collection} "
        f"VALUES {{'id': {idx}, 'text': '{text}', 'kind': 'bench'}}"
    )


def ignore_drop(conn: Connection, collection: str) -> None:
    try:
        conn.run_query(f"DROP COLLECTION {collection}")
    except Exception:
        pass


async def ignore_drop_async(conn: AsyncConnection, collection: str) -> None:
    try:
        await conn.run_query(f"DROP COLLECTION {collection}")
    except Exception:
        pass


def setup_sync(collection: str, *, prefer_grpc: bool) -> None:
    with Connection(URL, secret=SECRET, prefer_grpc=prefer_grpc) as conn:
        ignore_drop(conn, collection)
        for idx, text in enumerate(DOCS, start=1):
            conn.run_query(insert_query(collection, idx, text))


async def setup_async(collection: str, *, prefer_grpc: bool) -> None:
    async with AsyncConnection(URL, secret=SECRET, prefer_grpc=prefer_grpc) as conn:
        await ignore_drop_async(conn, collection)
        for idx, text in enumerate(DOCS, start=1):
            await conn.run_query(insert_query(collection, idx, text))


def bench_sync(mode: str, collection: str, *, prefer_grpc: bool) -> Result:
    setup_sync(collection, prefer_grpc=prefer_grpc)
    query = search_query(collection)
    with Connection(URL, secret=SECRET, prefer_grpc=prefer_grpc) as conn:
        for _ in range(WARMUP):
            conn.run_query(query)
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            conn.run_query(query)
        total_ms = (time.perf_counter() - start) * 1000
    return Result(mode, total_ms, total_ms / ITERATIONS, ITERATIONS / (total_ms / 1000))


async def bench_async(mode: str, collection: str, *, prefer_grpc: bool) -> Result:
    await setup_async(collection, prefer_grpc=prefer_grpc)
    query = search_query(collection)
    async with AsyncConnection(URL, secret=SECRET, prefer_grpc=prefer_grpc) as conn:
        for _ in range(WARMUP):
            await conn.run_query(query)
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            await conn.run_query(query)
        total_ms = (time.perf_counter() - start) * 1000
    return Result(mode, total_ms, total_ms / ITERATIONS, ITERATIONS / (total_ms / 1000))


async def bench_async_concurrent(
    mode: str,
    collection: str,
    *,
    prefer_grpc: bool,
) -> Result:
    query = search_query(collection)
    async with AsyncConnection(URL, secret=SECRET, prefer_grpc=prefer_grpc) as conn:
        for _ in range(WARMUP):
            await conn.run_query(query)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one() -> None:
            async with sem:
                await conn.run_query(query)

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(ITERATIONS)))
        total_ms = (time.perf_counter() - start) * 1000
    return Result(mode, total_ms, total_ms / ITERATIONS, ITERATIONS / (total_ms / 1000))


def print_table(title: str, results: list[Result]) -> None:
    print(f"\n### {title}\n")
    print("| Mode | Total ms | Avg ms/op | Ops/sec |")
    print("|---|---:|---:|---:|")
    for r in results:
        print(f"| {r.mode} | {r.total_ms:,.2f} | {r.avg_ms:,.2f} | {r.qps:,.2f} |")


async def main() -> None:
    print("QQL SEARCH benchmark")
    print(f"URL: {URL}")
    print(f"Workload: {ITERATIONS} measured SEARCH queries, {WARMUP} warmup")
    print("Embedding: local FastEmbed dense model, warmed before timing")

    latency = [
        bench_sync("sync REST", "qql_bench_sync_rest", prefer_grpc=False),
        await bench_async("async REST", "qql_bench_async_rest", prefer_grpc=False),
        bench_sync("sync gRPC", "qql_bench_sync_grpc", prefer_grpc=True),
        await bench_async("async gRPC", "qql_bench_async_grpc", prefer_grpc=True),
    ]
    print_table("Single-flight latency", latency)

    concurrent = [
        await bench_async_concurrent(
            f"async REST x{CONCURRENCY}",
            "qql_bench_async_rest",
            prefer_grpc=False,
        ),
        await bench_async_concurrent(
            f"async gRPC x{CONCURRENCY}",
            "qql_bench_async_grpc",
            prefer_grpc=True,
        ),
    ]
    print_table("Async concurrent throughput", concurrent)


if __name__ == "__main__":
    asyncio.run(main())
