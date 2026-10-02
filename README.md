## Vector Database Benchmark

### Objective

Load the same embedded corpus from day 36 into three vector stores and compare their ingestion time, query latency, retrieval quality, and external service cost.

**Vector stores tested:**
- FAISS
- Chroma
- Qdrant

### Benchmark Setup

To keep the comparison consistent:

- Same corpus used across all three stores
- Same sentence-based chunking strategy
- Same `all-MiniLM-L6-v2` embedding model
- Embeddings generated once and reused across all stores
- Same retrieval questions
- `top_k = 3`
- Ingestion timing excludes embedding generation
- Query latency measures vector search only
- All stores were run locally

### Results

| Vector Store | Chunks | Ingestion Time (s) | Avg Query Latency (s) | Recall@3 | Cost |
|---|---:|---:|---:|---:|---|
| FAISS | 2 | 0.000125 | 0.029993 | 1.00 | $0 external |
| Chroma | 2 | 0.717938 | 0.019502 | 1.00 | $0 external |
| Qdrant | 2 | 0.175511 | 0.077038 | 1.00 | $0 external |

### What the Results Show

All three vector stores achieved **1.00 Recall@3** on the five evaluation queries used in this experiment.

The measured ingestion and query times differed across the three implementations. These measurements represent this specific local environment and dataset rather than general performance characteristics of each vector store.

Because the benchmark corpus produced only two chunks, the latency results are primarily useful for validating the benchmarking workflow. A larger corpus would be required to draw meaningful conclusions about performance at scale.

### Cost

FAISS, Chroma, and Qdrant were all evaluated using local deployments, so the experiment incurred **$0 in external vector-database service charges**.

This does not imply zero infrastructure cost; local CPU, memory, storage, and electricity costs were not estimated.

### Architecture

```text
                    Source Corpus
                         │
                         ▼
                    Chunking
                         │
                         ▼
                Sentence Embeddings
                all-MiniLM-L6-v2
                         │
                         ▼
              Same Embedded Chunks
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        FAISS          Chroma         Qdrant
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Similarity Search
                         │
                         ▼
                 Recall@3 Evaluation
```
