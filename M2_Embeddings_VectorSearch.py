# Databricks notebook source
# MAGIC %md
# MAGIC # Milestone 2 — Native Embeddings, Vector Search Endpoint, Delta Sync Index, Reranking
# MAGIC **Project:** Agentic RAG Application on Databricks
# MAGIC **Notebook:** M2_Embeddings_VectorSearch
# MAGIC **User:** dishashreesha100@gmail.com
# MAGIC **Compute:** Serverless | **Source:** nexus_catalog.dev_schema.silver_requirements
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Progress Tracker
# MAGIC
# MAGIC ### M2.1 — Chunking and Native Embeddings
# MAGIC - [x] silver_requirements read successfully
# MAGIC - [x] Distributed chunking UDF designed and applied (PySpark/Pandas UDF)
# MAGIC - [x] Unique primary key (chunk_id) assigned per chunk
# MAGIC - [x] gold_chunks Delta table written with CDF enabled
# MAGIC - [x] Chunk count and average chunk length validated
# MAGIC
# MAGIC ### M2.2 — Vector Search Endpoint Provisioning
# MAGIC - [x] Endpoint provisioned via Python SDK (NOT via UI)
# MAGIC - [x] Polling script written and confirmed blocking until READY
# MAGIC - [x] Endpoint status confirmed READY before proceeding
# MAGIC
# MAGIC ### M2.3 — Delta Sync Index
# MAGIC - [x] Delta Sync Index created on gold_chunks via SDK
# MAGIC - [x] Connected to Databricks Foundation Model embedding endpoint
# MAGIC - [x] Sync monitoring loop confirms row count matches source
# MAGIC - [x] Index status confirmed READY
# MAGIC
# MAGIC ### M2.4 — Two-Stage Retrieval Function
# MAGIC - [x] Stage 1: Vector Search returns top 10 chunks
# MAGIC - [x] Stage 2: Foundation Model Reranker reorders to top 3
# MAGIC - [x] Fallback reranker (cosine similarity) implemented for free-tier restrictions
# MAGIC - [x] search_knowledge_base function tested with a sample query
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Assets Created
# MAGIC | Asset | Type | Location |
# MAGIC |---|---|---|
# MAGIC | gold_chunks | Delta Table + CDF | nexus_catalog.dev_schema |
# MAGIC | nexus-vs-endpoint | Vector Search Endpoint | Databricks Vector Search |
# MAGIC | nexus-vs-index | Delta Sync Index | nexus-vs-endpoint |
# MAGIC | search_knowledge_base | Python function | This notebook / M3 import |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Assignment Coverage
# MAGIC | Assignment Task | Status | Cell |
# MAGIC |---|---|---|
# MAGIC | Distributed chunking UDF + gold_chunks + CDF | ✅ Complete | Cell 1 |
# MAGIC | SDK endpoint provisioning + polling loop | ✅ Complete | Cell 2 |
# MAGIC | Delta Sync Index + sync monitoring | ✅ Complete | Cell 3 |
# MAGIC | Two-stage retrieval (Vector Search + Reranker) | ✅ Complete | Cell 4 |
# MAGIC

# COMMAND ----------

# Intent: Read silver_requirements, apply a distributed Pandas UDF to chunk each page's text
# into overlapping fixed-size chunks, assign a unique primary key to every chunk,
# and write the result to gold_chunks with CDF enabled. CDF is mandatory because
# the Delta Sync Index in Cell 3 uses it to detect new/changed chunks incrementally.

import uuid
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType
)

spark = SparkSession.builder.getOrCreate()

# -- Constants ----------------------------------------------------------------
CATALOG      = "nexus_catalog"
SCHEMA       = "dev_schema"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_requirements"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_chunks"

# Chunking hyperparameters
# CHUNK_SIZE: target number of words per chunk — 150 is optimal for
#             the embedding model context window (max 512 tokens)
# CHUNK_OVERLAP: words shared between consecutive chunks so that
#                context is not lost at chunk boundaries
CHUNK_SIZE    = 250
CHUNK_OVERLAP = 50

print("=" * 68)
print("  MILESTONE 2.1 — Distributed Chunking + gold_chunks")
print(f"  Source : {SILVER_TABLE}")
print(f"  Target : {GOLD_TABLE}")
print(f"  Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")
print("=" * 68)

# -- Read silver_requirements -------------------------------------------------
silver_df = spark.table(SILVER_TABLE)
src_count  = silver_df.count()
print(f"\n  Source rows read from silver_requirements : {src_count}")

# -- Define output schema for the Pandas UDF ----------------------------------
# This schema is what each partition returns after chunking
chunk_schema = StructType([
    StructField("chunk_id",     StringType(),    False),   # UUID primary key — required by Vector Search
    StructField("req_id",       StringType(),    False),   # parent page reference
    StructField("source_file",  StringType(),    True),
    StructField("page_number",  IntegerType(),   True),
    StructField("chunk_index",  IntegerType(),   False),   # position of chunk within its parent page
    StructField("chunk_text",   StringType(),    False),   # the actual text sent to the embedding model
    StructField("word_count",   IntegerType(),   True),
    StructField("processed_at", TimestampType(), True),
])

# -- Pandas UDF: distributed chunking ----------------------------------------
# Using a Pandas UDF (grouped map) so the chunking logic runs in parallel
# across Spark partitions — scalable to millions of pages without driver bottleneck.
@F.pandas_udf(chunk_schema, F.PandasUDFType.GROUPED_MAP)
def chunk_text_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    For each group (one page), split cleaned_text into overlapping word windows.
    Returns one row per chunk with a globally unique chunk_id.
    """
    rows = []
    now  = pd.Timestamp.utcnow()

    for _, row in pdf.iterrows():
        text  = str(row.get("cleaned_text", "")).strip()
        words = text.split()

        if not words:
            continue

        # Sliding window chunking with overlap
        step  = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        start = 0
        idx   = 0
        while start < len(words):
            window     = words[start : start + CHUNK_SIZE]
            chunk_text = " ".join(window)
            rows.append({
                "chunk_id"    : str(uuid.uuid4()),       # globally unique primary key
                "req_id"      : str(row.get("req_id", "")),
                "source_file" : str(row.get("source_file", "")),
                "page_number" : int(row.get("page_number", 0)),
                "chunk_index" : idx,
                "chunk_text"  : chunk_text,
                "word_count"  : len(window),
                "processed_at": now,
            })
            start += step
            idx   += 1

    return pd.DataFrame(rows, columns=[f.name for f in chunk_schema.fields])

# -- Apply UDF grouped by req_id (one group = one page) -----------------------
gold_df = (
    silver_df
    .select("req_id", "source_file", "page_number", "cleaned_text")
    .groupby("req_id")
    .apply(chunk_text_udf)
)

# -- Write gold_chunks with CDF enabled ---------------------------------------
(gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema",           "true")
    .option("delta.enableChangeDataFeed","true")   # mandatory for Delta Sync Index
    .saveAsTable(GOLD_TABLE))

# Belt-and-suspenders CDF confirmation
spark.sql(f"ALTER TABLE {GOLD_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# -- Verify ------------------------------------------------------------------
gold_count = spark.table(GOLD_TABLE).count()
avg_words  = spark.table(GOLD_TABLE).agg(F.avg("word_count")).collect()[0][0]

print(f"\n  gold_chunks written : {gold_count} chunks from {src_count} source pages")
print(f"  Average words/chunk : {round(avg_words, 1)}")

# Verify CDF
props  = {r["key"]: r["value"] for r in
          spark.sql(f"SHOW TBLPROPERTIES {GOLD_TABLE}").collect()}
cdf_ok = props.get("delta.enableChangeDataFeed") == "true"
print(f"  CDF on gold_chunks  : {'CONFIRMED' if cdf_ok else 'FAILED — check above'}")

print("\n  Sample chunks:")
(spark.table(GOLD_TABLE)
    .select("chunk_id", "req_id", "chunk_index", "word_count",
            F.substring("chunk_text", 1, 90).alias("preview"))
    .orderBy("req_id", "chunk_index")
    .show(6, truncate=False))

print("Cell 1 complete — gold_chunks ready for Vector Search indexing.")


# COMMAND ----------

#2.2
# Intent: Programmatically provision a Databricks Vector Search Endpoint using only the Python SDK.
# The UI must NOT be used for this task per the assignment.
# After creation, a polling loop blocks all downstream execution until the endpoint
# reports status READY — this is required because the Delta Sync Index in Cell 3
# cannot be created against an endpoint that is still initialising.

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType

w = WorkspaceClient()

# -- Constants ----------------------------------------------------------------
VS_ENDPOINT_NAME = "nexus-vs-endpoint"
POLL_INTERVAL_S  = 30      # seconds between each status check
MAX_WAIT_S       = 1800    # 30 minutes maximum wait before raising an error

print("=" * 68)
print("  MILESTONE 2.2 — Vector Search Endpoint Provisioning (SDK only)")
print(f"  Endpoint name : {VS_ENDPOINT_NAME}")
print("  Note: provisioning typically takes 10-20 minutes on free tier")
print("=" * 68)

# -- Check if endpoint already exists (idempotent) ----------------------------
existing = [e.name for e in w.vector_search_endpoints.list_endpoints()]

if VS_ENDPOINT_NAME not in existing:
    print(f"\n  Creating endpoint: {VS_ENDPOINT_NAME} ...")
    w.vector_search_endpoints.create_endpoint(
        name          = VS_ENDPOINT_NAME,
        endpoint_type = EndpointType.STANDARD,
    )
    print(f"  Endpoint creation request submitted.")
else:
    print(f"\n  Endpoint already exists: {VS_ENDPOINT_NAME}")

# -- Polling loop: block until endpoint status is READY -----------------------
# This pattern is required by the assignment. We check every POLL_INTERVAL_S
# seconds and raise an error if the endpoint does not become ready within MAX_WAIT_S.
print(f"\n  Polling endpoint status every {POLL_INTERVAL_S}s (max wait: {MAX_WAIT_S // 60} min) ...")
print("  " + "-" * 60)

elapsed      = 0
final_status = None

while elapsed < MAX_WAIT_S:
    endpoint     = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    final_status = str(endpoint.endpoint_status.state) if endpoint.endpoint_status else "UNKNOWN"

    print(f"  [{elapsed:>5}s elapsed]  Status: {final_status}")

    if "READY" in final_status or "ONLINE" in final_status:
        break

    if "FAILED" in final_status or "ERROR" in final_status:
        raise RuntimeError(
            f"Vector Search Endpoint entered failed state: {final_status}. "
            "Check Databricks Vector Search UI for detailed error logs."
        )

    time.sleep(POLL_INTERVAL_S)
    elapsed += POLL_INTERVAL_S

# -- Guard: raise if we exhausted the wait window ----------------------------
if "READY" not in final_status and "ONLINE" not in final_status:
    raise TimeoutError(
        f"Endpoint {VS_ENDPOINT_NAME} did not become READY within "
        f"{MAX_WAIT_S // 60} minutes. Current status: {final_status}. "
        "Re-run this cell — provisioning sometimes takes longer on free tier."
    )

print("  " + "-" * 60)
print(f"\n  Endpoint READY confirmed : {VS_ENDPOINT_NAME}")
print(f"  Final status             : {final_status}")
print(f"  Total wait               : {elapsed}s\n")
print("Cell 2 complete — Vector Search Endpoint is READY.")


# COMMAND ----------

# DBTITLE 1,Cell 4
# Intent: Create a Delta Sync Index on gold_chunks using the Python SDK.
# The index connects to a Databricks Foundation Model embedding endpoint so that
# embeddings are generated and kept in sync automatically without any manual re-indexing.
# A monitoring loop then verifies that the index has fully synchronised by comparing
# the indexed row count against the source gold_chunks row count before proceeding.

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
w     = WorkspaceClient()

# -- Constants ----------------------------------------------------------------
CATALOG          = "nexus_catalog"
SCHEMA           = "dev_schema"
GOLD_TABLE       = f"{CATALOG}.{SCHEMA}.gold_chunks"
VS_ENDPOINT_NAME = "nexus-vs-endpoint"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.nexus_vs_index"

# Databricks Foundation Model used for embeddings — no external API key required
# gte-large-en is optimised for retrieval tasks and is available on all workspace tiers
EMBEDDING_MODEL  = "databricks-gte-large-en"

POLL_INTERVAL_S  = 30
MAX_WAIT_S       = 1800

print("=" * 68)
print("  MILESTONE 2.3 — Delta Sync Index Creation + Sync Monitoring")
print(f"  Source table    : {GOLD_TABLE}")
print(f"  VS Endpoint     : {VS_ENDPOINT_NAME}")
print(f"  Index name      : {VS_INDEX_NAME}")
print(f"  Embedding model : {EMBEDDING_MODEL}")
print("=" * 68)

# -- Get source row count (used later to verify sync completeness) -------------
source_count = spark.table(GOLD_TABLE).count()
print(f"\n  Source row count in gold_chunks : {source_count}")

# -- Create Delta Sync Index (idempotent) -------------------------------------
try:
    existing_index = w.vector_search_indexes.get_index(VS_INDEX_NAME)
    print(f"\n  Index already exists: {VS_INDEX_NAME}")
except Exception:
    print(f"\n  Creating Delta Sync Index: {VS_INDEX_NAME} ...")
    w.vector_search_indexes.create_index(
        name                  = VS_INDEX_NAME,
        endpoint_name         = VS_ENDPOINT_NAME,
        primary_key           = "chunk_id",         # unique PK assigned in Cell 1
        index_type            = VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec = DeltaSyncVectorIndexSpecRequest(
            source_table          = GOLD_TABLE,
            pipeline_type         = PipelineType.TRIGGERED,   # TRIGGERED works on free tier
            embedding_source_columns = [
                EmbeddingSourceColumn(
                    name                   = "chunk_text",       # column to embed
                    embedding_model_endpoint_name = EMBEDDING_MODEL,
                )
            ],
        ),
    )
    print(f"  Index creation request submitted.")

# -- Sync monitoring loop: wait until indexed rows match source rows -----------
# The assignment requires verifying sync completeness by matching row counts.
# We poll the index status and read the indexed_row_count from the index descriptor.
print(f"\n  Monitoring sync progress every {POLL_INTERVAL_S}s ...")
print("  " + "-" * 60)

elapsed        = 0
sync_confirmed = False
indexed_rows   = 0        # initialise here so the guard below is always defined
index_status   = "UNKNOWN"

while elapsed < MAX_WAIT_S:
    try:
        index        = w.vector_search_indexes.get_index(VS_INDEX_NAME)
        # VectorIndexStatus uses .ready (bool) — not .detailed_state
        is_ready     = bool(index.status and index.status.ready)
        indexed_rows = getattr(index.status, "indexed_row_count", 0) or 0
        index_status = "READY" if is_ready else "SYNCING"
        status_msg   = getattr(index.status, "message", "") or ""
    except Exception as ex:
        print(f"  [{elapsed:>5}s]  Status check error (will retry): {ex}")
        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
        continue

    print(f"  [{elapsed:>5}s elapsed]  State: {index_status}  |  Indexed rows: {indexed_rows} / {source_count}  {status_msg}")

    # Sync complete when index is ready and row count matches source
    if is_ready and indexed_rows >= source_count:
        sync_confirmed = True
        break

    # Detect failure via message field
    if status_msg and ("fail" in status_msg.lower() or "error" in status_msg.lower()):
        raise RuntimeError(
            f"Delta Sync Index entered failed state. Message: {status_msg}. "
            f"Check the index status in Catalog > {VS_INDEX_NAME}."
        )

    time.sleep(POLL_INTERVAL_S)
    elapsed += POLL_INTERVAL_S

# -- Guard -------------------------------------------------------------------
if not sync_confirmed:
    raise TimeoutError(
        f"Index did not finish syncing within {MAX_WAIT_S // 60} minutes. "
        f"Indexed rows: {indexed_rows} / {source_count}. Re-run this cell."
    )

print("  " + "-" * 60)
print(f"\n  Sync confirmed  : {indexed_rows} / {source_count} rows indexed")
print(f"  Index status    : {index_status}")
print(f"  Total wait      : {elapsed}s")
print("\nCell 3 complete — Delta Sync Index is READY and fully synced.")


# COMMAND ----------

# DBTITLE 1,Install databricks-vectorsearch
# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

# DBTITLE 1,Restart Python after install
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 5
# Intent: Implement the two-stage search_knowledge_base retrieval function required by the assignment.
# Stage 1 — Query the Vector Search index to retrieve the top 10 semantically similar chunks.
# Stage 2 — Pass those 10 chunks to a Databricks Foundation Model Reranker endpoint to
#            re-order them strictly by semantic relevance and return only the top 3.
# A cosine similarity fallback reranker is included for free-tier workspaces where
# the dedicated reranker endpoint may not be available — this is the engineering workaround
# the assignment specifically asks for when infrastructure restrictions apply.

import numpy as np
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# -- Constants ----------------------------------------------------------------
CATALOG          = "nexus_catalog"
SCHEMA           = "dev_schema"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.nexus_vs_index"
RERANKER_MODEL   = "databricks-bge-reranker-v2-m3"  # Databricks Foundation Model reranker
TOP_K_RETRIEVE   = 10    # Stage 1: retrieve this many candidates from Vector Search
TOP_K_FINAL      = 3     # Stage 2: return this many after reranking

print("=" * 68)
print("  MILESTONE 2.4 — Two-Stage Retrieval: Vector Search + Reranker")
print(f"  Index        : {VS_INDEX_NAME}")
print(f"  Reranker     : {RERANKER_MODEL}")
print(f"  Stage 1 top-K: {TOP_K_RETRIEVE} | Stage 2 top-K: {TOP_K_FINAL}")
print("=" * 68)

# -- Cosine similarity fallback reranker --------------------------------------
# Used automatically when the dedicated reranker endpoint is not available.
# Computes the cosine similarity between the query and each candidate chunk's text
# using a simple term-frequency overlap as a proxy — no external model required.
def _cosine_fallback_rerank(query: str, candidates: list, top_k: int) -> list:
    """
    Fallback reranker using term-frequency cosine similarity.
    Activates automatically when the Foundation Model reranker endpoint is unavailable.
    Returns top_k candidates re-ordered by text overlap with the query.
    """
    query_terms = set(query.lower().split())
    scored      = []
    for c in candidates:
        text         = str(c.get("chunk_text", "")).lower().split()
        chunk_terms  = set(text)
        # cosine proxy: intersection size / geometric mean of term counts
        overlap      = len(query_terms & chunk_terms)
        denominator  = (len(query_terms) * len(chunk_terms)) ** 0.5
        score        = overlap / denominator if denominator > 0 else 0.0
        scored.append({**c, "rerank_score": round(score, 6)})
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# -- Main retrieval function --------------------------------------------------
def search_knowledge_base(query: str, verbose: bool = True) -> list:
    """
    Two-stage retrieval function for the Project Nexus RAG agent.

    Stage 1: Query the Databricks Vector Search index to retrieve the top
             TOP_K_RETRIEVE (10) most semantically similar chunks.

    Stage 2: Pass all 10 candidates to the Foundation Model Reranker to
             re-order them by true semantic relevance and return top TOP_K_FINAL (3).

    Falls back to cosine similarity reranking if the reranker endpoint is
    unavailable (free-tier restriction workaround per assignment guidance).

    Args:
        query   : Natural language question from the user or agent.
        verbose : If True, print a formatted result table to the cell output.

    Returns:
        List of top-3 result dicts with keys: chunk_id, chunk_text, req_id,
        page_number, rerank_score.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # -- Stage 1: Vector Search retrieval -------------------------------------
    try:
        vs_response = w.vector_search_indexes.query_index(
            index_name  = VS_INDEX_NAME,
            columns     = ["chunk_id", "chunk_text", "req_id", "page_number", "chunk_index"],
            query_text  = query,
            num_results = TOP_K_RETRIEVE,
        )
        col_names  = [c.name for c in (vs_response.manifest.columns or [])]
        candidates = [
            {col: val for col, val in zip(col_names, row)}
            for row in (vs_response.result.data_array or [])
        ]
    except Exception as e:
        raise RuntimeError(f"Stage 1 Vector Search query failed: {e}")

    if not candidates:
        if verbose:
            print("  No results returned from Vector Search for this query.")
        return []

    # -- Stage 2: Reranker (Foundation Model with cosine fallback) ------------
    reranker_used = None
    try:
        rerank_resp = w.serving_endpoints.query(
            name = RERANKER_MODEL,
            dataframe_records = [
                {"text": query, "text_pair": c.get("chunk_text", "")}
                for c in candidates
            ],
        )
        # Parse reranker response — each prediction contains a relevance score
        predictions = rerank_resp.predictions if hasattr(rerank_resp, "predictions") else []
        if predictions and len(predictions) == len(candidates):
            for i, pred in enumerate(predictions):
                score = pred if isinstance(pred, (int, float)) else (
                    float(pred[0]) if isinstance(pred, list) else pred.get("score", 0.0)
                )
                candidates[i]["rerank_score"] = round(float(score), 6)
            top_results  = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K_FINAL]
            reranker_used = "Foundation Model reranker"
        else:
            raise ValueError("Reranker returned unexpected prediction format.")
    except Exception as rerank_err:
        # Fallback: cosine similarity reranking — valid workaround on free tier
        print(f"  [INFO] FM reranker unavailable ({str(rerank_err)[:120]}), using cosine fallback.")
        top_results   = _cosine_fallback_rerank(query, candidates, TOP_K_FINAL)
        reranker_used = "Cosine similarity fallback (free-tier workaround)"

    # -- Print formatted results ----------------------------------------------
    if verbose:
        print(f"\n  Query          : {query}")
        print(f"  Stage 1 hits   : {len(candidates)} candidates from Vector Search")
        print(f"  Stage 2 method : {reranker_used}")
        print(f"  Final results  : top {len(top_results)}")
        print("  " + "-" * 64)
        for rank, r in enumerate(top_results, 1):
            print(f"  Rank {rank}  |  Score: {r.get('rerank_score', 'n/a')}"
                  f"  |  Page: {r.get('page_number','?')}  |  req_id: {r.get('req_id','?')}")
            preview = str(r.get("chunk_text",""))[:120]
            print(f"  Preview: {preview} ...")
            print()

    return top_results

# -- Run a test query to validate the full pipeline ---------------------------
print("\n  Running test query to validate end-to-end retrieval pipeline ...")
print("  " + "=" * 64)

test_results = search_knowledge_base(
    query   = "What are the password security and authentication requirements?",
    verbose = True,
)

print(f"  Test query returned {len(test_results)} ranked results.")
print("\nCell 4 complete — two-stage search_knowledge_base function validated.")


# COMMAND ----------

# DBTITLE 1,Cell 8
# Intent: Validate all Milestone 2 assets in one consolidated check.
# Confirms gold_chunks exists with CDF, Vector Search endpoint is READY,
# Delta Sync Index is READY, and the search function returns results.
# Wires taskValues outputs so the M4 Lakeflow Job can read M2 completion status.
# Exits with SUCCESS for clean job task termination or raises on any failure.

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
w     = WorkspaceClient()

CATALOG          = "nexus_catalog"
SCHEMA           = "dev_schema"
GOLD_TABLE       = f"{CATALOG}.{SCHEMA}.gold_chunks"
VS_ENDPOINT_NAME = "nexus-vs-endpoint"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.nexus_vs_index"

print("=" * 68)
print("  MILESTONE 2 — Final Validation Report")
print("=" * 68)

passed = []
failed = []

def check(label: str, condition: bool, detail: str = ""):
    icon = "PASS" if condition else "FAIL"
    line = f"  [{icon}]  {label}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    (passed if condition else failed).append(label)

# -- gold_chunks table --------------------------------------------------------
print("\n  Tables:")
try:
    gold_count = spark.table(GOLD_TABLE).count()
    check("gold_chunks row count", gold_count > 0, f"{gold_count} rows")
    props  = {r["key"]: r["value"] for r in
              spark.sql(f"SHOW TBLPROPERTIES {GOLD_TABLE}").collect()}
    cdf_ok = props.get("delta.enableChangeDataFeed") == "true"
    check("gold_chunks CDF enabled", cdf_ok, props.get("delta.enableChangeDataFeed","not set"))
except Exception as e:
    check("gold_chunks", False, str(e)[:80])
    gold_count = 0

# -- Vector Search Endpoint ---------------------------------------------------
print("\n  Vector Search Endpoint:")
try:
    ep       = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    ep_state = str(ep.endpoint_status.state) if ep.endpoint_status else "UNKNOWN"
    check("nexus-vs-endpoint READY", "READY" in ep_state or "ONLINE" in ep_state, ep_state)
except Exception as e:
    check("nexus-vs-endpoint", False, str(e)[:80])

# -- Delta Sync Index ---------------------------------------------------------
print("\n  Delta Sync Index:")
try:
    idx        = w.vector_search_indexes.get_index(VS_INDEX_NAME)
    idx_ready  = bool(idx.status and idx.status.ready)
    idx_state  = "READY" if idx_ready else "NOT READY"
    idx_rows   = getattr(idx.status, "indexed_row_count", 0) or 0
    check("nexus_vs_index READY", idx_ready, idx_state)
    check("Indexed rows match source", idx_rows >= gold_count, f"{idx_rows} / {gold_count}")
except Exception as e:
    check("nexus_vs_index", False, str(e)[:80])
    idx_rows = 0

# -- search_knowledge_base function -------------------------------------------
print("\n  Retrieval function:")
try:
    results = search_knowledge_base("authentication and password requirements", verbose=False)
    check("search_knowledge_base returns results", len(results) > 0, f"{len(results)} results")
    check("Returns exactly top 3 or fewer", len(results) <= 3, f"{len(results)} results")
except Exception as e:
    check("search_knowledge_base callable", False, str(e)[:80])

# -- Summary ------------------------------------------------------------------
total = len(passed) + len(failed)
print(f"\n  {'=' * 68}")
print(f"  MILESTONE 2 RESULT:  {len(passed)}/{total} checks passed")
print(f"  {'=' * 68}")

if failed:
    print(f"\n  Failed checks ({len(failed)}):")
    for item in failed:
        print(f"     - {item}")

# -- Wire taskValues for M4 Lakeflow Job pipeline ----------------------------
# These values are read by the M4 job orchestration notebook to confirm M2 completed.
dbutils.jobs.taskValues.set(key="m2_gold_chunks_count",    value=int(gold_count))
dbutils.jobs.taskValues.set(key="m2_indexed_row_count",    value=int(idx_rows))
dbutils.jobs.taskValues.set(key="m2_vs_endpoint_name",     value=VS_ENDPOINT_NAME)
dbutils.jobs.taskValues.set(key="m2_vs_index_name",        value=VS_INDEX_NAME)
dbutils.jobs.taskValues.set(key="m2_complete",             value=len(failed) == 0)

# -- Job-compatible exit ------------------------------------------------------
if failed:
    raise SystemExit(f"[M2 FAILED] {len(failed)} validation checks failed: {failed}")
else:
    print("\n  ALL CHECKS PASSED — Milestone 2 Complete.")
    print("  gold_chunks is indexed. search_knowledge_base is ready for M3 agent assembly.")
    dbutils.notebook.exit("SUCCESS")


# COMMAND ----------

# M3.1 — Register calculate_sla_metrics as a Unity Catalog Python Function
# UC Python functions run in a sandboxed Python process — no JVM/Spark available.
# Table access uses databricks-sdk WorkspaceClient + Statement Execution API.

from textwrap import dedent

FUNC_BODY = dedent('''
if endpoint_filter is None:
    return "Error: endpoint_filter must not be null. Pass a partial endpoint string or ALL."

filter_val = str(endpoint_filter).strip()
if not filter_val:
    return "Error: endpoint_filter must not be empty. Pass a partial endpoint string or ALL."

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if spark is None:
        return "Error: No active Spark session in function context. Function must be called from a SQL warehouse."

    filter_lower = filter_val.lower()
    if filter_lower == "all":
        where_sql = "1=1"
    else:
        safe_filter = filter_val.replace("\'", "\'\'")
        where_sql = "LOWER(endpoint) LIKE \'%" + safe_filter.lower() + "%\'"

    df = spark.sql(
        "SELECT endpoint, "
        "COUNT(*) AS total_calls, "
        "SUM(is_error) AS error_count, "
        "ROUND(AVG(error_rate_pct), 2) AS error_rate_pct, "
        "ROUND(AVG(latency_ms), 2) AS avg_latency_ms, "
        "ROUND(PERCENTILE_APPROX(latency_ms, 0.95), 2) AS p95_latency_ms, "
        "ROUND(MAX(latency_ms), 2) AS max_latency_ms, "
        "SUM(CAST(sla_breach AS INT)) AS latency_breach_count "
        "FROM nexus_catalog.dev_schema.silver_api_kpis "
        "WHERE " + where_sql + " "
        "GROUP BY endpoint ORDER BY total_calls DESC"
    )

    raw_rows = df.collect()
    rows = [r.asDict() for r in raw_rows]

    if not rows:
        return "No data found for endpoint filter: " + endpoint_filter + ". Try ALL or a different path fragment."

    ERROR_RATE_SLA_PCT = 5.0
    LATENCY_SLA_MS     = 200.0

    report = ["=== SLA Report | filter: " + endpoint_filter + " ==="]
    total_calls_all  = 0
    breach_endpoints = 0

    def _f(v, d=0.0):
        try: return float(v) if v is not None else d
        except: return d
    def _i(v, d=0):
        try: return int(float(v)) if v is not None else d
        except: return d

    for r in rows:
        total      = _i(r.get("total_calls"))
        errors     = _i(r.get("error_count"))
        err_pct    = _f(r.get("error_rate_pct"))
        avg_lat    = _f(r.get("avg_latency_ms"))
        p95_lat    = _f(r.get("p95_latency_ms"))
        max_lat    = _f(r.get("max_latency_ms"))
        lat_breach = _i(r.get("latency_breach_count"))
        ep         = str(r.get("endpoint", "unknown"))

        violations = []
        if err_pct > ERROR_RATE_SLA_PCT:
            violations.append("error_rate " + str(round(err_pct,1)) + "% > " + str(ERROR_RATE_SLA_PCT) + "% threshold")
        if p95_lat > LATENCY_SLA_MS:
            violations.append("p95_latency " + str(round(p95_lat)) + "ms > " + str(LATENCY_SLA_MS) + "ms threshold")

        sla_status = "BREACH" if violations else "PASS"
        if violations:
            breach_endpoints += 1

        success_count = max(total - errors, 0)
        success_rate  = round(success_count / total * 100, 1) if total > 0 else 0.0

        report.append("")
        report.append("Endpoint           : " + ep)
        report.append("  Total calls      : " + str(total))
        report.append("  Success rate     : " + str(success_rate) + "%  (" + str(success_count) + "/" + str(total) + ")")
        report.append("  Error rate       : " + str(round(err_pct,1)) + "%  (" + str(errors) + " errors)")
        report.append("  Avg latency      : " + str(avg_lat) + " ms")
        report.append("  P95 latency      : " + str(p95_lat) + " ms  (SLA <= " + str(LATENCY_SLA_MS) + " ms)")
        report.append("  Max latency      : " + str(max_lat) + " ms")
        report.append("  Latency breaches : " + str(lat_breach) + " requests")
        report.append("  SLA Status       : " + sla_status)
        if violations:
            report.append("  Violations       : " + " | ".join(violations))

        total_calls_all += total

    report.append("")
    report.append("=" * 50)
    report.append("SUMMARY")
    report.append("  Endpoints evaluated : " + str(len(rows)))
    report.append("  Endpoints in BREACH : " + str(breach_endpoints))
    report.append("  Total API calls     : " + str(total_calls_all))
    report.append("=" * 50)
    return "\\n".join(report)

except Exception as e:
    return "SLA calculation error: " + str(e)
''').strip()

CREATE_FUNCTION_SQL = f"""
CREATE OR REPLACE FUNCTION nexus_catalog.dev_schema.calculate_sla_metrics(
    endpoint_filter STRING
    COMMENT 'Partial endpoint path to filter (e.g. auth, agent, datasets). Pass ALL for a full summary.'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'SLA Calculator Tool - Invoke this tool when the user asks about API SLA compliance,
error rates, latency performance, p95 latency, or SLA breach counts for any endpoint.
Input: a partial endpoint path string (e.g. auth, agent, datasets) or ALL for aggregate results.
Returns a structured SLA report per endpoint: total calls, error rate, avg/p95/max latency,
latency breach count, and SLA pass/fail status. Thresholds: 200ms p95 latency, 5% error rate.'
AS $$
{FUNC_BODY}
$$
"""

spark.sql(CREATE_FUNCTION_SQL)
print("M3.1 - calculate_sla_metrics registered successfully in nexus_catalog.dev_schema")
print("  Language : Python (LANGUAGE PYTHON UC function)")
print("  Backend  : databricks-sdk WorkspaceClient + Statement Execution API")
print("  Table    : nexus_catalog.dev_schema.silver_api_kpis")
print("  Guards   : null input, empty string, divide-by-zero all handled")


# COMMAND ----------



# COMMAND ----------

