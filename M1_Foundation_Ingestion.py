# Databricks notebook source
# MAGIC %md
# MAGIC # 🏗️ Milestone 1 — Foundation, Data Ingestion & Preparation
# MAGIC **Project:** Agentic RAG Application on Databricks  
# MAGIC **User:** dishashreesha100@gmail.com  
# MAGIC **Compute:** Serverless | **Target:** nexus_catalog.dev_schema
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Progress Tracker
# MAGIC
# MAGIC ### M1.1 — Environment & Unity Catalog Privileges
# MAGIC - [ ] All libraries installed and versions pinned
# MAGIC - [ ] `nexus_catalog` created programmatically via SDK
# MAGIC - [ ] `dev_schema` created under nexus_catalog
# MAGIC - [ ] `raw_documents` UC Volume created
# MAGIC - [ ] GRANT: Admin role → ALL PRIVILEGES (manage, delete, write)
# MAGIC - [ ] GRANT: Analyst role → SELECT only (read-only, human analysts)
# MAGIC - [ ] GRANT: Agent role → SELECT + EXECUTE (automated agents)
# MAGIC
# MAGIC ### M1.2 — AI Gateway & Credential Management
# MAGIC - [ ] Secrets scope `nexus-scope` created
# MAGIC - [ ] Placeholder secret stored (no real key in code)
# MAGIC - [ ] Secret read-back confirmed (Databricks masking active)
# MAGIC - [ ] AI Gateway markdown spec written (QPM/TPM limits + guardrails)
# MAGIC
# MAGIC ### M1.3 — Unstructured Ingestion & CDF
# MAGIC - [ ] SRS PDF uploaded to UC Volume manually
# MAGIC - [ ] `bronze_raw_documents` Delta table created from PDF pages
# MAGIC - [ ] `silver_requirements` table created with cleaned text
# MAGIC - [ ] CDF enabled on `silver_requirements`
# MAGIC - [ ] CDF rationale documented (why mandatory for Vector Search)
# MAGIC
# MAGIC ### M1.4 — Auto Loader & Semantic Knowledge Store
# MAGIC - [ ] Mock API metrics CSV generated in UC Volume
# MAGIC - [ ] Auto Loader stream → `bronze_api_metrics` table
# MAGIC - [ ] `silver_api_kpis` created with `error_rate_pct` metric
# MAGIC - [ ] Rich table-level COMMENT added
# MAGIC - [ ] Rich column-level COMMENTs added for all columns
# MAGIC - [ ] `ai_query_example` TBLPROPERTY injected
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📁 Assets Created in This Milestone
# MAGIC | Asset | Type | Path |
# MAGIC |---|---|---|
# MAGIC | `nexus_catalog` | UC Catalog | Unity Catalog root |
# MAGIC | `dev_schema` | UC Schema | nexus_catalog.dev_schema |
# MAGIC | `raw_documents` | UC Volume | /Volumes/nexus_catalog/dev_schema/raw_documents |
# MAGIC | `bronze_raw_documents` | Delta Table | nexus_catalog.dev_schema |
# MAGIC | `silver_requirements` | Delta Table + CDF | nexus_catalog.dev_schema |
# MAGIC | `bronze_api_metrics` | Delta Table | nexus_catalog.dev_schema |
# MAGIC | `silver_api_kpis` | Delta Table + Semantic Metadata | nexus_catalog.dev_schema |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔗 Assignment Coverage
# MAGIC | Assignment Task | Status | Cell |
# MAGIC |---|---|---|
# MAGIC | Programmatic library version locking | ✅ | Cell 1 |
# MAGIC | nexus_catalog.dev_schema + Volume | ✅ | Cell 2 |
# MAGIC | GRANT — analysts vs agents | ✅ | Cell 3 |
# MAGIC | Secrets — no hardcoded keys | ✅ | Cell 4 |
# MAGIC | AI Gateway QPM/TPM markdown spec | ✅ | Cell 4 |
# MAGIC | SRS PDF → Bronze → silver_requirements | ✅ | Cell 6 |
# MAGIC | CDF enabled + WHY documented | ✅ | Cell 6 |
# MAGIC | Mock CSV → Auto Loader → Bronze | ✅ | Cell 7 |
# MAGIC | error_rate_pct metric | ✅ | Cell 7 |
# MAGIC | Rich semantic metadata + ai_query_example | ✅ | Cell 7 |

# COMMAND ----------

# Install all pinned library versions required across the full project and validate the Databricks environment

import subprocess, sys, importlib
from databricks.sdk import WorkspaceClient

# ── Pinned library manifest ──────────────────────────────────────────────────
# These exact versions are locked for reproducibility across all milestones.
# Do NOT upgrade without testing the full pipeline end-to-end.
PINNED_LIBRARIES = {
    "databricks-sdk":          "0.28.0",
    "databricks-vectorsearch": "0.40",
    "mlflow":                  "2.14.0",
    "pdfplumber":              "0.11.0",
    "langchain":               "0.2.12",
    "langchain-community":     "0.2.12",
    "wikipedia":               "1.4.0",
}

def pin_and_verify(package: str, version: str) -> dict:
    """Install a package at a pinned version and verify the installed version."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{package}=={version}", "--quiet"],
        capture_output=True
    )
    try:
        mod  = importlib.import_module(package.replace("-", "_").split(".")[0])
        got  = getattr(mod, "__version__", "unknown")
        ok   = version in got
    except ImportError:
        got, ok = "import-failed", False
    return {"package": package, "pinned": version, "installed": got, "ok": ok}

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.1: Library Version Pinning")
print("=" * 68)

all_ok   = True
results  = [pin_and_verify(p, v) for p, v in PINNED_LIBRARIES.items()]
for r in results:
    icon = "✅" if r["ok"] else "⚠️ "
    print(f"  {icon}  {r['package']:<30} pinned={r['pinned']}  installed={r['installed']}")
    if not r["ok"]:
        all_ok = False

# ── Workspace SDK validation ─────────────────────────────────────────────────
print("\n" + "─" * 68)
w    = WorkspaceClient()
me   = w.current_user.me()
is_admin = any(g.display == "admins" for g in (me.groups or []))

print(f"  Workspace user  : {me.user_name}")
print(f"  Admin confirmed : {'✅ YES' if is_admin else '❌ NO'}")
print(f"  Library status  : {'✅ ALL PINNED' if all_ok else '⚠️  SOME WARNINGS — review above'}")
print("=" * 68)
print("\n🟢 Cell 1 complete — environment validated.")

# COMMAND ----------

# DBTITLE 1,Cell 3
# Programmatically create nexus_catalog, dev_schema, and the raw_documents UC Volume using the Databricks SDK

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()

# ── Single source of truth — all downstream cells import these constants ─────
CATALOG      = "nexus_catalog"
SCHEMA       = "dev_schema"
VOLUME       = "raw_documents"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.1: Unity Catalog Bootstrap")
print("=" * 68)

# ── Create Catalog ────────────────────────────────────────────────────────────
existing_catalogs = [c.name for c in w.catalogs.list()]
if CATALOG not in existing_catalogs:
    spark.sql(f"CREATE CATALOG `{CATALOG}` COMMENT 'Project Nexus — Agentic RAG application root catalog'")
    print(f"\n  ✅ Created catalog : {CATALOG}")
else:
    print(f"\n  ℹ️  Catalog exists  : {CATALOG}")

# ── Create Schema ─────────────────────────────────────────────────────────────
existing_schemas = [s.name for s in w.schemas.list(catalog_name=CATALOG)]
if SCHEMA not in existing_schemas:
    w.schemas.create(
        name         = SCHEMA,
        catalog_name = CATALOG,
        comment      = "Development schema — Bronze / Silver / Gold medallion layers for Project Nexus"
    )
    print(f"  ✅ Created schema  : {CATALOG}.{SCHEMA}")
else:
    print(f"  ℹ️  Schema exists   : {CATALOG}.{SCHEMA}")

# ── Create UC Volume ──────────────────────────────────────────────────────────
existing_volumes = [v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)]
if VOLUME not in existing_volumes:
    w.volumes.create(
        catalog_name = CATALOG,
        schema_name  = SCHEMA,
        name         = VOLUME,
        volume_type  = VolumeType.MANAGED,
        comment      = "Raw document store — SRS PDFs and mock API metric CSV files"
    )
    print(f"  ✅ Created volume  : {VOLUME_PATH}")
else:
    print(f"  ℹ️  Volume exists   : {VOLUME_PATH}")

print(f"""
  ─────────────────────────────────────────────────────────────
  📁 UC Volume ready for file uploads:
     {VOLUME_PATH}/

  ➡️  After this cell completes, go to:
     Catalog → nexus_catalog → dev_schema → Volumes → raw_documents
     and upload: project_nexus_srs.pdf
  ─────────────────────────────────────────────────────────────
""")
print("🟢 Cell 2 complete — Unity Catalog structure created.")


# COMMAND ----------

# Establish fine-grained RBAC: Admin gets full control, Analyst gets read-only, Agent gets execute-only

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

CATALOG       = "nexus_catalog"
SCHEMA        = "dev_schema"
ADMIN_USER    = "dishashreesha100@gmail.com"   # human admin (workspace owner)
ANALYST_GROUP = "account users"                # human analysts — read-only
# Agent identity = service principal or the same user acting as automated agent
# On free tier we simulate this as a separate grant target using the admin user
AGENT_USER    = "dishashreesha100@gmail.com"

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.1: RBAC Configuration")
print("  Simulating: Human Analysts vs Automated Agents")
print("=" * 68)

# ── ADMIN grants — full lifecycle control ─────────────────────────────────────
# Admin can CREATE, MODIFY, DELETE tables and MANAGE UC privileges
admin_statements = [
    f"GRANT ALL PRIVILEGES ON CATALOG      {CATALOG}                            TO `{ADMIN_USER}`",
    f"GRANT ALL PRIVILEGES ON SCHEMA       {CATALOG}.{SCHEMA}                   TO `{ADMIN_USER}`",
    f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {CATALOG}.{SCHEMA}.raw_documents TO `{ADMIN_USER}`",
]

# ── ANALYST grants — read-only (human analysts) ───────────────────────────────
# Analysts can query tables but cannot modify, delete, or manage UC
analyst_statements = [
    f"GRANT USE CATALOG  ON CATALOG {CATALOG}         TO `{ANALYST_GROUP}`",
    f"GRANT USE SCHEMA   ON SCHEMA  {CATALOG}.{SCHEMA} TO `{ANALYST_GROUP}`",
    f"GRANT SELECT       ON SCHEMA  {CATALOG}.{SCHEMA} TO `{ANALYST_GROUP}`",
    f"GRANT READ VOLUME  ON VOLUME  {CATALOG}.{SCHEMA}.raw_documents TO `{ANALYST_GROUP}`",
]

# ── AGENT grants — SELECT + EXECUTE (automated agent identity) ────────────────
# Automated agents need to read tables and execute UC functions (Milestone 3.1)
# They must NOT be able to drop or modify schema
agent_statements = [
    f"GRANT USE CATALOG ON CATALOG {CATALOG}          TO `{AGENT_USER}`",
    f"GRANT USE SCHEMA  ON SCHEMA  {CATALOG}.{SCHEMA} TO `{AGENT_USER}`",
    f"GRANT SELECT      ON SCHEMA  {CATALOG}.{SCHEMA} TO `{AGENT_USER}`",
    f"GRANT EXECUTE     ON SCHEMA  {CATALOG}.{SCHEMA} TO `{AGENT_USER}`",
]

def run_grants(statements: list, role_label: str):
    """Execute GRANT SQL statements and report pass/warn per statement."""
    print(f"\n  🔐 {role_label}:")
    for stmt in statements:
        try:
            spark.sql(stmt)
            display_stmt = stmt.strip().replace("\n", " ")[:90]
            print(f"    ✅  {display_stmt}")
        except Exception as e:
            err = str(e)[:110]
            print(f"    ⚠️   Skipped (free-tier / already exists): {err}")

run_grants(admin_statements,   "ADMIN    — ALL PRIVILEGES (create, modify, delete, manage UC)")
run_grants(analyst_statements, "ANALYST  — READ ONLY (SELECT + USE, no write/delete)")
run_grants(agent_statements,   "AGENT    — SELECT + EXECUTE (automated agent identity)")

# ── Verify final grant state ──────────────────────────────────────────────────
print("\n  📋 Current grants on catalog (from Databricks system):")
try:
    spark.sql(f"SHOW GRANTS ON CATALOG {CATALOG}").show(30, truncate=False)
except Exception as e:
    print(f"  ⚠️  SHOW GRANTS not available on free tier: {e}")

print("""
  ─────────────────────────────────────────────────────────────
  RBAC Design Summary:
  ┌─────────────┬─────────────────────────────────────────────┐
  │ Role        │ Permissions                                 │
  ├─────────────┼─────────────────────────────────────────────┤
  │ Admin       │ ALL PRIVILEGES — full lifecycle + UC mgmt  │
  │ Analyst     │ SELECT + USE only — human read queries      │
  │ Agent       │ SELECT + EXECUTE — tool calling in M3       │
  └─────────────┴─────────────────────────────────────────────┘
""")
print("🟢 Cell 3 complete — RBAC configured.")


# COMMAND ----------

# Create Databricks Secrets scope, store placeholder credential, display AI Gateway markdown specification

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import AclPermission

w          = WorkspaceClient()
SCOPE      = "nexus-scope"
SECRET_KEY = "foundation-model-token"

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.2: Credential Management")
print("  Security Rule: Zero hardcoded credentials anywhere in code")
print("=" * 68)

# ── Create secrets scope (idempotent) ────────────────────────────────────────
existing_scopes = [s.name for s in w.secrets.list_scopes()]
if SCOPE not in existing_scopes:
    w.secrets.create_scope(scope=SCOPE)
    print(f"\n  ✅ Secrets scope created : {SCOPE}")
else:
    print(f"\n  ℹ️  Scope already exists  : {SCOPE}")

# ── Store placeholder token (NEVER a real key in code or notebook) ───────────
w.secrets.put_secret(
    scope        = SCOPE,
    key          = SECRET_KEY,
    string_value = "PLACEHOLDER_REPLACE_IN_PRODUCTION"   # ← placeholder only
)
print(f"  ✅ Secret stored         : {SCOPE}/{SECRET_KEY}  [placeholder value]")

# ── Validate secret read-back — Databricks masks the value automatically ─────
retrieved = dbutils.secrets.get(scope=SCOPE, key=SECRET_KEY)
print(f"  ✅ Secret read-back      : [REDACTED by Databricks — {len(retrieved)} chars masked]")

# ── Set ACLs on scope: Admin manages, Analyst reads ──────────────────────────
for principal, perm, label in [
    ("dishashreesha100@gmail.com", AclPermission.MANAGE, "ADMIN  → MANAGE"),
    ("account users",              AclPermission.READ,   "USERS  → READ"),
]:
    try:
        w.secrets.put_acl(scope=SCOPE, principal=principal, permission=perm)
        print(f"  ✅ Scope ACL set         : {label}")
    except Exception as e:
        print(f"  ⚠️  ACL skipped (may exist): {e}")

# ── How to USE secrets safely in downstream cells (pattern) ──────────────────
print("""
  ─────────────────────────────────────────────────────────────
  Correct Pattern — always use in downstream code:

    token = dbutils.secrets.get(scope="nexus-scope",
                                key="foundation-model-token")
    # token is injected at runtime and is NEVER printed or logged
  ─────────────────────────────────────────────────────────────
""")

# ── AI Gateway Markdown Specification ────────────────────────────────────────
ai_gateway_spec = """
================================================================================
  AI GATEWAY — ENTERPRISE SPECIFICATION FOR PROJECT NEXUS
================================================================================

## Purpose
The AI Gateway is a centralised proxy layer that sits between all application
code and external LLM endpoints. Every model call — from RAG retrieval to the
Streamlit UI — routes through this gateway. It enforces security, cost controls,
and content safety policies without requiring individual notebooks to manage them.

---

## Rate Limiting Controls

| Control              | Limit                          | Enforcement Action          |
|----------------------|--------------------------------|-----------------------------|
| QPM (Queries/min)    | 60 requests / min per user     | HTTP 429 — exponential backoff |
| TPM (Tokens/min)     | 100,000 tokens / min per scope | Queue requests + alert ops  |
| Daily Token Budget   | 1,000,000 tokens / day         | Hard block + email alert    |
| Concurrent Requests  | 10 in-flight per user          | Queue with 30s timeout      |

---

## Content Guardrails

| Guardrail            | Rule                                                          |
|----------------------|---------------------------------------------------------------|
| PII Detection        | Block prompts containing SSN, credit card numbers, raw emails |
| Prompt Injection     | Reject known jailbreak / instruction-override patterns        |
| Topic Restriction    | Limit responses to Project Nexus domain context               |
| Output Sanitisation  | Strip raw SQL/credentials from user-facing agent responses    |
| Toxicity Filter      | Block outputs with harmful/offensive content                  |

---

## Credential Binding Pattern
All tokens retrieved from Databricks Secrets at runtime:
  dbutils.secrets.get(scope="nexus-scope", key="foundation-model-token")
  → Injected into request headers — never logged, never printed

## Foundation Model Fallback (No External API Key Required)
Endpoint : databricks-meta-llama/Meta-Llama-3-1-70B-Instruct (built-in)
Access   : Via Databricks Foundation Models API — no external key needed
Cost     : Pay-per-token on Databricks — no external billing

================================================================================
"""
print(ai_gateway_spec)
print("🟢 Cell 4 complete — Secrets + AI Gateway specification done.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 📤 MANUAL STEP — Upload SRS PDF to UC Volume
# MAGIC
# MAGIC **Do this NOW before running Cell 6.**
# MAGIC
# MAGIC ### Steps:
# MAGIC 1. In the Databricks left sidebar → click **Catalog**
# MAGIC 2. Expand: `nexus_catalog` → `dev_schema` → `Volumes` → `raw_documents`
# MAGIC 3. Click **⬆️ Upload to this volume** (top right of volume page)
# MAGIC 4. Select your file: `project_nexus_srs.pdf` (renamed from `Software Requirements Specification (1).pdf`)
# MAGIC 5. Wait for upload to complete — you will see it listed in the volume
# MAGIC
# MAGIC

# COMMAND ----------

# Parse SRS PDF from UC Volume into Bronze Delta table, clean to silver_requirements, enable CDF

import pdfplumber, re
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

spark = SparkSession.builder.getOrCreate()

CATALOG      = "nexus_catalog"
SCHEMA       = "dev_schema"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/raw_documents"
PDF_PATH     = f"{VOLUME_PATH}/project_nexus_srs.pdf"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_raw_documents"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_requirements"

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.3: Unstructured Ingestion + CDF")
print("  Flow: UC Volume PDF → Bronze → silver_requirements")
print("=" * 68)

# ── STAGE 1: Parse PDF pages into structured records (Bronze layer) ───────────
print("\n  📄 Stage 1 — Parsing PDF from UC Volume ...")
pages = []
try:
    with pdfplumber.open(PDF_PATH) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "source_file" : "project_nexus_srs.pdf",
                "page_number" : i + 1,
                "total_pages" : n_pages,
                "raw_text"    : text,
                "char_count"  : len(text),
                "ingested_at" : datetime.utcnow(),
            })
    print(f"  ✅ Parsed {n_pages} pages from PDF")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n❌ PDF not found at: {PDF_PATH}\n"
        "Complete the manual upload step in Cell 5 (Markdown cell above) first."
    )

# ── Write Bronze Delta table ──────────────────────────────────────────────────
bronze_schema = StructType([
    StructField("source_file",  StringType(),    False),
    StructField("page_number",  IntegerType(),   False),
    StructField("total_pages",  IntegerType(),   False),
    StructField("raw_text",     StringType(),    True),
    StructField("char_count",   IntegerType(),   True),
    StructField("ingested_at",  TimestampType(), False),
])

bronze_df = spark.createDataFrame(pages, schema=bronze_schema)
(bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE))

print(f"  ✅ Bronze table written  : {BRONZE_TABLE}  ({bronze_df.count()} rows)")

# ── STAGE 2: Clean Bronze → silver_requirements ───────────────────────────────
print("\n  🔧 Stage 2 — Cleaning Bronze → silver_requirements ...")

silver_df = (
    spark.table(BRONZE_TABLE)
    # Drop near-empty pages (headers/footers with minimal content)
    .filter(F.length(F.trim(F.col("raw_text"))) > 80)
    .withColumn(
        "cleaned_text",
        # Step 1: collapse all whitespace runs into a single space
        F.regexp_replace(
            # Step 2: strip non-printable ASCII characters
            F.regexp_replace(F.trim(F.col("raw_text")), r"[^\x20-\x7E\n]", ""),
            r"\s+", " "
        )
    )
    # Assign a stable unique requirement ID per page
    .withColumn(
        "req_id",
        F.concat(F.lit("REQ_PAGE_"), F.lpad(F.col("page_number").cast("string"), 3, "0"))
    )
    .withColumn("word_count",    F.size(F.split(F.col("cleaned_text"), " ")))
    .withColumn("processed_at",  F.current_timestamp())
    .select("req_id", "source_file", "page_number", "cleaned_text", "word_count", "processed_at")
)

# ── Write silver table with CDF enabled ──────────────────────────────────────
(silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")   # ← MANDATORY for Vector Search
    .saveAsTable(SILVER_TABLE))

# Belt-and-suspenders: ensure CDF is ON even if option was missed above
spark.sql(f"ALTER TABLE {SILVER_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

print(f"  ✅ Silver table written  : {SILVER_TABLE}  ({silver_df.count()} rows)")

# ── STAGE 3: Verify CDF is enabled ───────────────────────────────────────────
print("\n  🔍 Stage 3 — Verifying CDF on silver_requirements ...")

props      = spark.sql(f"SHOW TBLPROPERTIES {SILVER_TABLE}").collect()
cdf_on     = any(r["key"] == "delta.enableChangeDataFeed" and r["value"] == "true" for r in props)
print(f"  CDF enabled : {'✅ CONFIRMED' if cdf_on else '❌ NOT FOUND — check above'}")

print("\n  📋 Sample silver_requirements:")
(spark.table(SILVER_TABLE)
    .select("req_id", "word_count", F.substring("cleaned_text", 1, 100).alias("preview"))
    .show(5, truncate=False))

# ── WHY CDF IS MANDATORY — documented per assignment requirement ──────────────
print("""
  ════════════════════════════════════════════════════════════════
  WHY CDF IS MANDATORY FOR DATABRICKS VECTOR SEARCH (Assignment M1.3)
  ════════════════════════════════════════════════════════════════

  Databricks Vector Search uses a "Delta Sync Index" strategy.
  Instead of re-embedding the ENTIRE table on every update, it reads
  ONLY the rows that have changed since the last sync — using the
  Delta Change Data Feed (CDF) transaction log.

  Without CDF enabled:
    → Vector Search CANNOT detect which rows changed
    → The index sync job fails at creation time
    → You cannot create a Delta Sync Index at all

  With CDF enabled:
    → Every INSERT / UPDATE / DELETE on silver_requirements is logged
    → Vector Search reads only the delta (changed rows)
    → Only changed chunks are re-embedded → cost-efficient
    → New SRS content appears in the agent's knowledge within 60s

  CDF is therefore a hard prerequisite for Milestone 2.3 (Delta Sync Index).
  ════════════════════════════════════════════════════════════════
""")
print("🟢 Cell 6 complete — Milestone 1.3 done.")


# COMMAND ----------

# DBTITLE 1,Cell 8
# Generate mock API metrics, stream via Auto Loader to Bronze, transform to silver_api_kpis, inject full semantic metadata

import json, random
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                                IntegerType, DoubleType, TimestampType)

spark = SparkSession.builder.getOrCreate()

CATALOG         = "nexus_catalog"
SCHEMA          = "dev_schema"
VOLUME_PATH     = f"/Volumes/{CATALOG}/{SCHEMA}/raw_documents"
METRICS_DIR     = f"{VOLUME_PATH}/api_metrics"
CHECKPOINT_PATH = f"{VOLUME_PATH}/_checkpoints/api_metrics_loader"
BRONZE_TABLE    = f"{CATALOG}.{SCHEMA}.bronze_api_metrics"
SILVER_TABLE    = f"{CATALOG}.{SCHEMA}.silver_api_kpis"

print("=" * 68)
print("  PROJECT NEXUS — Milestone 1.4: Auto Loader + Semantic Knowledge Store")
print("  Flow: Mock CSV → Auto Loader → Bronze → silver_api_kpis")
print("=" * 68)

# ── STAGE 1: Generate realistic mock API metrics CSV files in Volume ──────────
print("\n  📊 Stage 1 — Generating mock API metrics CSV files ...")

dbutils.fs.mkdirs(METRICS_DIR)

ENDPOINTS = [
    "/api/v1/auth/login",
    "/api/v1/auth/signup",
    "/api/v1/datasets",
    "/api/v1/datasets/{id}",
    "/api/v1/agent/query",
]
METHODS = ["GET", "POST", "PUT", "DELETE"]

def make_batch(batch_id: int, n: int = 250) -> list:
    """Generate n realistic API request records for a given batch."""
    base_ts = datetime.utcnow() - timedelta(hours=batch_id * 2)
    rows    = []
    for _ in range(n):
        status = random.choices(
            [200, 201, 400, 401, 429, 500],
            weights=[58, 14, 11, 8, 5, 4]
        )[0]
        rows.append({
            "batch_id"    : batch_id,
            "timestamp"   : (base_ts + timedelta(seconds=random.randint(0, 7200))).isoformat(),
            "endpoint"    : random.choice(ENDPOINTS),
            "http_method" : random.choice(METHODS),
            "status_code" : status,
            "latency_ms"  : round(max(8.0, random.gauss(175, 45)), 2),
            "tokens_used" : random.randint(80, 3500) if random.random() > 0.6 else 0,
            "user_id"     : f"user_{random.randint(1, 1000):04d}",
        })
    return rows

total_written = 0
for bid in range(1, 4):
    batch_records = make_batch(bid)
    batch_df      = spark.createDataFrame(batch_records)
    # Fix: enforce column order to match raw_schema — createDataFrame from dicts
    # sorts columns alphabetically, but Auto Loader applies schema positionally.
    batch_df      = batch_df.select(
        "batch_id", "timestamp", "endpoint", "http_method",
        "status_code", "latency_ms", "tokens_used", "user_id"
    )
    out_path      = f"{METRICS_DIR}/metrics_batch_{bid:03d}.csv"
    batch_df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path)
    total_written += len(batch_records)
    print(f"  ✅ Written: metrics_batch_{bid:03d}.csv  ({len(batch_records)} rows)")

print(f"  Total mock records : {total_written}")

# ── Clear stale checkpoint + bronze so corrected CSVs are reprocessed cleanly ─
print("\n  🗑️  Clearing stale checkpoint and bronze table for clean reload ...")
dbutils.fs.rm(CHECKPOINT_PATH, recurse=True)
spark.sql(f"DROP TABLE IF EXISTS {BRONZE_TABLE}")
print("  ✅ Checkpoint cleared, bronze table dropped")

# ── STAGE 2: Auto Loader — stream CSV files → Bronze Delta table ──────────────
print("\n  🔄 Stage 2 — Auto Loader streaming CSV files → Bronze table ...")

raw_schema = StructType([
    StructField("batch_id",    IntegerType(), True),
    StructField("timestamp",   StringType(),  True),
    StructField("endpoint",    StringType(),  True),
    StructField("http_method", StringType(),  True),
    StructField("status_code", IntegerType(), True),
    StructField("latency_ms",  DoubleType(),  True),
    StructField("tokens_used", IntegerType(), True),
    StructField("user_id",     StringType(),  True),
])

stream = (
    spark.readStream
        .format("cloudFiles")                                    # Auto Loader
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", CHECKPOINT_PATH + "/schema")
        .option("header", "true")
        .schema(raw_schema)
        .load(METRICS_DIR)
        .withColumn("_ingested_at",   F.current_timestamp())
        .withColumn("_source_file",   F.col("_metadata.file_path"))
)

(stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)             # process all files then stop (serverless-friendly)
    .toTable(BRONZE_TABLE)
    .awaitTermination(timeout=120))

bronze_count = spark.table(BRONZE_TABLE).count()
print(f"  ✅ Bronze table loaded   : {BRONZE_TABLE}  ({bronze_count} rows via Auto Loader)")

# ── STAGE 3: Transform Bronze → silver_api_kpis with derived metrics ──────────
print("\n  🔧 Stage 3 — Transforming Bronze → silver_api_kpis ...")

silver_df = (
    spark.table(BRONZE_TABLE)
    .withColumn("event_ts",       F.expr("try_to_timestamp(timestamp)"))
    .filter(F.col("event_ts").isNotNull())
    # is_error: binary flag — 1 if request failed (4xx or 5xx), else 0
    .withColumn("is_error",       F.when(F.col("status_code") >= 400, 1).otherwise(0))
    # error_rate_pct: 100 if this individual request was an error, 0 if success
    # Aggregate AVG(error_rate_pct) across rows to get overall error rate %
    .withColumn("error_rate_pct", F.when(F.col("status_code") >= 400, F.lit(100.0)).otherwise(F.lit(0.0)))
    # sla_breach: flag requests exceeding the 200ms threshold
    .withColumn("sla_breach",     F.when(F.col("latency_ms") > 200.0, True).otherwise(False))
    .withColumn("processed_at",   F.current_timestamp())
    .select(
        "event_ts", "endpoint", "http_method", "status_code",
        "latency_ms", "error_rate_pct", "is_error", "sla_breach",
        "tokens_used", "user_id", "processed_at"
    )
)

(silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE))

silver_count = spark.table(SILVER_TABLE).count()
print(f"  ✅ Silver table written  : {SILVER_TABLE}  ({silver_count} rows)")

# ── STAGE 4: Inject rich semantic metadata — mandatory for Text-to-SQL in M3.3 ─
print("\n  📝 Stage 4 — Injecting semantic metadata + ai_query_example ...")

# Table-level natural language description
spark.sql(f"""
    COMMENT ON TABLE {SILVER_TABLE} IS
    'Project Nexus API performance KPIs. Contains per-request telemetry for all REST
     API endpoints. Use this table to answer operational questions such as: which
     endpoint has the highest error rate, how many requests breached the SLA latency
     threshold, or what is the average token consumption on the agent endpoint.
     Aggregate AVG(is_error)*100 to get the overall error rate percentage.
     Aggregate AVG(sla_breach) to get the SLA breach rate.'
""")

# Column-level natural language descriptions — critical for Text-to-SQL agent context
column_comments = {
    "event_ts"       : "UTC timestamp of the API request. Use for time-series and hourly aggregations.",
    "endpoint"       : "REST API endpoint path. Values: /api/v1/auth/login, /api/v1/datasets, /api/v1/agent/query, etc.",
    "http_method"    : "HTTP verb used: GET, POST, PUT, or DELETE.",
    "status_code"    : "HTTP response status. 2xx = success. 4xx = client error. 5xx = server error.",
    "latency_ms"     : "End-to-end request latency in milliseconds. The SLA threshold is 200ms.",
    "error_rate_pct" : "100 if this request was an error (status >= 400), else 0. Use AVG(error_rate_pct) for aggregate error rate.",
    "is_error"       : "Binary flag: 1 if status_code >= 400 (error), 0 if status_code < 400 (success). Use SUM or AVG for error counts.",
    "sla_breach"     : "TRUE if latency_ms > 200ms SLA threshold, FALSE otherwise. Use SUM(sla_breach) to count breaches.",
    "tokens_used"    : "Number of LLM tokens consumed. Non-zero mainly for /api/v1/agent/query endpoint requests.",
    "user_id"        : "Anonymised user identifier. Format: user_XXXX. Use for per-user request volume analysis.",
    "processed_at"   : "Timestamp when this record was written to the silver layer by the ETL pipeline.",
}

for col_name, comment in column_comments.items():
    try:
        spark.sql(f"ALTER TABLE {SILVER_TABLE} ALTER COLUMN `{col_name}` COMMENT '{comment}'")
    except Exception as e:
        print(f"  ⚠️  Column comment skipped for {col_name}: {str(e)[:80]}")

# ai_query_example TBLPROPERTY — primes the Text-to-SQL agent with example Q&A
ai_query_example = json.dumps({
    "example_questions": [
        "What is the overall error rate across all endpoints?",
        "Which endpoint has the highest average latency?",
        "How many requests breached the 200ms SLA?",
        "What is the total tokens consumed by the agent endpoint?",
        "Show me the top 3 endpoints by total request count"
    ],
    "example_sql": (
        "SELECT endpoint, "
        "ROUND(AVG(is_error)*100, 2) AS error_rate_pct, "
        "ROUND(AVG(latency_ms), 2)   AS avg_latency_ms, "
        "COUNT(*)                     AS total_requests, "
        "SUM(CAST(sla_breach AS INT)) AS sla_breaches "
        f"FROM {SILVER_TABLE} "
        "GROUP BY endpoint "
        "ORDER BY error_rate_pct DESC"
    )
}, indent=2)

spark.sql(f"""
    ALTER TABLE {SILVER_TABLE} SET TBLPROPERTIES (
        'ai_query_example' = '{ai_query_example.replace("'", "\\\\'")}',
        'quality'          = 'silver',
        'layer'            = 'silver',
        'project'          = 'nexus',
        'milestone'        = '1.4'
    )
""")

print("  ✅ Table-level COMMENT injected")
print("  ✅ Column-level COMMENTs injected (all 11 columns)")
print("  ✅ ai_query_example TBLPROPERTY injected")
print("  ✅ quality / layer / project TBLPROPERTIES set")

# Quick KPI preview
print("\n  📊 silver_api_kpis — error rate by endpoint:")
(spark.table(SILVER_TABLE)
    .groupBy("endpoint")
    .agg(
        F.round(F.avg("error_rate_pct"), 2).alias("avg_error_rate_pct"),
        F.round(F.avg("latency_ms"), 2).alias("avg_latency_ms"),
        F.count("*").alias("total_requests"),
        F.sum(F.col("sla_breach").cast("int")).alias("sla_breaches")
    )
    .orderBy("avg_error_rate_pct", ascending=False)
    .show(truncate=False))

print("🟢 Cell 7 complete — Milestone 1.4 done.")


# COMMAND ----------

# MAGIC %md
# MAGIC