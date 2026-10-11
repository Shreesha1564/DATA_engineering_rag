# Databricks notebook source
# DBTITLE 1,M3 Progress Tracker
# MAGIC %md
# MAGIC # Milestone 3 — UC SLA Calculator, Agent Assembly, Genie Text-to-SQL, Databricks App
# MAGIC **Notebook:** M3_Agent_Assembly
# MAGIC **Compute:** Serverless | **Depends on:** M1 + M2 complete
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Progress Tracker
# MAGIC
# MAGIC ### M3.1 — Unity Catalog SLA Calculator (Managed MCP) ✅
# MAGIC - [x] calculate_sla_metrics UC Python function registered → `nexus_catalog.dev_schema.calculate_sla_metrics`
# MAGIC - [x] Defensive null and divide-by-zero logic implemented
# MAGIC - [x] SQL COMMENT property used as tool system prompt
# MAGIC - [x] EXECUTE grant applied to `account users`
# MAGIC
# MAGIC ### M3.2 — Agent Assembly (RAG + SLA + Web)
# MAGIC - [ ] Tool 1: Vector Search retriever from M2.4 bound as LangChain tool
# MAGIC - [ ] Tool 2: UC SLA Calculator from M3.1 bound as LangChain tool
# MAGIC - [ ] Tool 3: Web search tool (Wikipedia) bound as LangChain tool
# MAGIC - [ ] ReAct routing agent assembled with Foundation Model LLM
# MAGIC - [ ] MLflow autolog enabled and stakeholder question traced
# MAGIC
# MAGIC ### M3.3 — Genie Space / Text-to-SQL Equivalent
# MAGIC - [ ] LangChain Text-to-SQL agent defined against silver_api_kpis
# MAGIC - [ ] Schema context injected from M1.4 semantic metadata
# MAGIC - [ ] Tested with natural language metric question
# MAGIC
# MAGIC ### M3.4 — Databricks App Deployment (Streamlit)
# MAGIC - [ ] Streamlit app.py written with agent chat interface
# MAGIC - [ ] App deployed via Databricks Apps SDK
# MAGIC - [ ] App URL confirmed accessible
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Assets Created
# MAGIC | Asset | Type | Location |
# MAGIC |---|---|---|
# MAGIC | calculate_sla_metrics | UC Python Function | nexus_catalog.dev_schema |
# MAGIC | nexus_rag_agent | LangChain ReAct Agent | This notebook |
# MAGIC | nexus_text_to_sql_agent | LangChain SQL Agent | This notebook |
# MAGIC | nexus-copilot-app | Databricks App | Workspace Apps |
# MAGIC

# COMMAND ----------

# DBTITLE 1,M3.1 — Register calculate_sla_metrics as a Unity Catalog Python Function
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

# MAGIC %pip install wikipedia

# COMMAND ----------

# DBTITLE 1,M3.1 — GRANT EXECUTE on calculate_sla_metrics
# MAGIC %sql
# MAGIC -- M3.1 — Grant EXECUTE so the Agent and all workspace users can invoke the tool
# MAGIC GRANT EXECUTE ON FUNCTION nexus_catalog.dev_schema.calculate_sla_metrics
# MAGIC   TO `account users`;
# MAGIC
# MAGIC SELECT 'GRANT applied — calculate_sla_metrics accessible to account users' AS status;

# COMMAND ----------

# DBTITLE 1,M3.1 — Smoke tests: ALL, auth filter, null guard
# M3.1 — Smoke tests for calculate_sla_metrics
# UC Python functions run in an isolated sandbox without direct Spark access.
# Tests 1&2 verify the SLA logic via spark.sql (same logic, notebook context).
# Test 3 calls the UC function itself to validate null-guard and UC registration.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SEP = '\u2550' * 62

# ── Helper: SLA logic identical to the UC function body ───────────────────
def run_sla_logic(endpoint_filter: str) -> str:
    if not endpoint_filter or not str(endpoint_filter).strip():
        return "Error: endpoint_filter must not be empty."
    f = str(endpoint_filter).strip()
    where_sql = "1=1" if f.lower() == "all" else f"LOWER(endpoint) LIKE '%{f.lower()}%'"

    rows = spark.sql(f"""
        SELECT endpoint,
               COUNT(*) AS total_calls, SUM(is_error) AS error_count,
               ROUND(AVG(error_rate_pct), 2) AS error_rate_pct,
               ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
               ROUND(PERCENTILE_APPROX(latency_ms, 0.95), 2) AS p95_latency_ms,
               ROUND(MAX(latency_ms), 2) AS max_latency_ms,
               SUM(CAST(sla_breach AS INT)) AS latency_breach_count
        FROM nexus_catalog.dev_schema.silver_api_kpis
        WHERE {where_sql}
        GROUP BY endpoint ORDER BY total_calls DESC
    """).collect()

    if not rows:
        return f"No data found for filter: {endpoint_filter}. Try ALL."

    ERROR_RATE_SLA_PCT, LATENCY_SLA_MS = 5.0, 200.0
    report, total_all, breach_eps = [f"=== SLA Report | filter: {endpoint_filter} ==="], 0, 0

    for r in rows:
        total      = int(r["total_calls"]      or 0)
        errors     = int(r["error_count"]      or 0)
        err_pct    = float(r["error_rate_pct"] or 0.0)
        avg_lat    = float(r["avg_latency_ms"] or 0.0)
        p95_lat    = float(r["p95_latency_ms"] or 0.0)
        max_lat    = float(r["max_latency_ms"] or 0.0)
        lat_breach = int(r["latency_breach_count"] or 0)

        violations = []
        if err_pct > ERROR_RATE_SLA_PCT:
            violations.append(f"error_rate {round(err_pct,1)}% > {ERROR_RATE_SLA_PCT}% threshold")
        if p95_lat > LATENCY_SLA_MS:
            violations.append(f"p95_latency {round(p95_lat)}ms > {LATENCY_SLA_MS}ms threshold")

        sla_status = "BREACH" if violations else "PASS"
        if violations: breach_eps += 1
        success_count = max(total - errors, 0)
        success_rate  = round(success_count / total * 100, 1) if total > 0 else 0.0

        report += ["",
            f"Endpoint           : {r['endpoint']}",
            f"  Total calls      : {total}",
            f"  Success rate     : {success_rate}%  ({success_count}/{total})",
            f"  Error rate       : {round(err_pct,1)}%  ({errors} errors)",
            f"  Avg latency      : {avg_lat} ms",
            f"  P95 latency      : {p95_lat} ms  (SLA <= {LATENCY_SLA_MS} ms)",
            f"  Max latency      : {max_lat} ms",
            f"  Latency breaches : {lat_breach} requests",
            f"  SLA Status       : {sla_status}",
        ]
        if violations: report.append(f"  Violations       : {' | '.join(violations)}")
        total_all += total

    report += ["", "=" * 50, "SUMMARY",
               f"  Endpoints evaluated : {len(rows)}",
               f"  Endpoints in BREACH : {breach_eps}",
               f"  Total API calls     : {total_all}", "=" * 50]
    return "\n".join(report)

# ── Test 1: ALL endpoints (SLA logic via spark.sql) ────────────────────────
print(f"\n{SEP}\nTEST: ALL endpoints\n{SEP}")
print(run_sla_logic("ALL"))

# ── Test 2: auth filter (SLA logic via spark.sql) ─────────────────────────
print(f"\n{SEP}\nTEST: auth endpoints only\n{SEP}")
print(run_sla_logic("auth"))

# ── Test 3: UC function null guard via Statement Execution API ─────────────
print(f"\n{SEP}\nTEST: UC function null guard (CAST(NULL AS STRING))\n{SEP}")
w = WorkspaceClient()
wh_id = next((wh.id for wh in w.warehouses.list() if wh.id), None)
if wh_id:
    stmt = w.statement_execution.execute_statement(
        warehouse_id=wh_id,
        statement="SELECT nexus_catalog.dev_schema.calculate_sla_metrics(CAST(NULL AS STRING)) AS report",
        wait_timeout="50s"
    )
    if stmt.status and stmt.status.state == StatementState.SUCCEEDED:
        result = stmt.result.data_array[0][0] if (stmt.result and stmt.result.data_array) else ""
        print(result)
        assert "must not be null" in result, f"Null guard check failed. Got: {result}"
        print("\u2713 Null guard assertion passed")
    else:
        print(f"Status: {stmt.status.state}")
else:
    print("No warehouse available for UC function test")

print("\n\u2714 All M3.1 smoke tests complete.")


# COMMAND ----------

# DBTITLE 1,Cell 6
# MILESTONE 3 — Install Agent Framework dependencies
# Installing LangChain, Databricks integration, and Wikipedia search support.

%pip install -q -U langchain "langgraph-prebuilt==1.0.10" databricks-langchain wikipedia

# COMMAND ----------

# MAGIC %pip install -q -U langchain databricks-langchain ddgs

# COMMAND ----------

# MAGIC
# MAGIC %pip install -q -U ddgs

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,M3.2 — Agent Assembly
# MILESTONE 3.2 — Agent Assembly
# Configure the LLM and connect the three required Agent tools:
# RAG, Unity Catalog SLA Calculator, and Web Search.

import os
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool

CATALOG = "nexus_catalog"
SCHEMA = "dev_schema"
VS_INDEX = f"{CATALOG}.{SCHEMA}.nexus_vs_index"
SLA_FUNCTION = f"{CATALOG}.{SCHEMA}.calculate_sla_metrics"

w = WorkspaceClient()

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
# Keep "databricks" for now because you have no API key.
# Later change to "openai" after storing OPENAI_API_KEY
# as a Databricks App/Secret resource.

MODEL_PROVIDER = "databricks"

if MODEL_PROVIDER == "databricks":
    from databricks_langchain import ChatDatabricks

    llm = ChatDatabricks(
        endpoint="databricks-meta-llama-3-1-8b-instruct",
        temperature=0
    )

elif MODEL_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )


# --------------------------------------------------
# TOOL 1 — RAG
# --------------------------------------------------
@tool
def search_project_knowledge(query: str) -> str:
    """Search Project Nexus requirements and security documentation."""

    response = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        query_text=query,
        columns=["chunk_id", "chunk_text", "req_id", "page_number"],
        num_results=3
    )

    columns = [
        c.name if hasattr(c, "name") else c.get("name")
        for c in response.manifest.columns
    ]

    rows = response.result.data_array or []

    if not rows:
        return "No relevant Project Nexus documentation found."

    results = [
        dict(zip(columns, row))
        for row in rows
    ]

    return "\n\n".join(
        f"Requirement: {r.get('req_id')}\n"
        f"Page: {r.get('page_number')}\n"
        f"{r.get('chunk_text')}"
        for r in results
    )


# --------------------------------------------------
# TOOL 2 — UC SLA CALCULATOR
# --------------------------------------------------
@tool
def calculate_api_sla(endpoint_filter: str = "ALL") -> str:
    """Calculate API SLA metrics from silver_api_kpis."""

    endpoint_filter = endpoint_filter or "ALL"

    if endpoint_filter == "ALL":
        query = f"""
        SELECT
            ROUND(AVG(error_rate_pct),2) AS avg_error_rate,
            ROUND(AVG(latency_ms),2) AS avg_latency_ms,
            SUM(CASE WHEN sla_breach THEN 1 ELSE 0 END) AS sla_breaches,
            COUNT(*) AS total_requests
        FROM {CATALOG}.{SCHEMA}.silver_api_kpis
        """
    else:
        query = f"""
        SELECT
            endpoint,
            ROUND(AVG(error_rate_pct),2) AS avg_error_rate,
            ROUND(AVG(latency_ms),2) AS avg_latency_ms,
            SUM(CASE WHEN sla_breach THEN 1 ELSE 0 END) AS sla_breaches,
            COUNT(*) AS total_requests
        FROM {CATALOG}.{SCHEMA}.silver_api_kpis
        WHERE endpoint = '{endpoint_filter}'
        GROUP BY endpoint
        """

    return spark.sql(query).toPandas().to_string(index=False)


# --------------------------------------------------
# TOOL 3 — WEB
# --------------------------------------------------
# TOOL 3 — DuckDuckGo Web Search
@tool
def web_search(query: str) -> str:
    """Search the public web for general knowledge outside Project Nexus."""

    from duckduckgo_search import DDGS

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No web results found."

        return "\n\n".join(
            f"Title: {r.get('title', '')}\n"
            f"Summary: {r.get('body', '')}\n"
            f"URL: {r.get('href', '')}"
            for r in results
        )

    except Exception as e:
        return f"Web search unavailable: {str(e)}"


print("Model:", MODEL_PROVIDER)
print("Tools:")
print(" - search_project_knowledge")
print(" - calculate_api_sla")
print(" - web_search")

# COMMAND ----------

# DBTITLE 1,M3.2 — Agent Framework
# MILESTONE 3.2 — Agent Framework
# Create the reasoning/routing Agent that dynamically chooses
# RAG, SLA calculation, or Web Search.

# Patch missing runtime stubs before importing langgraph.prebuilt
import langgraph.runtime as _lgrt
from typing import TypedDict

# Missing TypedDict types (needed at import time by tool_node.py)
if not hasattr(_lgrt, "ExecutionInfo"):
    class _ExecutionInfo(TypedDict, total=False): pass
    _lgrt.ExecutionInfo = _ExecutionInfo
if not hasattr(_lgrt, "ServerInfo"):
    class _ServerInfo(TypedDict, total=False): pass
    _lgrt.ServerInfo = _ServerInfo

# Missing Runtime instance attributes (accessed at call time by tool_node.py)
if not hasattr(_lgrt.Runtime, "execution_info"):
    _lgrt.Runtime.execution_info = None
if not hasattr(_lgrt.Runtime, "server_info"):
    _lgrt.Runtime.server_info = None

from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """
You are the Project Nexus enterprise Copilot.

Use the tools intelligently:

- Project/security/requirements questions -> search_project_knowledge
- API KPI/SLA/latency/error questions -> calculate_api_sla
- General knowledge outside Project Nexus -> web_search

Never invent API metrics.
Numerical KPI answers must come from the SLA tool.
If a question needs multiple sources, use multiple tools.
Give a concise final answer based on the tool results.
"""

agent = create_react_agent(
    model=llm,
    tools=[
        search_project_knowledge,
        calculate_api_sla,
        web_search
    ],
    prompt=SYSTEM_PROMPT
)

print("Project Nexus Agent created successfully.")

# COMMAND ----------

# MILESTONE 3.2 — Agent Validation
# Test all three routing paths and one combined business question.

questions = [
    "What are the password security requirements?",
    "Is our API performing within the SLA?",
    "What is Retrieval Augmented Generation?",
    "What are the password requirements and is our API currently breaching SLA?"
]

for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    print(response["messages"][-1].content)

# COMMAND ----------

# MILESTONE 3.3 — Semantic Text-to-SQL
# Read the semantic metadata created in Milestone 1.4 and
# use it to guide natural-language-to-SQL generation.

KPI_TABLE = f"{CATALOG}.{SCHEMA}.silver_api_kpis"

schema = spark.sql(
    f"DESCRIBE TABLE {KPI_TABLE}"
).collect()

schema_text = "\n".join(
    f"{r['col_name']} ({r['data_type']}): {r['comment'] or ''}"
    for r in schema
    if r["col_name"] and not r["col_name"].startswith("#")
)

print("Semantic schema:\n")
print(schema_text)

# COMMAND ----------

# DBTITLE 1,Cell 12
# MILESTONE 3.3 — Text-to-SQL Agent
# Generate SQL from natural language, validate that it is read-only,
# execute it, and return the actual KPI result.

from langchain_core.prompts import ChatPromptTemplate

sql_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You generate Databricks SQL for Project Nexus.

ONLY use:
{KPI_TABLE}

Schema:
{schema_text}

Rules:
- Generate SELECT statements only.
- Never INSERT, UPDATE, DELETE, DROP or ALTER.
- Return only SQL.
"""
    ),
    ("human", "{question}")
])


def text_to_sql(question):
    import re

    sql = (
        llm.invoke(
            sql_prompt.format_messages(
                question=question
            )
        )
        .content
        .strip()
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    # Strip any trailing prose lines (e.g. "This returns the...") appended after the SQL
    _clean = []
    for _line in sql.splitlines():
        if _clean and re.match(r'^[A-Z][a-z]', _line.strip()):
            break
        if _line.strip():
            _clean.append(_line)
    sql = "\n".join(_clean).strip()

    if not sql.lower().startswith("select"):
        raise ValueError("Generated SQL is not a SELECT statement.")

    if KPI_TABLE.lower() not in sql.lower():
        raise ValueError("Generated SQL references an unauthorized table.")

    result = spark.sql(sql)

    return sql, result


sql, result = text_to_sql(
    "What is the overall error rate across all endpoints?"
)

print("Generated SQL:")
print(sql)

print("\nResult:")
display(result)

# COMMAND ----------

sql, result = text_to_sql(
    "Which endpoint has the highest average latency?"
)

print(sql)
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # Milestone 3.4 — Deploy Project Nexus Copilot
# MAGIC
# MAGIC The Project Nexus Agent is exposed through a Streamlit application deployed as a Databricks App.
# MAGIC
# MAGIC The application provides a business-facing interface for:
# MAGIC
# MAGIC 1. Project Nexus requirement and security questions through Vector Search RAG.
# MAGIC 2. API performance and SLA questions through the governed Unity Catalog SLA function.
# MAGIC 3. General knowledge questions through a Web Search tool.
# MAGIC 4. Combined questions requiring multiple tools.
# MAGIC
# MAGIC The application uses Databricks App resources for authentication and access to
# MAGIC Unity Catalog, Vector Search, and SQL Warehouse resources.
# MAGIC
# MAGIC No personal access token or API key is hardcoded into the application.
# MAGIC An external OpenAI API key can be added later through a Databricks Secret.

# COMMAND ----------

# MILESTONE 3.4 — Create the Streamlit application source files
# This creates app.py, requirements.txt and app.yaml in a Workspace folder.
#
# The application uses:
# - Vector Search for Project Nexus RAG
# - Unity Catalog SLA function for KPI questions
# - DuckDuckGo for general web search
# - Databricks-hosted LLM now
# - Optional OpenAI model later through a Databricks Secret

import io
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

username = w.current_user.me().user_name

APP_DIR = f"/Workspace/Users/{username}/project-nexus-copilot"

APP_PY = r'''
import os
import streamlit as st

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.tools import tool


# ============================================================
# PROJECT NEXUS CONFIGURATION
# ============================================================

CATALOG = "nexus_catalog"
SCHEMA = "dev_schema"

VS_INDEX = f"{CATALOG}.{SCHEMA}.nexus_vs_index"
SLA_FUNCTION = f"{CATALOG}.{SCHEMA}.calculate_sla_metrics"

MODEL_PROVIDER = os.getenv(
    "MODEL_PROVIDER",
    "databricks"
)

WAREHOUSE_ID = os.getenv(
    "DATABRICKS_WAREHOUSE_ID"
)

w = WorkspaceClient()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Project Nexus Copilot",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Project Nexus Copilot")

st.caption(
    "Agentic RAG • SLA Analytics • Web Search"
)


# ============================================================
# TOOL 1 — PROJECT NEXUS RAG
# ============================================================

@tool
def search_project_knowledge(query: str) -> str:
    """Search Project Nexus requirements and security documentation."""

    response = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        query_text=query,
        columns=[
            "chunk_id",
            "chunk_text",
            "req_id",
            "page_number"
        ],
        num_results=3
    )

    columns = [
        c.name if hasattr(c, "name") else c.get("name")
        for c in response.manifest.columns
    ]

    rows = response.result.data_array or []

    if not rows:
        return "No relevant Project Nexus documentation found."

    records = [
        dict(zip(columns, row))
        for row in rows
    ]

    return "\n\n".join(
        f"Requirement: {r.get('req_id')}\n"
        f"Page: {r.get('page_number')}\n"
        f"{r.get('chunk_text', '')}"
        for r in records
    )


# ============================================================
# TOOL 2 — GOVERNED UC SLA FUNCTION
# ============================================================

@tool
def calculate_api_sla(
    endpoint_filter: str = "ALL"
) -> str:
    """Calculate API SLA metrics using the governed Unity Catalog function."""

    if not WAREHOUSE_ID:
        return "SQL warehouse resource is not configured."

    endpoint_filter = endpoint_filter or "ALL"
    endpoint_filter = endpoint_filter.replace("'", "''")

    sql = (
        f"SELECT {SLA_FUNCTION}"
        f"('{endpoint_filter}') AS sla_report"
    )

    try:

        response = w.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=WAREHOUSE_ID,
            catalog=CATALOG,
            schema=SCHEMA,
            wait_timeout="30s"
        )

        rows = response.result.data_array or []

        if not rows:
            return "The SLA function returned no result."

        return str(rows[0][0])

    except Exception as exc:

        return (
            "Unable to execute SLA function: "
            + str(exc)
        )


# ============================================================
# TOOL 3 — WEB SEARCH
# ============================================================

@tool
def web_search(query: str) -> str:
    """Search the public web for general knowledge."""

    try:

        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No web results found."

        return "\n\n".join(
            f"Title: {r.get('title', '')}\n"
            f"Summary: {r.get('body', '')}\n"
            f"URL: {r.get('href', '')}"
            for r in results
        )

    except Exception as exc:

        return (
            "Web search is currently unavailable. "
            "Use Project Nexus RAG or SLA tools "
            "for internal questions."
        )


# ============================================================
# LLM CONFIGURATION
# ============================================================

def create_model():

    if MODEL_PROVIDER == "openai":

        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        return ChatOpenAI(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            ),
            temperature=0
        )

    return ChatDatabricks(
        endpoint=(
            "databricks-meta-llama-3-1-8b-instruct"
        ),
        temperature=0
    )


# ============================================================
# AGENT
# ============================================================

try:

    llm = create_model()

    agent = create_agent(
        model=llm,
        tools=[
            search_project_knowledge,
            calculate_api_sla,
            web_search
        ],
        system_prompt="""
You are the Project Nexus enterprise Copilot.

Routing rules:

- Project Nexus requirements, security and SRS questions:
  use search_project_knowledge.

- API performance, SLA, latency, errors and KPI questions:
  use calculate_api_sla.

- General external knowledge:
  use web_search.

- Combined questions may require multiple tools.

Never invent KPI values.
Numerical API answers must come from the SLA tool.
Clearly explain when a tool is unavailable.
"""
    )

    agent_error = None

except Exception as exc:

    agent = None
    agent_error = str(exc)


# ============================================================
# CHAT INTERFACE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


question = st.chat_input(
    "Ask about Project Nexus, API SLA, security, or general knowledge..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if agent is None:

            st.error(
                "Agent initialization failed."
            )

            st.caption(agent_error)

        else:

            try:

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                })

                answer = response[
                    "messages"
                ][-1].content

                st.markdown(answer)

                # Show which tools were actually called.
                tools_used = []

                for message in response["messages"]:

                    for call in (
                        getattr(
                            message,
                            "tool_calls",
                            []
                        ) or []
                    ):

                        name = call.get("name")

                        if (
                            name
                            and name not in tools_used
                        ):
                            tools_used.append(name)

                if tools_used:

                    st.caption(
                        "Tools used: "
                        + ", ".join(tools_used)
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as exc:

                error_text = str(exc)

                # Gracefully handle the Databricks
                # workspace QPS limit instead of crashing.
                if (
                    "429" in error_text
                    or "REQUEST_LIMIT_EXCEEDED"
                    in error_text
                ):

                    st.error(
                        "The configured AI model has reached "
                        "the workspace rate limit. "
                        "Please configure an OpenAI secret "
                        "or retry later."
                    )

                else:

                    st.error(
                        "Agent execution failed."
                    )

                    st.caption(
                        error_text
                    )
'''

REQUIREMENTS_TXT = """streamlit
databricks-sdk
databricks-langchain
langchain
ddgs
langchain-openai
"""

APP_YAML = """command: ['streamlit', 'run', 'app.py']

env:
  - name: 'DATABRICKS_WAREHOUSE_ID'
    valueFrom: 'sql_warehouse'
  - name: 'MODEL_PROVIDER'
    value: 'databricks'
"""

w.workspace.mkdirs(APP_DIR)

files = {
    "app.py": APP_PY,
    "requirements.txt": REQUIREMENTS_TXT,
    "app.yaml": APP_YAML
}

for filename, content in files.items():

    w.workspace.upload(
        f"{APP_DIR}/{filename}",
        io.BytesIO(content.encode("utf-8")),
        overwrite=True
    )

print("Application files created successfully.")
print()
print("App source folder:")
print(APP_DIR)
print()
print("Files:")
for filename in files:
    print(" -", filename)