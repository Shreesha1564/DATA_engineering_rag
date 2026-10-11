Agentic RAG & Data + AI Pipeline on Databricks

1. Project Overview

This project is an end-to-end Agentic RAG application built on
Databricks.

The goal is to answer business questions that require information from
different sources instead of relying only on a normal chatbot. The
system combines:

Unstructured documents → processed and stored using Delta Lake

Vector Search → retrieves relevant document content using
semantic similarity

SQL / Unity Catalog functions → performs structured calculations
such as SLA analysis

Web Search → handles questions that are outside the internal
knowledge base

LLM / Agent framework → decides which tool should be used for a
question

Streamlit / Databricks App → provides a user-facing interface

MLflow → provides tracing and evaluation

Lakeflow Jobs → orchestrates the complete pipeline

The assignment describes this as an Agentic RAG system, where the
LLM is given tools and can dynamically choose between retrieval, SQL
execution, and web search.

2. Business Problem

The business has information spread across different systems.

For example, a stakeholder may ask:

"Is our API performing within SLA parameters, and what security
standards govern our password management protocols?"

The information required to answer this question exists in different
places:

Security policies are stored in documents.

API performance metrics are stored in SQL tables.

A traditional approach would require a person to manually search
documents, write SQL queries, and combine the results.

The goal of this project is to automate this process using an AI Agent
with tools.

3. High-Level Architecture

                    User Question
                         |
                         v
                +------------------+
                |   AI Agent / LLM |
                +------------------+
                         |
              Decides which tool
              is required
              /       |        \
             /        |         \
            v         v          v
      Vector Search   SQL      Web Search
          |            |           |
          v            v           v
     Documents     API/KPI      External
     / RAG data     tables       knowledge
          \            |           /
           \           |          /
            +----------+----------+
                       |
                       v
                 Final Answer

4. Data Engineering Pipeline

The data pipeline is built using Databricks + Delta Lake + Unity
Catalog.

Unstructured document flow

Documents
   |
   v
Unity Catalog Volume
   |
   v
Bronze Delta Table
   |
   v
Cleaning / Extraction
   |
   v
silver_requirements
   |
   v
Chunking
   |
   v
gold_chunks
   |
   v
Vector Search Index

Structured API metrics flow

Mock API Metrics
      |
      v
Volume
      |
      v
Auto Loader
      |
      v
Bronze Table
      |
      v
silver_api_kpis

The API KPI table contains derived metrics such as error rate
percentage and is enriched with metadata so that an AI system can
understand the meaning of the data.

5. Unity Catalog

Unity Catalog is used as the central governance layer.

The project uses a dedicated catalog and schema:

nexus_catalog
    |
    └── dev_schema

Unity Catalog is useful here because it provides centralized management
and access control for:

Tables

Volumes

Functions

Vector Search related data

AI tools

The project also uses SQL GRANT statements to demonstrate different
access privileges for human analysts and automated agents.

6. Delta Lake and Change Data Feed

Delta tables are used as the main storage layer.

Important tables include:

silver_requirements
silver_api_kpis
gold_chunks

Change Data Feed (CDF) is enabled on the relevant Delta tables.

Why CDF?

CDF allows downstream systems to identify which rows have changed
instead of processing the complete table every time.

This is particularly useful for keeping a Vector Search index
synchronized with changes in the source Delta table.

7. Document Chunking

Large documents are divided into smaller logical chunks before
embedding.

The purpose of chunking is to make semantic retrieval more effective.

Instead of searching an entire document:

Large Document
      |
      v
Many smaller chunks
      |
      v
Embeddings
      |
      v
Vector Search

Each chunk receives a unique identifier and is stored in the
gold_chunks Delta table.

Important considerations:

Chunk size

Overlap between chunks

Unique chunk ID

Source document information

Text content

The resulting chunks become the knowledge units used by the RAG system.

8. Vector Search

Databricks Vector Search is used for semantic retrieval.

The basic process is:

Document Chunk
      |
      v
Embedding Model
      |
      v
Vector Representation
      |
      v
Vector Search Index

When a user asks a question:

User Question
      |
      v
Question Embedding
      |
      v
Similarity Search
      |
      v
Most Relevant Chunks

The project uses a Databricks Foundation Model embedding endpoint to
generate embeddings.

The retrieval pipeline is designed to first retrieve relevant chunks and
then use semantic relevance to select the best information.

9. Two-Stage Retrieval

The retrieval component follows a two-stage approach.

Stage 1 --- Vector Search

Retrieve the top relevant chunks from the Vector Search index.

Query
  |
  v
Vector Search
  |
  v
Top 10 chunks

Stage 2 --- Reranking

The retrieved chunks are passed to a reranker model.

Top 10 chunks
      |
      v
Reranker
      |
      v
Top 3 most relevant chunks

This improves retrieval quality because the initial vector search
provides candidates while the reranker performs a more focused semantic
comparison.

10. Unity Catalog Function as an AI Tool

An SLA Calculator is implemented as a Unity Catalog Python Function.

Instead of allowing the LLM to execute arbitrary Python code, the
required calculation is exposed as a controlled tool.

Example:

Agent
  |
  | "Calculate API SLA"
  v
SLA Calculator
  |
  v
Structured result

The function includes defensive handling for:

Null values

Divide-by-zero

Invalid inputs

The SQL COMMENT property is also used to describe the function and
guide the Agent about when the tool should be used.

11. Agentic Tool Calling

The key difference between a normal RAG chatbot and this project is
tool selection.

The Agent has access to three main capabilities:

Tool 1 --- RAG / Vector Search

Used when the answer exists in internal documents.

Example:

"What are our password security requirements?"

The Agent routes the question to Vector Search.

Tool 2 --- SQL / SLA Calculator

Used when a calculation or structured API metric is required.

Example:

"Is our API meeting the SLA?"

The Agent uses the SQL-based tool.

Tool 3 --- Web Search

Used when the required information is outside the internal knowledge
base.

Example:

"What is the latest industry standard for X?"

The Agent can use web search.

12. Agent Routing Logic

Conceptually:

                    User Query
                        |
                        v
                  Agent / LLM
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
      Internal       Numeric /      External
      documents      KPI query      knowledge
          |             |             |
          v             v             v
     Vector Search     SQL        Web Search
          \             |             /
           \            |            /
            +-----------+-----------+
                        |
                        v
                  Final Response

The important idea is that the LLM decides which tool is
appropriate, rather than always performing the same retrieval
operation.

13. Text-to-SQL / Data Q&A

The project also includes a Data Q&A capability for the
silver_api_kpis table.

The objective is:

Natural Language Question
          |
          v
       LLM
          |
          v
      SQL Query
          |
          v
silver_api_kpis
          |
          v
      SQL Result
          |
          v
   Natural Language Answer

The semantic metadata added to the table helps the AI understand what
the columns represent.

If native Genie functionality is unavailable because of workspace
restrictions, the assignment allows a code-based Text-to-SQL equivalent.

14. Streamlit / Databricks App

The final system should not require business users to work directly
inside a notebook.

A Streamlit application provides a simple interface:

+--------------------------------------+
|          Data AI Copilot             |
+--------------------------------------+
| Ask your question:                   |
|                                      |
| [ Is our API meeting the SLA? ]      |
|                                      |
|              [ Ask ]                 |
+--------------------------------------+
| Answer                               |
| ...                                  |
+--------------------------------------+

The application acts as the user-facing layer for the Agent.

15. MLflow Observability

MLflow is used to observe the Agent.

Important information that can be captured includes:

Agent execution

Tool calls

Token consumption

Latency

Reasoning / execution flow

Tool-calling DAG

This is important because an Agent can involve multiple model and tool
calls, making normal application logging insufficient.

16. LLM-as-a-Judge Evaluation

The Agent is evaluated using a test dataset containing:

Question
Ground-truth answer
Agent prediction

MLflow GenAI evaluation is used to score the responses on a 1--5
accuracy scale.

The final scorecard contains:

Numerical score

Judge's justification

This provides a more systematic way to evaluate the Agent instead of
checking answers manually.

17. AI Query / Batch LLM Processing

The project also demonstrates ai_query() for batch LLM processing.

Conceptually:

Large Delta Table
       |
       v
    ai_query()
       |
       v
Foundation Model
       |
       v
Extracted information
       |
       v
Result table

This is useful when the same LLM operation needs to be performed across
many rows rather than one user request at a time.

18. Lakeflow Jobs

The complete pipeline can be orchestrated using a Databricks Workflow /
Lakeflow Job.

A simplified DAG is:

Setup
  |
  v
Ingestion
  |
  v
Cleaning
  |
  v
Chunking
  |
  v
Vector Index Sync
  |
  v
Agent / Evaluation

The job is configured with:

env=prod

Maximum of 3 retries

Target notebook

This makes the pipeline repeatable rather than requiring manual
execution of every notebook step.

19. Important Databricks Concepts Learned

Unity Catalog

Governance layer for data and AI assets.

Delta Lake

Reliable table storage with ACID transactions.

Change Data Feed

Tracks changes in Delta tables for incremental downstream processing.

Auto Loader

Incrementally ingests new files into Delta tables.

Vector Search

Retrieves semantically similar document chunks.

Embeddings

Convert text into numerical vectors representing semantic meaning.

RAG

Retrieves relevant external knowledge before generating an answer.

Agentic AI

Allows an LLM to select and execute tools to solve a problem.

Tool Calling

Allows the LLM to interact with controlled functions and external
systems.

MLflow

Provides tracing and evaluation for the Agent.

Lakeflow Jobs

Provides workflow orchestration and scheduling.

20. Free Edition / Infrastructure Constraints

The assignment explicitly notes that Databricks workspace tiers can
restrict infrastructure features.

Because of this, some enterprise capabilities may not be available in a
Free/Serverless environment.

The important engineering approach is:

Do not assume that a premium feature is available. Identify the
limitation and implement the closest supported alternative.

For example, where a native Databricks feature is unavailable, the
assignment allows an equivalent code-based implementation in certain
cases.
