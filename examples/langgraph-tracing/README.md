# LangGraph Agent with MLflow Tracing

A sample AI agent built with LangGraph that queries the National Parks Service, with automatic tracing via `mlflow.langchain.autolog()`.

## What This Demonstrates

- **ReAct Agent with LangGraph using NPS mcp** - A simple agent loop using LangGraph's `StateGraph`

- **MLflow Auto-Tracing** - Automatic trace capture with `mlflow.langchain.autolog()` (single graph invocation builds one trace )
- **Tracing conversation** - Using LangGraph's thread IDs or manually with `@mlflow.trace` decorator to view a group of traces as a session in the UI

## Prerequisites

1. **Python 3.12+** (recommended via `uv`)
2. **NPS MCP Server** running on `localhost:3005`
3. **OpenAI API Key**

## Quick Start

### 1. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.12
# OR `python3.12 -m venv .venv`
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
uv pip install mlflow langchain langchain-core langchain-openai langgraph langgraph-prebuilt langchain-mcp-adapters python-dotenv
```

### 3. Set Environment Variables

Create a `.env` file in this directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 4. NPS MCP Server

Refer setting up server in the `agents_tracing-eval_mlflow/nps_agent/` folder.

### 5. Run the Notebook

Open and run `nps-agent-langgraph.ipynb` in Jupyter:

### 6. View Traces in MLflow UI

After running the notebook, start the MLflow UI to view traces:

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5001 in your browser.

## Project Structure

```
langgraph-tracing/
├── README.md                    # This file
├── nps-agent-langgraph.ipynb    # Main notebook with examples
├── .env                         # Environment variables (not committed)
└── mlflow.db                    # SQLite database for traces (created on run)
```

## References

- [MLflow LangGraph Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/)
- [MLflow User & Session Tracking](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/)
- [LangChain MCP Integration](https://python.langchain.com/docs/integrations/tools/mcp/)
- [OpenAI Agents SDK MCP](https://openai.github.io/openai-agents-python/mcp/) (related example using OpenAI SDK)