# Cell 1: Imports and Setup
import mlflow
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Langgraph-NPS-Agent")
mlflow.langchain.autolog()

# NPS_MCP_URL = "https://patternable-spiracular-deirdre.ngrok-free.dev/sse"
NPS_MCP_URL = "http:localhost:3005/sse"

# Cell 2: Initialize MCP Client and get tools
client = MultiServerMCPClient(
    {
        "nps": {
            "transport": "sse",
            "url": NPS_MCP_URL,
        }
    }
)

# Get tools from MCP server
tools = await client.get_tools()
print(f"Loaded {len(tools)} tools from MCP server:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")

# Cell 3: Build the LangGraph ReAct Agent
llm = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True,
    )

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)

# Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to tools if tool call, else END
)
builder.add_edge("tools", "assistant")

# Compile
nps_agent = builder.compile()

# Visualize
# display(Image(nps_agent.get_graph(xray=True).draw_mermaid_png()))

# Cell 4: Run the agent
messages = [
    HumanMessage(content="Tell me about some parks in Rhode Island, and let me know if there are any upcoming events at them.")
]

response = await nps_agent.ainvoke({"messages": messages})

# Print final response
print("\n--- Final Response ---")
print(response["messages"][-1].content)

