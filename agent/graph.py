"""
agent/graph.py — LangGraph multi-step reasoning graph (Cluster 03: Sovereign Developer OS)

Stateful LangGraph agent that handles multi-turn developer queries.
Can retrieve code context, execute code, read/write files, and suggest edits.

SDKs: LangGraph, Ollama, LlamaIndex
"""
import json
import logging
import time
from typing import TypedDict, Annotated, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from agent.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Sovereign, a local AI pair programmer with deep knowledge of the user's codebase.
You have access to these tools:
- read_file(path, start_line, end_line): Read a file or slice of it
- write_file(path, content): Write content to a file
- execute_python(code): Run Python code and return output
- run_shell(command): Run a safe shell command (ls, find, grep, git, etc.)
- list_files(directory, extensions): List files in a directory
- search_codebase(query): Search the indexed codebase semantically

Rules:
- Always retrieve relevant context before answering code questions
- When suggesting code changes, show the exact diff or replacement
- If you use a tool, explain what you found before answering
- Be concise. No unnecessary preamble.
- Respond in JSON when calling tools: {"tool": "name", "args": {...}}
- Respond in plain text when giving the final answer.
"""


class DevState(TypedDict):
    session_id: str
    messages: Annotated[list, operator.add]
    context: str
    tool_calls: list
    final_answer: Optional[str]
    steps: int
    error: Optional[str]


class SovereignAgent:
    """
    LangGraph-based developer assistant.
    Multi-step: retrieve → reason → tool use → synthesize answer.
    """

    def __init__(self, config: dict, rag=None, store=None):
        self._cfg = config
        self._rag = rag
        self._store = store
        self._model = None
        self._graph = None
        self._max_steps = config.get("agent", {}).get("max_steps", 10)

    def load(self) -> None:
        ollama_cfg = self._cfg.get("ollama", {})
        self._model = ChatOllama(
            model=ollama_cfg.get("model", "llama3"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            temperature=self._cfg.get("agent", {}).get("temperature", 0.2),
        )
        self._graph = self._build_graph()
        logger.info("SovereignAgent loaded")

    def _build_graph(self):
        g = StateGraph(DevState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("reason", self._reason_node)
        g.add_node("tool_exec", self._tool_exec_node)
        g.add_node("synthesize", self._synthesize_node)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "reason")
        g.add_conditional_edges("reason", self._should_use_tool,
                                {"tool": "tool_exec", "answer": "synthesize"})
        g.add_conditional_edges("tool_exec", self._should_continue,
                                {"continue": "reason", "done": "synthesize"})
        g.add_edge("synthesize", END)
        return g.compile()

    def _retrieve_node(self, state: DevState) -> DevState:
        last_user = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        context = ""
        if self._rag:
            try:
                result = self._rag.query(last_user)
                context = f"Relevant codebase context:\n{result['answer']}\n\nSource files: {result['source_files']}"
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)
        return {**state, "context": context}

    def _reason_node(self, state: DevState) -> DevState:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        if state["context"]:
            msgs.append(SystemMessage(content=f"Codebase context:\n{state['context']}"))
        msgs.extend(state["messages"])
        try:
            response = self._model.invoke(msgs)
            return {**state, "messages": [AIMessage(content=response.content)],
                    "steps": state["steps"] + 1, "error": None}
        except Exception as e:
            logger.error("LLM error: %s", e)
            return {**state, "error": str(e), "steps": state["steps"] + 1}

    def _tool_exec_node(self, state: DevState) -> DevState:
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else ""
        try:
            parsed = json.loads(content)
            tool_name = parsed.get("tool")
            args = parsed.get("args", {})
            if tool_name in TOOL_REGISTRY:
                if tool_name == "search_codebase":
                    result = TOOL_REGISTRY[tool_name](self._rag, **args)
                else:
                    result = TOOL_REGISTRY[tool_name](**args)
                tool_msg = ToolMessage(content=json.dumps(result)[:4000], tool_call_id=tool_name)
                calls = state["tool_calls"] + [{"tool": tool_name, "args": args, "result": result}]
                return {**state, "messages": [tool_msg], "tool_calls": calls}
            else:
                return {**state, "messages": [ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id="unknown")]}
        except (json.JSONDecodeError, Exception) as e:
            return {**state, "messages": [ToolMessage(content=f"Tool parse error: {e}", tool_call_id="error")]}

    def _synthesize_node(self, state: DevState) -> DevState:
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        try:
            json.loads(content)
            # still looks like a tool call — ask for plain answer
            msgs = list(state["messages"]) + [HumanMessage(content="Now give your final answer in plain text.")]
            response = self._model.invoke(msgs)
            answer = response.content
        except (json.JSONDecodeError, ValueError):
            answer = content
        return {**state, "final_answer": answer}

    def _should_use_tool(self, state: DevState) -> str:
        if state.get("error") or state["steps"] >= self._max_steps:
            return "answer"
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else ""
        try:
            parsed = json.loads(content)
            if "tool" in parsed:
                return "tool"
        except Exception:
            pass
        return "answer"

    def _should_continue(self, state: DevState) -> str:
        if state["steps"] >= self._max_steps:
            return "done"
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else ""
        try:
            parsed = json.loads(content)
            if "tool" in parsed:
                return "continue"
        except Exception:
            pass
        return "done"

    def chat(self, session_id: str, message: str) -> dict:
        """Single-turn chat. Returns {answer, tool_calls, steps, latency_ms}."""
        if not self._graph:
            raise RuntimeError("Agent not loaded. Call load() first.")
        t0 = time.perf_counter()
        initial: DevState = {
            "session_id": session_id,
            "messages": [HumanMessage(content=message)],
            "context": "",
            "tool_calls": [],
            "final_answer": None,
            "steps": 0,
            "error": None,
        }
        result = self._graph.invoke(initial)
        latency_ms = (time.perf_counter() - t0) * 1000

        if self._store:
            self._store.add_message(session_id, "user", message)
            self._store.add_message(session_id, "assistant",
                                    result.get("final_answer", ""), latency_ms=latency_ms)
        return {
            "answer": result.get("final_answer", result.get("error", "No answer")),
            "tool_calls": result.get("tool_calls", []),
            "steps": result.get("steps", 0),
            "latency_ms": round(latency_ms, 1),
        }
