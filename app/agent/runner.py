from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol, TypedDict

from app.llm.openvino_qwen import OpenVINOQwen
from app.tools.document_create import create_document
from app.tools.file_search import file_search


@dataclass
class AgentResult:
    message: str
    data: dict | list | None = None


class AgentState(TypedDict, total=False):
    prompt: str
    decision: dict[str, Any]
    selected_tool: str | None
    tool_input: dict[str, Any] | None
    tool_output: dict[str, Any] | list[Any] | None
    message: str
    fallback_reason: str | None


class Planner(Protocol):
    def plan(self, user_prompt: str) -> dict[str, Any]:
        ...


class LLMToolPlanner:
    """Use an LLM to choose one tool and generate its arguments as JSON."""

    def __init__(self, llm: OpenVINOQwen | None = None) -> None:
        self.llm = llm or OpenVINOQwen()

    def plan(self, user_prompt: str) -> dict[str, Any]:
        prompt = self._build_prompt(user_prompt)
        raw = self.llm.invoke(prompt)
        payload = self._extract_json(raw)

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner returned invalid JSON: {raw}") from exc

        action = data.get("action")
        if action not in {"use_tool", "respond"}:
            raise ValueError(f"Planner action must be 'use_tool' or 'respond': {data}")

        if action == "respond":
            return {"action": "respond", "answer": str(data.get("answer", ""))}

        tool_name = data.get("tool_name")
        if tool_name not in {"file_search_tool", "document_create_tool"}:
            raise ValueError(f"Unsupported tool_name from planner: {tool_name}")

        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Planner arguments must be an object")

        return {"action": "use_tool", "tool_name": tool_name, "arguments": arguments}

    def _build_prompt(self, user_prompt: str) -> str:
        return (
            "You are a tool planner. Output JSON only. No markdown.\\n"
            "Choose exactly one action.\\n"
            "Allowed actions:\\n"
            "1) use_tool -> choose one tool and arguments\\n"
            "2) respond -> direct answer when no tool is needed\\n"
            "Tools:\\n"
            "- file_search_tool arguments: root_path(str), pattern(str), max_results(int 1..200)\\n"
            "- document_create_tool arguments: title(str), content(str), format('md'|'txt'), output_dir(str|null)\\n"
            "If user mentions whole computer, use root_path='this_pc'.\\n"
            "If user asks for python files, prefer pattern='*.py'.\\n"
            "JSON schema:\\n"
            "{\"action\":\"use_tool\",\"tool_name\":\"file_search_tool\",\"arguments\":{...}}\\n"
            "or\\n"
            "{\"action\":\"respond\",\"answer\":\"...\"}\\n"
            f"User request: {user_prompt}"
        )

    def _extract_json(self, text: str) -> str:
        if not text:
            raise ValueError("Planner returned empty response")

        fenced = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if obj:
            return obj.group(0)

        raise ValueError(f"No JSON object found in planner output: {text}")


class _InternalCompiledGraph:
    """Fallback graph executor used only when langgraph is unavailable."""

    def __init__(self, agent: MVPAgent) -> None:
        self.agent = agent

    def invoke(self, initial_state: AgentState) -> AgentState:
        state: AgentState = dict(initial_state)
        state.update(self.agent._node_plan(state))
        route = self.agent._route_from_plan(state)
        if route == "use_tool":
            state.update(self.agent._node_execute_tool(state))
        else:
            state.update(self.agent._node_respond(state))
        state.update(self.agent._node_finalize(state))
        return state


class MVPAgent:
    """MVP agent with one-turn-one-tool LangGraph flow."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or LLMToolPlanner()
        self._graph_backend = "langgraph"
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            self._graph_backend = "internal"
            return _InternalCompiledGraph(self)

        graph = StateGraph(AgentState)
        graph.add_node("plan", self._node_plan)
        graph.add_node("execute_tool", self._node_execute_tool)
        graph.add_node("respond", self._node_respond)
        graph.add_node("finalize", self._node_finalize)
        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            self._route_from_plan,
            {
                "use_tool": "execute_tool",
                "respond": "respond",
            },
        )
        graph.add_edge("execute_tool", "finalize")
        graph.add_edge("respond", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def create_document(self, title: str, content: str, format: str = "md", output_dir: str | None = None) -> AgentResult:
        data = create_document(title=title, content=content, format=format, output_dir=output_dir)
        return AgentResult(message=f"Document created: {data['saved_path']}", data=data)

    def search_files(self, root_path: str = ".", pattern: str = "*.md", max_results: int = 20) -> AgentResult:
        data = file_search(root_path=root_path, pattern=pattern, max_results=max_results)
        return AgentResult(message=f"Found {len(data)} file(s)", data=data)

    def run_prompt(self, prompt: str) -> AgentResult:
        state = self._graph.invoke({"prompt": prompt})
        return AgentResult(
            message=state["message"],
            data={
                "selected_tool": state.get("selected_tool"),
                "tool_input": state.get("tool_input"),
                "tool_output": state.get("tool_output"),
                "fallback_reason": state.get("fallback_reason"),
                "graph_backend": self._graph_backend,
            },
        )

    def _node_plan(self, state: AgentState) -> AgentState:
        prompt = state.get("prompt", "")
        fallback_reason = None
        try:
            decision = self.planner.plan(prompt)
        except (ValueError, RuntimeError) as exc:
            fallback_reason = str(exc)
            decision = self._fallback_plan(prompt)

        return {"decision": decision, "fallback_reason": fallback_reason}

    def _route_from_plan(self, state: AgentState) -> str:
        decision = state.get("decision", {})
        if decision.get("action") == "use_tool":
            return "use_tool"
        return "respond"

    def _node_execute_tool(self, state: AgentState) -> AgentState:
        decision = state.get("decision", {})
        tool_name = str(decision.get("tool_name", ""))
        args = decision.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        if tool_name == "file_search_tool":
            params = self._normalize_search_args(args)
            tool_result = self.search_files(**params)
        elif tool_name == "document_create_tool":
            params = self._normalize_create_args(args)
            tool_result = self.create_document(**params)
        else:
            params = {}
            tool_result = AgentResult(message="Unsupported tool", data=None)

        return {
            "selected_tool": tool_name,
            "tool_input": params,
            "tool_output": tool_result.data,
            "message": tool_result.message,
        }

    def _node_respond(self, state: AgentState) -> AgentState:
        decision = state.get("decision", {})
        answer = str(decision.get("answer", "")).strip() or "No action needed."
        return {
            "selected_tool": None,
            "tool_input": None,
            "tool_output": None,
            "message": answer,
        }

    def _node_finalize(self, state: AgentState) -> AgentState:
        message = state.get("message", "")
        selected = state.get("selected_tool")
        fallback_reason = state.get("fallback_reason")

        if selected:
            message = f"Auto selected: {selected}. {message}"
            if fallback_reason:
                message = f"{message} (fallback planner used)"

        return {"message": message}

    def _fallback_plan(self, prompt: str) -> dict[str, Any]:
        tool_name = self._select_tool(prompt)
        if tool_name == "file_search_tool":
            arguments = self._extract_search_args(prompt)
        else:
            arguments = self._extract_create_args(prompt)
        return {"action": "use_tool", "tool_name": tool_name, "arguments": arguments}

    def _select_tool(self, prompt: str) -> str:
        text = prompt.lower()
        create_keywords = ["作成", "保存", "まとめ", "書いて", "文書", "ドキュメント", "メモ", "議事録", "report"]
        search_keywords = ["検索", "探し", "見つけ", "一覧", "どこ", "find", "search", "ファイル", "python", ".py"]

        create_score = sum(1 for k in create_keywords if k in text)
        search_score = sum(1 for k in search_keywords if k in text)
        return "file_search_tool" if search_score > create_score else "document_create_tool"

    def _extract_search_args(self, prompt: str) -> dict[str, Any]:
        lower = prompt.lower()
        pattern_match = re.search(r"(\*\.[a-zA-Z0-9]+)", prompt)
        if pattern_match:
            pattern = pattern_match.group(1)
        elif ".py" in lower or "python" in lower:
            pattern = "*.py"
        elif "txt" in lower:
            pattern = "*.txt"
        elif "md" in lower or "markdown" in lower:
            pattern = "*.md"
        else:
            pattern = "*"

        root_path = "."
        if any(token in prompt for token in ["このコンピュータ", "PC全体", "pc全体"]):
            root_path = "this_pc"
        else:
            path_match = re.search(r"([A-Za-z0-9_.\\/-]+)\s*(?:フォルダ|folder|配下|以下)", prompt)
            if path_match:
                root_path = path_match.group(1)

        max_results = 20
        max_match = re.search(r"(\d+)\s*(?:件|個|results?)", lower)
        if max_match:
            max_results = max(1, min(200, int(max_match.group(1))))

        return {"root_path": root_path, "pattern": pattern, "max_results": max_results}

    def _extract_create_args(self, prompt: str) -> dict[str, Any]:
        lower = prompt.lower()
        fmt = "txt" if "txt" in lower else "md"

        output_dir = None
        path_match = re.search(r"([A-Za-z0-9_.\\/-]+)\s*(?:フォルダ|folder|配下|以下)", prompt)
        if path_match:
            output_dir = path_match.group(1)

        quoted = re.findall(r"[\"'「](.*?)[\"'」]", prompt)
        if len(quoted) >= 2:
            title = quoted[0].strip() or "Agent_Note"
            content = quoted[1].strip() or prompt.strip()
        elif len(quoted) == 1:
            title = quoted[0].strip() or "Agent_Note"
            content = prompt.strip()
        else:
            title = self._derive_title(prompt)
            content = prompt.strip()

        return {"title": title, "content": content, "format": fmt, "output_dir": output_dir}

    def _derive_title(self, prompt: str) -> str:
        cleaned = re.sub(r"\s+", " ", prompt).strip()
        return cleaned[:40] if cleaned else "Agent_Note"

    def _normalize_search_args(self, args: dict[str, Any]) -> dict[str, str | int]:
        root_path = str(args.get("root_path", "."))
        pattern = str(args.get("pattern", "*.md"))

        raw_max = args.get("max_results", 20)
        try:
            max_results = int(raw_max)
        except (TypeError, ValueError):
            max_results = 20
        max_results = max(1, min(200, max_results))

        return {"root_path": root_path, "pattern": pattern, "max_results": max_results}

    def _normalize_create_args(self, args: dict[str, Any]) -> dict[str, str | None]:
        title = str(args.get("title", "Agent_Note")).strip() or "Agent_Note"
        content = str(args.get("content", "")).strip() or "(empty)"
        fmt = str(args.get("format", "md")).lower().strip()
        if fmt not in {"md", "txt"}:
            fmt = "md"

        output_dir = args.get("output_dir")
        if output_dir is None:
            out = None
        else:
            out = str(output_dir)

        return {"title": title, "content": content, "format": fmt, "output_dir": out}
