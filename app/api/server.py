from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.runner import MVPAgent
from app.llm.openvino_qwen import OpenVINOQwen


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)


class CreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    format: str = Field(default="md", pattern="^(md|txt)$")
    output_dir: str | None = None


class SearchRequest(BaseModel):
    root_path: str = "."
    pattern: str = "*.md"
    max_results: int = Field(default=20, ge=1, le=200)


class AgentResponse(BaseModel):
    message: str
    data: dict[str, Any] | list[Any] | None


def get_agent() -> MVPAgent:
    return MVPAgent()


def create_app() -> FastAPI:
    app = FastAPI(title="OpenVINO LangGraph Agent API", version="1.0.0")

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/agent/chat", response_model=AgentResponse)
    def chat(req: ChatRequest, agent: MVPAgent = Depends(get_agent)) -> AgentResponse:
        try:
            result = agent.run_prompt(req.prompt)
            return AgentResponse(message=result.message, data=result.data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/tools/create", response_model=AgentResponse)
    def create_doc(req: CreateRequest, agent: MVPAgent = Depends(get_agent)) -> AgentResponse:
        try:
            result = agent.create_document(
                title=req.title,
                content=req.content,
                format=req.format,
                output_dir=req.output_dir,
            )
            return AgentResponse(message=result.message, data=result.data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/tools/search", response_model=AgentResponse)
    def search(req: SearchRequest, agent: MVPAgent = Depends(get_agent)) -> AgentResponse:
        try:
            result = agent.search_files(
                root_path=req.root_path,
                pattern=req.pattern,
                max_results=req.max_results,
            )
            return AgentResponse(message=result.message, data=result.data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/model/download")
    def download_model() -> dict[str, str]:
        try:
            source = OpenVINOQwen().ensure_model_downloaded()
            return {"message": "model_ready", "model_source": source}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
