"""
企业技术支持智研知识库（PRD-KB）—— API 路由模块

提供健康检查、同步对话与流式对话三个接口，统一使用 Agent 自主检索模式。
"""
import json
from typing import Any
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from api.schemas import ChatRequest, ChatResponse, HealthResponse
from api.dependencies import get_app_state, AppState
from utils.logger import logger

router = APIRouter()

TOOL_LABEL_MAP = {
    "knowledge_base_search": "检索知识库",
    "log_analysis": "分析日志",
    "count_text_characters": "统计字数",
}


@router.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    """健康检查接口"""
    return HealthResponse(
        status="ok",
        agent_mode=True,
        knowledge_base_ready=state.ready
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """同步对话接口：Agent 自主检索模式

    使用 Agent 图处理用户问题，Agent 自主判断是否需要调用检索工具。
    """
    if not state.ready:
        return ChatResponse(
            answer="服务正在初始化中，请稍后重试",
            session_id=req.session_id
        )

    session_config: Any = {"configurable": {"thread_id": req.session_id}}

    try:
        resp = state.agent_graph.invoke(
            {"messages": [HumanMessage(content=req.question)]},
            config=session_config
        )
        answer = resp["messages"][-1].content
        logger.info(f"会话 [{req.session_id}] 问答完成")
        return ChatResponse(answer=answer, session_id=req.session_id)
    except Exception as e:
        logger.error(f"会话 [{req.session_id}] 处理异常: {str(e)}")
        return ChatResponse(
            answer="抱歉，处理问题时发生错误，请重新提问",
            session_id=req.session_id
        )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """流式对话接口：SSE 逐 token 推送，Agent 自主检索模式

    通过 astream_events 监听 LangGraph 执行事件，向前端推送工具调用过程与生成内容。
    """
    if not state.ready:
        async def error_gen():
            yield f"data: {json.dumps({'token': '服务正在初始化中，请稍后重试'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session_config: Any = {"configurable": {"thread_id": req.session_id}}
    logger.info(f"流式会话 [{req.session_id}] 开始")

    async def event_generator():
        thinking_done = False
        try:
            async for event in state.agent_graph.astream_events(
                {"messages": [HumanMessage(content=req.question)]},
                config=session_config,
                version="v2"
            ):
                kind = event["event"]
                if kind == "on_chain_start" and event["name"] == "retrieve":
                    yield f"data: {json.dumps({'thinking': {'text': '正在检索知识库...'}})}\n\n"
                elif kind == "on_chain_end" and event["name"] == "retrieve":
                    yield f"data: {json.dumps({'thinking': {'text': '检索完成，正在生成回答...'}})}\n\n"
                elif kind == "on_tool_start":
                    name = event["name"]
                    label = TOOL_LABEL_MAP.get(name, name)
                    yield f"data: {json.dumps({'thinking': {'text': f'正在{label}...'}})}\n\n"
                elif kind == "on_tool_end":
                    name = event["name"]
                    label = TOOL_LABEL_MAP.get(name, name)
                    yield f"data: {json.dumps({'thinking': {'text': f'{label}完成'}})}\n\n"
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        if not thinking_done:
                            thinking_done = True
                            yield f"data: {json.dumps({'thinking': {'done': True}})}\n\n"
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"流式会话 [{req.session_id}] 异常: {str(e)}")
            yield f"data: {json.dumps({'error': '处理问题时发生错误，请重新提问'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")