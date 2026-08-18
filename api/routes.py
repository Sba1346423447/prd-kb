"""
企业技术支持智研知识库（PRD-KB）—— API 路由模块

提供健康检查、同步对话与流式对话三个接口，统一使用 Agent 自主检索模式。
"""
import json
import uuid
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from api.schemas import ChatRequest, ChatResponse, HealthResponse
from api.dependencies import get_app_state, AppState
from core.multimodal import (
    build_human_content,
    content_to_text,
    parse_stored_content,
    serialize_multimodal_content,
)
from utils.logger import logger

router = APIRouter()

MAX_IMAGES_PER_REQUEST = 4

TOOL_LABEL_MAP = {
    "knowledge_base_search": "检索知识库",
    "log_analysis": "分析日志",
    "count_text_characters": "统计字数",
}


def _build_user_content(req: ChatRequest) -> Any:
    if not req.images:
        return req.question
    _validate_images(req.images)
    return build_human_content(req.question, req.images)


def _validate_images(images: List[str]) -> None:
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {MAX_IMAGES_PER_REQUEST} 张图片",
        )
    for image in images:
        if not (isinstance(image, str) and image.startswith("data:image/")):
            raise HTTPException(
                status_code=400,
                detail="图片必须是 data:image/ 开头的 data URL",
            )


def _build_history_messages(history: List[tuple]) -> List[Any]:
    """将持久化的对话历史转换为 LangChain 消息列表

    Args:
        history: (role, content) 元组列表，role 为 human 或 ai

    Returns:
        按时间顺序排列的 LangChain 消息列表
    """
    messages = []
    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=parse_stored_content(content)))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


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
    每次请求从持久化历史重建对话上下文，问答完成后写入会话历史。
    """
    if not state.ready:
        return ChatResponse(
            answer="服务正在初始化中，请稍后重试",
            session_id=req.session_id
        )

    # 每次请求使用独立线程，从 SQLite 历史重建上下文，避免 checkpoint 累积与并发竞态
    history_messages = _build_history_messages(state.session_store.load_history(req.session_id))
    user_content = _build_user_content(req)
    history_messages.append(HumanMessage(content=user_content))
    session_config: Any = {"configurable": {"thread_id": f"{req.session_id}_{uuid.uuid4().hex}"}}

    try:
        resp = state.agent_graph.invoke(
            {"messages": history_messages},
            config=session_config
        )
        answer = content_to_text(resp["messages"][-1].content)
        state.session_store.save_message(
            req.session_id,
            "human",
            serialize_multimodal_content(user_content),
        )
        state.session_store.save_message(req.session_id, "ai", answer)
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
    每次请求从持久化历史重建对话上下文，问答完成后写入会话历史。
    """
    if not state.ready:
        async def error_gen():
            yield f"data: {json.dumps({'token': '服务正在初始化中，请稍后重试'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    history_messages = _build_history_messages(state.session_store.load_history(req.session_id))
    user_content = _build_user_content(req)
    history_messages.append(HumanMessage(content=user_content))
    session_config: Any = {"configurable": {"thread_id": f"{req.session_id}_{uuid.uuid4().hex}"}}
    logger.info(f"流式会话 [{req.session_id}] 开始")

    async def event_generator():
        thinking_done = False
        try:
            async for event in state.agent_graph.astream_events(
                {"messages": history_messages},
                config=session_config,
                version="v2"
            ):
                kind = event["event"]
                if kind == "on_tool_start":
                    name = event["name"]
                    label = TOOL_LABEL_MAP.get(name, name)
                    yield f"data: {json.dumps({'thinking': {'text': f'正在{label}...'}})}\n\n"
                elif kind == "on_tool_end":
                    name = event["name"]
                    label = TOOL_LABEL_MAP.get(name, name)
                    yield f"data: {json.dumps({'thinking': {'text': f'{label}完成'}})}\n\n"
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token_text = content_to_text(chunk.content)
                    if token_text:
                        if not thinking_done:
                            thinking_done = True
                            yield f"data: {json.dumps({'thinking': {'done': True}})}\n\n"
                        yield f"data: {json.dumps({'token': token_text})}\n\n"

            snapshot = state.agent_graph.get_state(session_config)
            answer = content_to_text(snapshot.values["messages"][-1].content)
            state.session_store.save_message(
                req.session_id,
                "human",
                serialize_multimodal_content(user_content),
            )
            state.session_store.save_message(req.session_id, "ai", answer)

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"流式会话 [{req.session_id}] 异常: {str(e)}")
            yield f"data: {json.dumps({'error': '处理问题时发生错误，请重新提问'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
