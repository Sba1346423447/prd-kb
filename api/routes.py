"""
企业技术支持智研知识库（PRD-KB）—— API 路由模块

提供健康检查、同步对话、流式对话与会话管理接口，
对话接口要求登录并校验会话归属，支持 Agent 自主检索与 Pure 固定流水线直出两种模式。
"""
import json
import uuid
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionCreateResponse,
    SessionListResponse,
)
from api.dependencies import get_app_state, AppState, require
from core.models import User
from core.mode_router import route_mode
from core.permissions import Permission
from core.multimodal import (
    build_human_content,
    content_to_text,
    parse_stored_content,
    serialize_multimodal_content,
)
from core.tools import retrieve_context
from prompts import get_direct_system_prompt
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


def _build_pure_messages(state: AppState, user_content: Any, history_messages: List[Any]) -> List[Any]:
    """Pure 直出模式消息构建：单次检索并将上下文注入系统提示词

    检索 query 取用户输入的文本部分（多模态内容经 content_to_text 提取），
    消息序列为 [系统提示词(含检索上下文)] + 对话历史(含本次提问)，
    构建完成后直接交给 LLM 生成，不经过 Agent 工具决策循环。

    Args:
        state: 全局应用状态（提供检索器、检索配置与 Reranker）
        user_content: 本次用户输入内容（字符串或多模态内容块列表）
        history_messages: 已含本次提问的消息列表

    Returns:
        可直接交给 LLM 的完整消息列表
    """
    query = content_to_text(user_content)
    context = retrieve_context(state.retriever, state.retrieval_config, state.reranker, query)
    return [SystemMessage(content=get_direct_system_prompt(context))] + history_messages


def _resolve_mode(req: ChatRequest) -> str:
    """解析本次请求最终使用的检索模式

    Auto 模式经规则路由（core/mode_router.route_mode）判定：
    明确的知识库简单问题→ Pure 直出；其余（闲聊/域外/工具/复杂/模糊）→ Agent 自由回答。
    不依赖检索裁决：向量检索对任何 query 都返回 top-k，"是否非空"无法作为有无内容依据。

    Args:
        req: 对话请求体

    Returns:
        "agent" 或 "pure"：auto 模式经规则路由后返回实际模式
    """
    if req.mode != "auto":
        return req.mode
    resolved = route_mode(req.question, bool(req.images))
    logger.info(f"Auto 模式路由：query='{req.question}' 图片={bool(req.images)} -> {resolved}")
    return resolved


def _check_session_ownership(state: AppState, session_id: str, user: User) -> None:
    """校验会话归属当前用户，防止越权访问他人会话

    Args:
        state: 全局应用状态
        session_id: 会话 ID
        user: 当前登录用户

    Raises:
        HTTPException: 会话不存在或不归属当前用户返回 403
    """
    if not state.session_store.owns_session(session_id, user.id):
        logger.warning(f"用户 [{user.username}] 越权访问会话 [{session_id}] 被拒绝")
        raise HTTPException(status_code=403, detail="会话不存在或无权访问")


@router.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    """健康检查接口（无鉴权，供探活使用）"""
    return HealthResponse(
        status="ok",
        agent_mode=True,
        knowledge_base_ready=state.ready
    )


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    state: AppState = Depends(get_app_state),
    user: User = Depends(require(Permission.SESSION_MANAGE)),
):
    """创建会话接口：服务端生成会话 ID 并绑定当前用户

    Args:
        state: 全局应用状态
        user: 当前登录用户（需 session.manage 权限位）

    Returns:
        SessionCreateResponse: 新会话 ID 与标题
    """
    session_id = state.session_store.create_session(user.id)
    logger.info(f"用户 [{user.username}] 创建会话 [{session_id}]")
    return SessionCreateResponse(session_id=session_id)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    state: AppState = Depends(get_app_state),
    user: User = Depends(require(Permission.SESSION_MANAGE)),
):
    """查询当前用户的全部会话，按创建时间倒序"""
    return SessionListResponse(sessions=state.session_store.list_sessions(user.id))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    state: AppState = Depends(get_app_state),
    user: User = Depends(require(Permission.CHAT)),
):
    """同步对话接口：支持 Agent 自主检索与 Pure 固定流水线直出两种模式

    Agent 模式使用 Agent 图处理问题，自主判断是否调用检索工具；
    Pure 模式固定执行单次检索后将上下文直接交给 LLM 生成。
    每次请求从持久化历史重建对话上下文，问答完成后写入会话历史。
    """
    _check_session_ownership(state, req.session_id, user)
    if not state.ready:
        return ChatResponse(
            answer="服务正在初始化中，请稍后重试",
            session_id=req.session_id
        )

    # 每次请求使用独立线程，从 MySQL 历史重建上下文，避免 checkpoint 累积与并发竞态
    history_messages = _build_history_messages(state.session_store.load_history(req.session_id))
    user_content = _build_user_content(req)
    history_messages.append(HumanMessage(content=user_content))

    try:
        resolved_mode = _resolve_mode(req)
        if resolved_mode == "pure":
            messages = _build_pure_messages(state, user_content, history_messages)
            resp = state.llm.invoke(messages)
            answer = content_to_text(resp.content)
        else:
            session_config: Any = {"configurable": {"thread_id": f"{req.session_id}_{uuid.uuid4().hex}"}}
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
        logger.info(f"会话 [{req.session_id}] 问答完成（mode={req.mode}→{resolved_mode}）")
        return ChatResponse(answer=answer, session_id=req.session_id)
    except Exception as e:
        logger.error(f"会话 [{req.session_id}] 处理异常: {str(e)}")
        return ChatResponse(
            answer="抱歉，处理问题时发生错误，请重新提问",
            session_id=req.session_id
        )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    state: AppState = Depends(get_app_state),
    user: User = Depends(require(Permission.CHAT)),
):
    """流式对话接口：SSE 逐 token 推送，支持 Agent 与 Pure 两种模式

    Agent 模式通过 astream_events 监听 LangGraph 执行事件，推送工具调用过程与生成内容；
    Pure 模式固定单次检索后直接流式生成，推送检索进度与生成内容。
    每次请求从持久化历史重建对话上下文，问答完成后写入会话历史。
    """
    _check_session_ownership(state, req.session_id, user)
    if not state.ready:
        async def error_gen():
            yield f"data: {json.dumps({'token': '服务正在初始化中，请稍后重试'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    history_messages = _build_history_messages(state.session_store.load_history(req.session_id))
    user_content = _build_user_content(req)
    history_messages.append(HumanMessage(content=user_content))
    logger.info(f"流式会话 [{req.session_id}] 开始（mode={req.mode}）")

    async def event_generator():
        thinking_done = False
        resolved_mode = _resolve_mode(req)
        try:
            if resolved_mode == "pure":
                yield f"data: {json.dumps({'thinking': {'text': '正在检索知识库...'}})}\n\n"
                messages = _build_pure_messages(state, user_content, history_messages)
                yield f"data: {json.dumps({'thinking': {'text': '检索知识库完成'}})}\n\n"
                thinking_done = True
                yield f"data: {json.dumps({'thinking': {'done': True}})}\n\n"
                answer = ""
                async for chunk in state.llm.astream(messages):
                    token_text = content_to_text(chunk.content)
                    if token_text:
                        answer += token_text
                        yield f"data: {json.dumps({'token': token_text})}\n\n"
            else:
                session_config: Any = {"configurable": {"thread_id": f"{req.session_id}_{uuid.uuid4().hex}"}}
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
