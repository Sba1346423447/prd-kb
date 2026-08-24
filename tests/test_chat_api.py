"""
批次3：对话与会话接口测试

覆盖未登录拒绝、会话创建与列表隔离、跨用户越权 403、
本人会话进入业务分支（ready=False 初始化中响应），
以及 Pure 直出模式的参数校验与固定流水线全链路。
"""
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage

from api.dependencies import app_state


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestAuthGate:
    def test_chat_without_token_rejected(self, client):
        resp = client.post("/chat", json={"question": "hi", "session_id": "s"})
        assert resp.status_code == 401

    def test_chat_stream_without_token_rejected(self, client):
        resp = client.post("/chat/stream", json={"question": "hi", "session_id": "s"})
        assert resp.status_code == 401

    def test_chat_with_garbage_token_rejected(self, client):
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": "s"},
            headers={"Authorization": "Bearer bad.token.here"},
        )
        assert resp.status_code == 401


class TestSessionAPI:
    def test_create_session_without_token_rejected(self, client):
        assert client.post("/sessions").status_code == 401

    def test_create_session_returns_server_generated_id(self, client, tokens):
        resp = client.post("/sessions", headers=auth(tokens["alice"]))
        assert resp.status_code == 200
        assert resp.json()["session_id"]

    def test_list_sessions_isolated_by_user(self, client, tokens):
        """会话列表只含本人会话，用户间互相不可见"""
        client.post("/sessions", headers=auth(tokens["alice"]))
        client.post("/sessions", headers=auth(tokens["alice"]))
        client.post("/sessions", headers=auth(tokens["bob"]))

        alice_sessions = client.get("/sessions", headers=auth(tokens["alice"])).json()["sessions"]
        bob_sessions = client.get("/sessions", headers=auth(tokens["bob"])).json()["sessions"]

        assert len(alice_sessions) == 2
        assert len(bob_sessions) == 1


class TestSessionOwnership:
    def test_chat_with_foreign_session_rejected(self, client, tokens):
        """bob 用 alice 的会话 ID 提问 → 403（水平越权防护）"""
        session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": session_id},
            headers=auth(tokens["bob"]),
        )
        assert resp.status_code == 403

    def test_chat_stream_with_foreign_session_rejected(self, client, tokens):
        session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
        resp = client.post(
            "/chat/stream",
            json={"question": "hi", "session_id": session_id},
            headers=auth(tokens["bob"]),
        )
        assert resp.status_code == 403

    def test_chat_with_nonexistent_session_rejected(self, client, tokens):
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": "no-such-session"},
            headers=auth(tokens["alice"]),
        )
        assert resp.status_code == 403

    def test_owner_session_passes_gate_into_business(self, client, tokens):
        """本人会话通过归属校验，进入业务分支（ready=False → 初始化中提示）"""
        session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": session_id},
            headers=auth(tokens["alice"]),
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id
        assert "初始化" in resp.json()["answer"]


class FakeRetriever:
    """Pure 模式测试替身：记录调用并返回固定文档"""

    def __init__(self):
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return [Document(
            page_content="部署文档：使用 docker compose 一键启动全栈服务",
            metadata={"file_name": "deploy.md", "chunk_index": 0},
        )]


class FakeLLM:
    """Pure 模式测试替身：记录收到的消息，同步与流式返回固定内容"""

    def __init__(self, chunks=("根据知识库：docker compose 一键启动",)):
        self.received = None
        self._chunks = chunks

    def invoke(self, messages):
        self.received = messages
        return AIMessage(content="根据知识库：docker compose 一键启动")

    async def astream(self, messages):
        self.received = messages
        for token in self._chunks:
            yield AIMessageChunk(content=token)


class TestPureMode:
    def test_invalid_mode_rejected(self, client, tokens):
        """非法的 mode 值应被 Literal 校验拒绝为 422"""
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": "s", "mode": "xyz"},
            headers=auth(tokens["alice"]),
        )
        assert resp.status_code == 422

    def test_pure_mode_not_ready_returns_init_message(self, client, tokens):
        """ready=False 时 Pure 模式与 Agent 模式行为一致，返回初始化中提示"""
        session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": session_id, "mode": "pure"},
            headers=auth(tokens["alice"]),
        )
        assert resp.status_code == 200
        assert "初始化" in resp.json()["answer"]

    def test_pure_mode_with_foreign_session_rejected(self, client, tokens):
        """Pure 模式同样受会话归属校验约束，越权返回 403"""
        session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
        resp = client.post(
            "/chat",
            json={"question": "hi", "session_id": session_id, "mode": "pure"},
            headers=auth(tokens["bob"]),
        )
        assert resp.status_code == 403

    def test_pure_mode_full_pipeline(self, client, tokens):
        """Pure 固定流水线全链路：单次检索 -> 上下文注入系统提示词 -> LLM 生成 -> 历史落库"""
        retriever = FakeRetriever()
        llm = FakeLLM()
        original = {
            "ready": app_state.ready, "retriever": app_state.retriever,
            "retrieval_config": app_state.retrieval_config,
            "reranker": app_state.reranker, "llm": app_state.llm,
        }
        app_state.ready = True
        app_state.retriever = retriever
        app_state.retrieval_config = {"enable_rerank": False, "max_result_docs": 5, "max_result_chars": 6000}
        app_state.reranker = None
        app_state.llm = llm
        try:
            session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
            resp = client.post(
                "/chat",
                json={"question": "怎么部署？", "session_id": session_id, "mode": "pure"},
                headers=auth(tokens["alice"]),
            )
            assert resp.status_code == 200
            assert "docker compose" in resp.json()["answer"]

            # 固定流水线：恰好一次检索，query 为用户提问文本
            assert retriever.calls == ["怎么部署？"]

            # 系统提示词注入了检索上下文，且消息序列为 [system, human]
            system_msg, human_msg = llm.received[0], llm.received[-1]
            assert isinstance(system_msg, SystemMessage)
            assert "docker compose 一键启动" in system_msg.content
            assert human_msg.content == "怎么部署？"

            # 问答写入会话历史
            rows = app_state.session_store.load_history(session_id)
            assert [(r[0], r[1]) for r in rows] == [
                ("human", "怎么部署？"),
                ("ai", "根据知识库：docker compose 一键启动"),
            ]
        finally:
            app_state.ready = original["ready"]
            app_state.retriever = original["retriever"]
            app_state.retrieval_config = original["retrieval_config"]
            app_state.reranker = original["reranker"]
            app_state.llm = original["llm"]

    def test_pure_mode_stream_pipeline(self, client, tokens):
        """Pure 流式链路：SSE 推送检索进度 -> 逐 token -> done，历史落库"""
        retriever = FakeRetriever()
        llm = FakeLLM(chunks=("根据", "知识库", "回答"))
        original = {
            "ready": app_state.ready, "retriever": app_state.retriever,
            "retrieval_config": app_state.retrieval_config,
            "reranker": app_state.reranker, "llm": app_state.llm,
        }
        app_state.ready = True
        app_state.retriever = retriever
        app_state.retrieval_config = {"enable_rerank": False, "max_result_docs": 5, "max_result_chars": 6000}
        app_state.reranker = None
        app_state.llm = llm
        try:
            session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
            resp = client.post(
                "/chat/stream",
                json={"question": "怎么部署？", "session_id": session_id, "mode": "pure"},
                headers=auth(tokens["alice"]),
            )
            assert resp.status_code == 200
            # SSE payload 经 json.dumps ensure_ascii 转义，解析事件后断言
            events = [
                json.loads(line[len("data: "):])
                for line in resp.text.splitlines()
                if line.startswith("data: ")
            ]
            thinking_texts = [t for e in events if "thinking" in e for t in [e["thinking"].get("text")] if t]
            tokens = "".join(e["token"] for e in events if "token" in e)

            assert any("正在检索" in t for t in thinking_texts)
            assert any("完成" in t for t in thinking_texts)
            assert any(e["thinking"].get("done") for e in events if "thinking" in e)
            assert tokens == "根据知识库回答"
            assert any(e.get("done") for e in events)

            rows = app_state.session_store.load_history(session_id)
            assert [(r[0], r[1]) for r in rows] == [
                ("human", "怎么部署？"),
                ("ai", "根据知识库回答"),
            ]
        finally:
            app_state.ready = original["ready"]
            app_state.retriever = original["retriever"]
            app_state.retrieval_config = original["retrieval_config"]
            app_state.reranker = original["reranker"]
            app_state.llm = original["llm"]


class FakeAgentGraph:
    """Agent 链路测试替身：invoke 返回固定 AI 消息"""

    def invoke(self, payload, config=None):
        return {"messages": [AIMessage(content="agent 综合回答")]}


class TestAutoMode:
    """Auto 模式：规则路由后按实际模式走对应链路"""

    def _setup(self, retriever=None, llm=None):
        retriever = retriever or FakeRetriever()
        llm = llm or FakeLLM()
        original = {
            "ready": app_state.ready, "retriever": app_state.retriever,
            "retrieval_config": app_state.retrieval_config,
            "reranker": app_state.reranker, "llm": app_state.llm,
            "agent_graph": app_state.agent_graph,
        }
        app_state.ready = True
        app_state.retriever = retriever
        app_state.retrieval_config = {"enable_rerank": False, "max_result_docs": 5, "max_result_chars": 6000}
        app_state.reranker = None
        app_state.llm = llm
        return retriever, llm, original

    def test_auto_short_query_routes_to_pure(self, client, tokens):
        """短 query（无复杂指令词）经 auto 路由到 Pure：恰好一次检索 + 上下文注入"""
        retriever, llm, original = self._setup()
        try:
            session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
            resp = client.post(
                "/chat",
                json={"question": "怎么部署？", "session_id": session_id, "mode": "auto"},
                headers=auth(tokens["alice"]),
            )
            assert resp.status_code == 200
            assert "docker compose" in resp.json()["answer"]
            # Pure 链路：恰好一次检索，query 为用户提问文本
            assert retriever.calls == ["怎么部署？"]
        finally:
            app_state.ready = original["ready"]
            app_state.retriever = original["retriever"]
            app_state.retrieval_config = original["retrieval_config"]
            app_state.reranker = original["reranker"]
            app_state.llm = original["llm"]

    def test_auto_complex_query_routes_to_agent(self, client, tokens):
        """含复杂指令词（如"设计"）经 auto 路由到 Agent：走 agent_graph"""
        _, llm, original = self._setup()
        app_state.agent_graph = FakeAgentGraph()
        try:
            session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
            resp = client.post(
                "/chat",
                json={"question": "元数据怎么设计", "session_id": session_id, "mode": "auto"},
                headers=auth(tokens["alice"]),
            )
            assert resp.status_code == 200
            assert resp.json()["answer"] == "agent 综合回答"
            # Agent 链路不直接调用 retriever（检索发生在 agent_graph 内部）
        finally:
            app_state.agent_graph = original["agent_graph"]
            app_state.ready = original["ready"]
            app_state.retriever = original["retriever"]
            app_state.retrieval_config = original["retrieval_config"]
            app_state.reranker = original["reranker"]
            app_state.llm = original["llm"]

    def test_auto_with_image_routes_to_agent(self, client, tokens):
        """含图片输入经 auto 路由到 Agent"""
        _, llm, original = self._setup()
        app_state.agent_graph = FakeAgentGraph()
        try:
            session_id = client.post("/sessions", headers=auth(tokens["alice"])).json()["session_id"]
            resp = client.post(
                "/chat",
                json={
                    "question": "这张架构图说明了什么",
                    "session_id": session_id,
                    "mode": "auto",
                    "images": ["data:image/png;base64,aGVsbG8="],
                },
                headers=auth(tokens["alice"]),
            )
            assert resp.status_code == 200
            assert resp.json()["answer"] == "agent 综合回答"
        finally:
            app_state.agent_graph = original["agent_graph"]
            app_state.ready = original["ready"]
            app_state.retriever = original["retriever"]
            app_state.retrieval_config = original["retrieval_config"]
            app_state.reranker = original["reranker"]
            app_state.llm = original["llm"]
