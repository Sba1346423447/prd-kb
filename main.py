"""
企业技术支持智研知识库（PRD-KB）—— 命令行交互入口

使用 Agent 自主检索模式，模型借助 Function Calling 自主决策是否调用知识库工具。

启动方式：
    python main.py
"""
import sys
from pathlib import Path
from typing import Any

project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from utils.logger import logger
from core.config_loader import load_config
from core.knowledge_base import init_knowledge_base
from core.strategy.retrieval_strategy import build_advanced_retriever
from core.llm_client import init_llm
from core.tools import get_rag_tools
from core.agent_chain import build_agent_graph
from core.strategy.rerank_strategy import get_reranker
from langchain_core.messages import HumanMessage


def main():
    """主业务执行入口，完成全链路初始化以及交互式问答循环

    初始化流程：配置加载 -> 知识库构建 -> LLM 初始化 -> 多路检索器构建 ->
    Rerank 模型预热 -> Agent 工具注册 -> Agent 图构建 -> 启动交互循环。
    """
    config = load_config()
    chroma_helper = init_knowledge_base(config, dir_path="docs/")
    retrieval_config = config["retrieval"]

    llm_config = config["llm"]
    llm = init_llm(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        model_name=llm_config["model_name"],
        temperature=llm_config["temperature"],
    )

    retriever = build_advanced_retriever(chroma_helper, retrieval_config)

    reranker = get_reranker(
        model_path=retrieval_config["rerank_model_path"],
        device=retrieval_config.get("device", "cpu")
    ) if retrieval_config.get("enable_rerank", False) else None

    rag_tools = get_rag_tools(retriever, retrieval_config, reranker)
    agent_graph, _ = build_agent_graph(llm, rag_tools)

    session_config: Any = {"configurable": {"thread_id": "session_001"}}

    print("=" * 60)
    print("PRD-KB 企业技术支持智研知识库")
    print("当前运行模式：【Agent 自主检索】模型自主判断是否检索知识库")
    print("输入 exit / quit / q 退出对话")
    print("=" * 60)

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("对话结束，程序退出")
            break

        try:
            print("助手：", end="", flush=True)
            for chunk in agent_graph.stream(
                {"messages": [HumanMessage(content=user_input)]},  # type: ignore[arg-type]
                config=session_config,
                stream_mode="messages"
            ):
                if isinstance(chunk, tuple):
                    msg_chunk = chunk[0]
                else:
                    msg_chunk = chunk
                if hasattr(msg_chunk, "content") and msg_chunk.content:  # type: ignore[union-attr]
                    print(msg_chunk.content, end="", flush=True)  # type: ignore[union-attr]
            print()

        except Exception as e:
            logger.error(f"对话处理异常: {str(e)}")
            print("助手：抱歉，处理问题时发生错误，请重新提问")


if __name__ == "__main__":
    try:
        main()
    except Exception as global_err:
        logger.critical(f"程序全局捕获异常: {str(global_err)}", exc_info=True)
        print("程序发生严重错误，已终止，请查看日志排查问题")