"""
Agent 智能问答链路

基于 LangGraph 实现 ReAct Agent 状态图，支持 LLM 自主决策工具调用、检索计数上限控制与 MemorySaver 会话记忆。
"""
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from utils.logger import logger
from utils.exceptions import LLMClientError
from prompts import get_agent_prompt

MAX_RETRIEVAL_COUNT = 4

class AgentState(MessagesState):
    """Agent 状态，扩展检索计数器字段"""
    retrieval_count: int

def build_agent_graph(llm, tools: list):
    """构建基于 LangGraph 的 ReAct Agent 状态图

    包含 agent 决策节点与 tools 执行节点，通过条件边实现工具调用循环，
    检索计数达到上限时强制终止工具调用并生成最终回答。

    Args:
        llm: 完成初始化配置的大语言模型实例
        tools: 所有可供 Agent 调用的工具列表

    Returns:
        tuple: (编译完成的 Agent 状态图, 工具名称到工具实例的映射字典)

    Raises:
        LLMClientError: Agent 链路构建失败时抛出
    """
    try:
        prompt = get_agent_prompt()
        llm_with_bind_tools = llm.bind_tools(tools)
        tool_mapping = {tool.name: tool for tool in tools}

        def agent_node(state: AgentState):
            """Agent 决策节点：LLM 思考，输出回答或工具调用指令

            检索次数达上限时，不绑定工具，强制 LLM 生成纯文本回答。
            """
            input_messages = state["messages"]
            formatted_prompt = prompt.invoke({"chat_history": input_messages[:-1], "question": input_messages[-1].content})
            if state.get("retrieval_count", 0) >= MAX_RETRIEVAL_COUNT:
                response = llm.invoke(formatted_prompt)
            else:
                response = llm_with_bind_tools.invoke(formatted_prompt)
            return {"messages": [response]}

        def tool_node(state: AgentState):
            """工具执行节点：解析 tool_call，执行对应工具并返回结果

            仅 knowledge_base_search 计入检索次数，单次工具异常不中断整体链路。
            """
            outputs = []
            last_msg = state["messages"][-1]
            new_retrieval_count = 0

            tool_calls = getattr(last_msg, "tool_calls", []) or []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                call_id = tool_call["id"]

                if tool_name not in tool_mapping:
                    err_msg = f"不存在指定工具：{tool_name}"
                    logger.warning(err_msg)
                    outputs.append(ToolMessage(content=err_msg, tool_call_id=call_id))
                    continue

                if tool_name == "knowledge_base_search":
                    new_retrieval_count += 1

                try:
                    logger.info(f"执行工具 {tool_name}，入参：{tool_args}")
                    tool_res = tool_mapping[tool_name].invoke(tool_args)
                    logger.info(f"工具 {tool_name} 执行完成，返回结果长度：{len(str(tool_res))}")
                    outputs.append(ToolMessage(content=str(tool_res), tool_call_id=call_id))
                except Exception as tool_err:
                    tool_res = f"工具 {tool_name} 执行异常：{str(tool_err)}"
                    logger.error(f"工具执行失败：{str(tool_err)}")
                    outputs.append(ToolMessage(content=tool_res, tool_call_id=call_id))

            total_count = state.get("retrieval_count", 0) + new_retrieval_count

            if total_count >= MAX_RETRIEVAL_COUNT:
                logger.info(f"检索次数已达上限{MAX_RETRIEVAL_COUNT}，追加强制回答提示")
                outputs.append(HumanMessage(
                    content="检索次数已达上限，请基于已获取的全部检索结果直接回答用户问题，不要再调用任何工具。"
                ))

            return {"messages": outputs, "retrieval_count": total_count}

        def should_continue(state: AgentState):
            """条件分支：判断是否继续调用工具还是直接结束

            检索次数达上限时，即使 LLM 仍尝试调用工具也强制终止。
            """
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                if state.get("retrieval_count", 0) >= MAX_RETRIEVAL_COUNT:
                    logger.warning(f"检索次数已达上限，LLM仍尝试调用工具，终止执行")
                    return END
                return "tools"
            return END

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)
        graph_builder.set_entry_point("agent")
        graph_builder.add_conditional_edges("agent", should_continue)
        graph_builder.add_edge("tools", "agent")

        checkpoint = MemorySaver()
        agent_graph = graph_builder.compile(checkpointer=checkpoint)

        logger.info(f"LangGraph Agent链路初始化完成，注册工具列表：{list(tool_mapping.keys())}")
        return agent_graph, tool_mapping
    except Exception as err:
        logger.error(f"LangGraph Agent链路构建失败：{str(err)}")
        raise LLMClientError(f"Agent链路初始化异常：{str(err)}")