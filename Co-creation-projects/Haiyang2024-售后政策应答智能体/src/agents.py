# -*- coding: utf-8 -*-
"""三种模式的智能体构建与执行封装。

三种模式的差异**只体现在编排方式**上，业务提示词、政策库、模型与温度完全一致：

| 模式 | 框架类 | 工具 | 循环上限 | 代表的能力档位 |
|---|---|---|---|---|
| A | SimpleAgent | 无 | 单轮 | 直接问模型（无依据） |
| B | FunctionCallAgent | search_policy | 3 轮工具调用 | 检索增强 |
| C | ReActAgent | 三个工具 | 5 步 | 多步自治 + 出稿自检 |

成本指标怎么来（框架没有现成的用量统计）：
    在**实例级**给 llm.invoke 与 tool.run 打一层计数包装。这样两条工具调用
    路径（原生函数调用与文本解析）都会被统计到，且不侵入框架源码。
    相比 token 数，调用次数更稳定也更容易解释——它直接反映多轮循环的开销。
"""

from __future__ import annotations

import contextlib
import io
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from hello_agents import FunctionCallAgent, HelloAgentsLLM, ReActAgent, SimpleAgent

from .prompts import REACT_TEMPLATE, SYSTEM_PROMPT, build_user_message
from .retriever import PolicyRetriever
from .tools import build_registry

#: 模式定义：键是模式代号，值是展示用信息
MODES: dict[str, dict[str, str]] = {
    "A": {
        "label": "无检索直答",
        "agent": "SimpleAgent（不带工具）",
        "capability": "直接依赖模型自身知识作答",
    },
    "B": {
        "label": "单跳检索",
        "agent": "FunctionCallAgent + search_policy",
        "capability": "原生函数调用检索政策依据",
    },
    "C": {
        "label": "多步自治",
        "agent": "ReActAgent + 三个工具",
        "capability": "多步推理、先规划再检索、出稿后自检红线",
    },
}


@dataclass
class RunResult:
    """一次执行的结果与成本指标。"""

    mode: str
    ticket: str
    answer: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed_ms: int = 0
    error: str = ""
    trace: str = ""  # 框架打印的推理轨迹（ReAct 模式下能看到 Thought/Action/Action）

    @property
    def answer_chars(self) -> int:
        return len(self.answer)


#: 命中限流时的重试次数与退避基数（秒）
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 3.0


def _with_retry(call: Callable[[], Any]) -> Any:
    """对限流类错误做指数退避重试，其他异常直接抛出。

    为什么必须有：批量评测时供应商会返回 429，若不退避，个别样本会被记成
    "空回答"，看起来像是 Agent 能力问题，实际是配额问题——这种脏数据一旦
    写进对比报告，结论就不可信了（首次小规模试跑已经发生过一次）。
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - 需要按错误文本判断是否可重试
            message = str(exc)
            retryable = "429" in message or "rate limit" in message.lower() or "timed out" in message.lower()
            if not retryable or attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    raise RuntimeError("重试逻辑异常：不应到达此处")


def _tidy_answer(answer: str) -> str:
    """把 Finish 里的单行字段展开成可读文本。

    模板要求最终回复按「客户回复｜依据规则｜内部建议｜风险提醒」写成一行，
    这里还原成换行显示，便于人工阅读与截图。
    """
    if answer.count("｜") >= 2:
        parts = [part.strip() for part in answer.split("｜") if part.strip()]
        if len(parts) >= 3:
            return "\n\n".join(parts)
    return answer


def _recover_from_trace(trace: str) -> str:
    """定稿解析失败时的兜底：从执行轨迹里取回最后一次合规检查的草稿。

    为什么会需要：框架的 ReActAgent 用 ``Action: (.*)`` 行内正则解析输出，
    ``Finish[...]`` 一旦写成多行就会解析成空串。实测模型仍有概率违反单行要求，
    而轨迹里的 ``🎬 行动: check_compliance[草稿]`` 是完整可用的文本，可直接取回。
    """
    drafts = re.findall(r"🎬 行动: check_compliance\[(.+)\]\s*$", trace, flags=re.MULTILINE)
    if drafts:
        return drafts[-1].strip()
    return ""


@dataclass
class _Counters:
    """实例级计数器。"""

    llm: list[int] = field(default_factory=list)
    tool: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.llm.clear()
        self.tool.clear()


class AgentRunner:
    """按模式构建智能体并执行单条工单。"""

    def __init__(self, retriever: PolicyRetriever, temperature: float = 0.0, quiet: bool = True):
        self.retriever = retriever
        self.temperature = temperature
        self.quiet = quiet

    # ── 内部：构建与埋点 ──

    def _new_llm(self) -> HelloAgentsLLM:
        return HelloAgentsLLM(temperature=self.temperature)

    @staticmethod
    def _instrument_llm(llm: Any, counters: _Counters) -> None:
        """统计模型调用次数。

        不能只包装 ``llm.invoke``：FunctionCallAgent 直接使用
        ``llm._client.chat.completions.create`` 走原生函数调用，绕过了 invoke，
        导致模式 B 的次数统计为 0（首次冒烟实测即为 0）。改为包装底层的 create，
        聚合调用、流式调用与原生函数调用三条路径就都能覆盖。
        """
        completions = getattr(getattr(getattr(llm, "_client", None), "chat", None), "completions", None)
        if completions is None:
            return
        original = completions.create

        def counted(*args: Any, **kwargs: Any) -> Any:
            counters.llm.append(1)
            return _with_retry(lambda: original(*args, **kwargs))

        completions.create = counted

    @staticmethod
    def _instrument_tools(tools: list[Any], counters: _Counters) -> None:
        for tool in tools:
            original = tool.run

            def counted(parameters: dict[str, Any], _original=original) -> Any:
                counters.tool.append(1)
                return _original(parameters)

            tool.run = counted

    def _build(self, mode: str, counters: _Counters) -> Any:
        """每种模式都构建全新实例，避免上一条样本的对话历史污染下一条。"""
        llm = self._new_llm()
        self._instrument_llm(llm, counters)

        if mode == "A":
            return SimpleAgent(name="售后助手-直答", llm=llm, system_prompt=SYSTEM_PROMPT)

        if mode == "B":
            registry, tools = build_registry(self.retriever, "search")
            self._instrument_tools(tools, counters)
            return FunctionCallAgent(
                name="售后助手-检索",
                llm=llm,
                system_prompt=SYSTEM_PROMPT,
                tool_registry=registry,
                max_tool_iterations=3,
            )

        if mode == "C":
            registry, tools = build_registry(self.retriever, "full")
            self._instrument_tools(tools, counters)
            return ReActAgent(
                name="售后助手-自治",
                llm=llm,
                tool_registry=registry,
                custom_prompt=REACT_TEMPLATE,
                max_steps=5,
            )

        raise ValueError(f"未知模式：{mode}（可选：{'、'.join(MODES)}）")

    # ── 对外：执行 ──

    def run(self, ticket: str, mode: str) -> RunResult:
        counters = _Counters()
        question = build_user_message(ticket)

        buffer = io.StringIO()
        started = time.perf_counter()
        error = ""
        answer = ""
        try:
            if self.quiet:
                # 构建阶段也要一起静默：register_tool 会打印"✅ 工具已注册"，
                # 这些行若漏到评测输出里，会把逐条结果冲得看不清
                with contextlib.redirect_stdout(buffer):
                    agent = self._build(mode, counters)
                    answer = agent.run(question)
            else:
                agent = self._build(mode, counters)
                answer = agent.run(question)
        except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整轮评测
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        trace = buffer.getvalue().strip()

        answer = _tidy_answer((answer or "").strip())
        if mode == "C" and not answer:
            recovered = _recover_from_trace(trace)
            if recovered:
                answer = recovered
                error = error or "ReAct 定稿解析为空，已从执行轨迹取回最后一次合规检查的草稿"

        return RunResult(
            mode=mode,
            ticket=ticket,
            answer=answer,
            llm_calls=len(counters.llm),
            tool_calls=len(counters.tool),
            elapsed_ms=elapsed_ms,
            error=error,
            trace=trace,
        )
