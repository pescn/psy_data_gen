"""
智能体基类实现
包含ChatBot和Agent两个基类，提供状态管理和LLM交互功能
以及Chat History转换工具函数
"""

import json
from typing import Dict, Generic, List, Any, Type, TypeVar, Union
from datetime import datetime

from openai import AsyncOpenAI
from pydantic import BaseModel
from opentelemetry import trace

from models import ConversationMessage, CounselorState, EmotionState, RiskAssessment
from settings import settings

# 为 Agent 定义泛型类型变量
TContext = TypeVar("TContext")
TResult = TypeVar("TResult", bound=BaseModel)


def convert_history_for_student(
    conversation_history: List[ConversationMessage],
) -> List[Dict[str, str]]:
    """
    将对话历史转换为学生Bot的视角
    学生Bot视角：assistant=学生(Q), user=咨询师(A)

    Args:
        conversation_history: 完整的对话历史

    Returns:
        List[Dict]: 适用于学生Bot的消息格式
    """
    converted_history = []

    for msg in conversation_history:
        if msg.role == "student":
            # 学生的消息作为assistant
            converted_history.append({"role": "assistant", "content": msg.content})
        elif msg.role == "counselor":
            # 咨询师的消息作为user
            converted_history.append({"role": "user", "content": msg.content})

    return converted_history


def convert_history_for_counselor(
    conversation_history: List[ConversationMessage],
) -> List[Dict[str, str]]:
    """
    将对话历史转换为咨询师Bot的视角
    咨询师Bot视角：assistant=咨询师(A), user=学生(Q)

    Args:
        conversation_history: 完整的对话历史

    Returns:
        List[Dict]: 适用于咨询师Bot的消息格式
    """
    converted_history = []

    for msg in conversation_history:
        if msg.role == "counselor":
            # 咨询师的消息作为assistant
            converted_history.append({"role": "assistant", "content": msg.content})
        elif msg.role == "student":
            # 学生的消息作为user
            converted_history.append({"role": "user", "content": msg.content})

    return converted_history


def get_last_n_messages(
    conversation_history: List[ConversationMessage], n: int = 10
) -> List[ConversationMessage]:
    """
    获取最近n条消息（用于限制上下文长度）

    Args:
        conversation_history: 完整对话历史
        n: 消息数量

    Returns:
        List[ConversationMessage]: 最近n条消息
    """
    return (
        conversation_history[-n:]
        if len(conversation_history) > n
        else conversation_history
    )


class ChatBot:
    """
    ChatBot基类
    用于存储对话历史，管理状态机，请求LLM API
    """

    api_key: str = settings.LLM_API_KEY
    base_url: str = settings.LLM_API_BASE_URL
    model: str = settings.LLM_MODEL
    temperature: float = settings.DEFAULT_TEMPERATURE
    max_tokens: int = settings.DEFAULT_MAX_TOKENS

    def __init__(self):
        """
        初始化ChatBot
        """
        self.current_round = 0
        self.llm_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.usage = None

    def convert_history_to_messages(
        self, conversation_history: List[ConversationMessage]
    ) -> List[Dict[str, str]]:
        """
        将对话历史，根据对话角色 Mapping 关系，转换为LLM API所需的消息格式

        Args:
            conversation_history: 完整的对话历史

        Returns:
            List[Dict]: LLM API格式的消息列表
        """
        raise NotImplementedError(
            "Subclasses must implement convert_history_to_messages method"
        )

    @property
    def system_prompt(self) -> str:
        """
        获取当前系统提示词
        子类可以重写此方法以提供动态提示词
        """
        raise NotImplementedError("Subclasses must implement system_prompt property")

    def trans_state(
        self, new_state: Union[CounselorState, EmotionState], reason: str = ""
    ):
        """
        更新状态

        Args:
            new_state: 新状态
            reason: 状态变更原因
        """
        raise NotImplementedError("Subclasses must implement trans_state method")

    async def chat(self, history: List[ConversationMessage]) -> str:
        if not self.llm_client or not self.model:
            raise ValueError("LLM client and model must be configured in subclass")

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.convert_history_to_messages(history)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                # temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            self.usage = response.usage
            current_span = trace.get_current_span()
            current_span.add_event(
                name="reasoning.generated",
                attributes={
                    "llm.reasoning": response.choices[0].message.reasoning_content,
                    "llm.response": response.choices[0].message.content.strip()
                },
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def update_state(
        self, new_state: Union[CounselorState, EmotionState], reason: str = ""
    ):
        """
        更新状态

        Args:
            new_state: 新状态
            reason: 状态变更原因
        """
        old_state = None

        if isinstance(new_state, CounselorState):
            old_state = self.current_state
            self.current_state = new_state
            self.state_history.append(
                {
                    "round": self.current_round,
                    "old_state": old_state.value if old_state else None,
                    "new_state": new_state.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        elif isinstance(new_state, EmotionState):
            old_emotion = self.current_emotion
            self.current_emotion = new_state
            self.emotion_history.append(
                {
                    "round": self.current_round,
                    "old_emotion": old_emotion.value if old_emotion else None,
                    "new_emotion": new_state.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        return {
            "current_round": self.current_round,
            "current_state": self.current_state.value if self.current_state else None,
            "current_emotion": self.current_emotion.value
            if self.current_emotion
            else None,
            "message_count": len(self.conversation_history),
        }


class Agent(Generic[TContext, TResult]):
    """
    Agent基类
    用于结构化地请求LLM并解析JSON格式返回
    """

    api_key: str = settings.LLM_API_KEY
    base_url: str = settings.LLM_API_BASE_URL
    model: str = settings.LLM_MODEL
    temperature: float = settings.DEFAULT_TEMPERATURE
    max_tokens: int = settings.DEFAULT_MAX_TOKENS

    context_class: Type[TContext] = None
    result_class: Type[TResult] = None

    def __init__(self, **kwargs):
        """
        初始化Agent

        Args:
            **kwargs: 其他初始化参数
        """
        # LLM客户端配置（子类中具体实现）
        self.llm_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.config = kwargs
        self.usage = None

    def prompt(self, context: TContext) -> str:
        """
        获取当前提示词
        子类可以重写此方法以提供动态提示词
        """
        raise NotImplementedError("Subclasses must implement prompt method")

    def clean_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理LLM返回的JSON数据，确保符合result_class的结构

        Args:
            data: LLM返回的原始JSON数据

        Returns:
            Dict[str, Any]: 清理后的数据
        """
        return data  # 默认情况下不需要做额外处理

    async def execute(self, context: TContext) -> TResult:
        """
        执行Agent任务，返回解析后的结果

        Returns:
            Dict[str, Any]: 解析后的结果
        """
        if not self.result_class:
            raise ValueError("result_class must be set in subclass")
        try:
            messages = [{"role": "user", "content": self.prompt(context)}]
            response = await self.llm_client.chat.completions.parse(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                response_format=self.result_class,
            )
            resp_content = response.choices[0].message.content.strip()
            resp_data = self.clean_response_data(json.loads(resp_content))
            self.data = self.result_class(**resp_data)
            self.usage = response.usage
            current_span = trace.get_current_span()
            current_span.add_event(
                name="reasoning.generated",
                attributes={
                    "llm.reasoning": response.choices[0].message.reasoning_content,
                    "llm.response": response.choices[0].message.content.strip(),
                },
            )
            return self.data

        except Exception as e:
            # print(resp_content)
            raise RuntimeError(f"LLM API call failed: {str(e)}") from e


class RiskAssessmentMixin:
    """
    风险评估混入类
    为ChatBot和Agent提供风险评估功能
    """

    def assess_risk(self, content: str) -> RiskAssessment:
        """
        评估内容的风险等级

        Args:
            content: 待评估的内容

        Returns:
            RiskAssessment: 风险评估结果
        """
        risk_keywords = settings.get_risk_keywords()

        # 计算各类风险等级
        suicide_risk = self._calculate_risk_level(content, risk_keywords["suicide"])
        self_harm_risk = self._calculate_risk_level(content, risk_keywords["self_harm"])
        harm_others_risk = self._calculate_risk_level(
            content, risk_keywords["harm_others"]
        )

        overall_risk = max(suicide_risk, self_harm_risk, harm_others_risk)

        # 收集触发的风险指标
        risk_indicators = []
        if suicide_risk > 0:
            risk_indicators.extend(
                [kw for kw in risk_keywords["suicide"] if kw in content]
            )
        if self_harm_risk > 0:
            risk_indicators.extend(
                [kw for kw in risk_keywords["self_harm"] if kw in content]
            )
        if harm_others_risk > 0:
            risk_indicators.extend(
                [kw for kw in risk_keywords["harm_others"] if kw in content]
            )

        return RiskAssessment(
            suicide_risk=suicide_risk,
            self_harm_risk=self_harm_risk,
            harm_others_risk=harm_others_risk,
            overall_risk=overall_risk,
            risk_indicators=list(set(risk_indicators)),  # 去重
            emergency_required=overall_risk >= settings.RISK_THRESHOLD,
        )

    def _calculate_risk_level(self, content: str, keywords: List[str]) -> int:
        """
        计算特定类型的风险等级

        Args:
            content: 内容
            keywords: 风险关键词列表

        Returns:
            int: 风险等级 (0-5)
        """
        content_lower = content.lower()
        matched_keywords = [kw for kw in keywords if kw in content_lower]

        if not matched_keywords:
            return 0

        # 根据匹配的关键词数量和权重计算风险等级
        risk_score = len(matched_keywords)

        # 特别高风险的关键词加权
        high_risk_keywords = ["自杀", "想死", "结束生命", "割腕", "杀死"]
        for kw in high_risk_keywords:
            if kw in content_lower:
                risk_score += 2

        # 转换为0-5的等级
        if risk_score >= 5:
            return 5
        elif risk_score >= 3:
            return 4
        elif risk_score >= 2:
            return 3
        elif risk_score >= 1:
            return 2
        else:
            return 1
