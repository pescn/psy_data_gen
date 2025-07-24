"""
智能体基类实现
包含ChatBot和Agent两个基类，提供状态管理和LLM交互功能
以及Chat History转换工具函数
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from openai import AsyncOpenAI

from models import ConversationMessage, CounselorState, EmotionState, RiskAssessment
from settings import SystemConfig


# ==================== Chat History转换工具函数 ====================


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


# ==================== 基类定义 ====================


class ChatBot(ABC):
    """
    ChatBot基类
    用于存储对话历史，管理状态机，请求LLM API
    """

    api_key: str = "sk-647284e3ebe250ce9a90f9ef72becb92"
    base_url: str = "http://172.16.32.39:8080/v1"
    model: str = "deepseek-v3"

    def __init__(self, bot_id: str = None, **kwargs):
        """
        初始化ChatBot

        Args:
            bot_id: Bot唯一标识符
            **kwargs: 其他初始化参数
        """
        self.bot_id = bot_id or str(uuid.uuid4())
        self.conversation_history: List[ConversationMessage] = []
        self.current_round = 0

        # 状态管理（主要用于咨询师Bot）
        self.current_state: Optional[CounselorState] = None
        self.state_history: List[Dict[str, Any]] = []

        # 情绪状态（主要用于学生Bot）
        self.current_emotion: Optional[EmotionState] = None
        self.emotion_history: List[Dict[str, Any]] = []

        # LLM客户端配置（子类中具体实现）
        self.llm_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.temperature = SystemConfig.DEFAULT_TEMPERATURE
        self.max_tokens = SystemConfig.DEFAULT_MAX_TOKENS

        # 提示词模板
        self.system_prompts: Dict[str, str] = {}
        self.current_system_prompt = ""

        # 初始化配置
        self._init_config(**kwargs)

    @abstractmethod
    def _init_config(self, **kwargs):
        """初始化配置（子类实现）"""
        pass

    @abstractmethod
    def _build_system_prompt(self, **context) -> str:
        """构建系统提示词（子类实现）"""
        raise NotImplementedError(
            "Subclasses must implement _build_system_prompt method"
        )

    def add_message(self, role: str, content: str, **metadata) -> ConversationMessage:
        """
        添加对话消息

        Args:
            role: 角色（student/counselor）
            content: 消息内容
            **metadata: 额外元数据

        Returns:
            ConversationMessage: 创建的消息对象
        """
        self.current_round += 1

        message = ConversationMessage(
            role=role,
            content=content,
            state=self.current_state.value if self.current_state else None,
            emotion=self.current_emotion,
            round_number=self.current_round,
            **metadata,
        )

        self.conversation_history.append(message)
        return message

    def get_conversation_context(
        self, max_history: int = None, for_role: str = None
    ) -> List[Dict[str, str]]:
        """
        获取对话上下文（用于LLM API调用）

        Args:
            max_history: 最大历史记录数量
            for_role: 指定角色视角 ('student' 或 'counselor')

        Returns:
            List[Dict]: LLM API格式的对话历史
        """
        history = self.conversation_history
        if max_history:
            history = get_last_n_messages(history, max_history)

        # 根据指定角色转换视角
        if for_role == "student":
            return convert_history_for_student(history)
        elif for_role == "counselor":
            return convert_history_for_counselor(history)
        else:
            # 默认行为：保持原有逻辑
            context = []
            for msg in history:
                context.append(
                    {
                        "role": "user" if msg.role == "student" else "assistant",
                        "content": msg.content,
                    }
                )
            return context

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

    async def generate_response(
        self, user_input: str = None, for_role: str = None, **context
    ) -> str:
        """
        生成回复

        Args:
            user_input: 用户输入（可选，用于添加到历史）
            for_role: 指定Bot角色 ('student' 或 'counselor')，用于正确转换Chat History视角
            **context: 额外上下文信息

        Returns:
            str: 生成的回复内容
        """
        # 添加用户输入到历史（如果提供）
        if user_input:
            self.add_message("user", user_input)

        # 构建系统提示词
        system_prompt = self._build_system_prompt(**context)

        # 构建消息列表，使用指定角色的视角
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.get_conversation_context(for_role=for_role))

        # 调用LLM API
        response = await self._call_llm_api(messages)

        # 添加响应到历史
        self.add_message("assistant", response)

        return response

    async def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用LLM API

        Args:
            messages: 消息列表

        Returns:
            str: LLM响应内容
        """
        if not self.llm_client or not self.model:
            raise ValueError("LLM client and model must be configured in subclass")

        try:
            # 这里是通用的OpenAI兼容API调用逻辑
            # 子类可以重写此方法来适配特定的API
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        return {
            "bot_id": self.bot_id,
            "current_round": self.current_round,
            "current_state": self.current_state.value if self.current_state else None,
            "current_emotion": self.current_emotion.value
            if self.current_emotion
            else None,
            "message_count": len(self.conversation_history),
        }

    def reset(self):
        """重置Bot状态"""
        self.conversation_history.clear()
        self.current_round = 0
        self.current_state = None
        self.current_emotion = None
        self.state_history.clear()
        self.emotion_history.clear()


class Agent(ABC):
    """
    Agent基类
    用于结构化地请求LLM并解析JSON格式返回
    """

    api_key: str = "sk-647284e3ebe250ce9a90f9ef72becb92"
    base_url: str = "http://172.16.32.39:8080/v1"
    model: str = "deepseek-v3"

    def __init__(self, agent_id: str = None, **kwargs):
        """
        初始化Agent

        Args:
            agent_id: Agent唯一标识符
            **kwargs: 其他初始化参数
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.execution_history: List[Dict[str, Any]] = []

        # LLM客户端配置（子类中具体实现）
        self.llm_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.temperature = SystemConfig.DEFAULT_TEMPERATURE
        self.max_tokens = SystemConfig.DEFAULT_MAX_TOKENS

        # 提示词模板
        self.prompt_template = ""

        # 初始化配置
        self._init_config(**kwargs)

    @abstractmethod
    def _init_config(self, **kwargs):
        """初始化配置（子类实现）"""
        pass

    @abstractmethod
    def _build_prompt(self, **context) -> str:
        """构建提示词（子类实现）"""
        pass

    @abstractmethod
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应（子类实现）"""
        pass

    async def execute(self, **context) -> Dict[str, Any]:
        """
        执行Agent任务

        Args:
            **context: 执行上下文

        Returns:
            Dict[str, Any]: 解析后的结果
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # 构建提示词
            prompt = self._build_prompt(**context)

            # 调用LLM API
            raw_response = await self._call_llm_api(prompt)

            # 解析响应
            parsed_result = self._parse_response(raw_response)

            # 记录执行历史
            execution_record = {
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "context": context,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_result": parsed_result,
                "success": True,
                "error": None,
            }

            self.execution_history.append(execution_record)
            return parsed_result

        except Exception as e:
            # 记录错误
            execution_record = {
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "context": context,
                "success": False,
                "error": str(e),
            }

            self.execution_history.append(execution_record)
            raise RuntimeError(f"Agent execution failed: {str(e)}")

    async def _call_llm_api(self, prompt: str) -> str:
        """
        调用LLM API

        Args:
            prompt: 提示词

        Returns:
            str: LLM响应内容
        """
        if not self.llm_client or not self.model:
            raise ValueError("LLM client and model must be configured in subclass")

        try:
            messages = [{"role": "user", "content": prompt}]

            # 这里是通用的OpenAI兼容API调用逻辑
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def _safe_json_parse(self, response: str) -> Dict[str, Any]:
        """
        安全地解析JSON响应

        Args:
            response: LLM返回的字符串

        Returns:
            Dict[str, Any]: 解析后的JSON对象
        """
        # 尝试从响应中提取JSON部分
        response = response.strip()

        # 移除可能的markdown代码块标记
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        # 查找JSON对象的开始和结束
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx : end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 如果提取失败，尝试直接解析整个响应
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {str(e)}\nResponse: {response}"
            )

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for record in self.execution_history if record.get("success", False)
        )

        return {
            "agent_id": self.agent_id,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions
            if total_executions > 0
            else 0,
            "last_execution": self.execution_history[-1]
            if self.execution_history
            else None,
        }

    def clear_history(self):
        """清除执行历史"""
        self.execution_history.clear()


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
        risk_keywords = SystemConfig.get_risk_keywords()

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
            emergency_required=overall_risk >= SystemConfig.RISK_THRESHOLD,
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
