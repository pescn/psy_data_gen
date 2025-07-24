"""
流程控制Agent
负责分析对话进展，判断状态转换时机，评估风险等级
"""

from typing import Dict, List, Optional, Any

from agents.base import Agent, RiskAssessmentMixin
from models import (
    ConversationMessage,
    CounselorState,
    RiskAssessment,
    StudentBackground,
    CounselorBackground,
)
from constants import STATE_TRANSITION_GUIDE, PSYCHOLOGICAL_ISSUES_DATA
from settings import SystemConfig


counselor_approaches_mapping = {
    "cognitive_behavioral_therapy": "认知行为疗法",
    "humanistic_therapy": "人本主义疗法",
    "psychoanalytic": "精神分析取向",
    "solution_focused": "解决焦点疗法",
    "mindfulness_therapy": "正念疗法",
}


class FlowControlAgent(Agent, RiskAssessmentMixin):
    """
    流程控制Agent
    每轮对话后自动评估是否需要状态转换，并进行风险评估
    """

    def _init_config(self, **kwargs):
        """初始化流程控制Agent配置"""
        # LLM配置（子类需要设置）
        # self.llm_client = kwargs.get('llm_client')
        # self.model = kwargs.get('model', 'gpt-4')

        # 背景信息（用于评估参考）
        self.student_background: Optional[StudentBackground] = kwargs.get(
            "student_background"
        )
        self.counselor_background: Optional[CounselorBackground] = kwargs.get(
            "counselor_background"
        )

        # 评估历史
        self.evaluation_history: List[Dict[str, Any]] = []

        # 构建提示词模板
        self._build_prompt_template()

    def _build_prompt_template(self):
        """构建提示词模板"""
        self.prompt_template = """你是一个专业的心理咨询流程控制专家，负责分析对话进展并判断是否需要进行状态转换。

请分析以下对话内容，并做出专业判断：

当前咨询师状态：{current_state}
对话轮数：{round_number}
学生背景信息：{student_background}
咨询师流派：{counselor_approach}

对话历史：
{conversation_history}

状态转换参考信息：
{state_guide}

分析任务：
1. **信息饱和度分析**：评估当前阶段的信息收集是否充分
2. **学生状态评估**：分析学生的信任度、开放度和配合程度
3. **咨询进展评估**：判断咨询是否达到当前阶段的目标
4. **风险评估**：检查是否存在自杀、自残或伤害他人的风险
5. **状态转换判断**：基于以上分析决定是否需要转换状态

请以JSON格式返回评估结果：
{{
    "round_analysis": {{
        "current_round": {round_number},
        "student_trust_level": "很低(0-0.2)/较低(0.2-0.4)/适中(0.4-0.6)/较高(0.6-0.8)/很高(0.8-1.0)",
        "student_openness": "很低/较低/适中/较高/很高",
        "information_saturation": "不充分(0-0.3)/部分充分(0.3-0.6)/基本充分(0.6-0.8)/充分(0.8-1.0)",
        "counselor_effectiveness": "需要改进/一般/良好/优秀"
    }},
    "state_transition": {{
        "need_transition": true/false,
        "current_state": "{current_state}",
        "recommended_state": "introduction/exploration/assessment/scale_recommendation",
        "transition_reason": "具体的转换理由",
        "confidence_level": "低/中/高"
    }},
    "risk_assessment": {{
        "overall_risk_level": 0-5的整数,
        "suicide_risk": 0-5的整数,
        "self_harm_risk": 0-5的整数,
        "harm_others_risk": 0-5的整数,
        "risk_indicators": ["检测到的风险关键词或表达"],
        "emergency_required": true/false,
        "risk_description": "风险情况的详细描述"
    }},
    "improvement_suggestions": [
        "对咨询师的改进建议1",
        "对咨询师的改进建议2"
    ],
    "next_focus": "下一轮咨询应该关注的重点"
}}

评估原则：
1. **信息饱和度判断**：
   - 引入阶段：是否了解了基本问题和建立了初步信任
   - 探索阶段：是否对核心困扰有了全面深入的了解
   - 评估阶段：是否形成了专业判断并获得学生认同
   - 量表阶段：是否完成了量表推荐和后续安排

2. **状态转换时机**：
   - 不要过早转换，确保当前阶段目标达成
   - 注意学生的接受程度和配合度
   - 考虑对话的自然流程

3. **风险评估重点**：
   - 仔细检查学生的表达中是否包含自杀、自残等风险信号
   - 注意隐含的绝望、无助情绪
   - 评估是否需要立即干预

4. **质量控制**：
   - 评估咨询师是否遵循了专业原则
   - 判断共情和信息收集的平衡
   - 识别可能的咨询失误
"""

    def _build_prompt(self, **context) -> str:
        """构建具体的提示词"""
        # 获取上下文信息
        conversation_history = context.get("conversation_history", [])
        current_state = context.get("current_state", CounselorState.INTRODUCTION)
        round_number = context.get("round_number", 0)

        # 格式化对话历史
        formatted_history = self._format_conversation_history(conversation_history)

        # 格式化学生背景信息
        student_bg_text = self._format_student_background()

        # 格式化咨询师信息
        counselor_approach = (
            counselor_approaches_mapping[
                self.counselor_background.therapy_approach.value
            ]
            if self.counselor_background
            else "综合取向"
        )

        # 获取当前状态的转换指导
        state_guide = self._format_state_guide(current_state)

        return self.prompt_template.format(
            current_state=current_state.value,
            round_number=round_number,
            student_background=student_bg_text,
            counselor_approach=counselor_approach,
            conversation_history=formatted_history,
            state_guide=state_guide,
        )

    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """格式化对话历史"""
        if not history:
            return "暂无对话历史"

        formatted = []
        for i, msg in enumerate(history[-10:], 1):  # 只取最近10轮对话
            role_name = "学生" if msg.role == "student" else "咨询师"
            emotion_info = f"(情绪: {msg.emotion.value})" if msg.emotion else ""
            formatted.append(f"{i}. {role_name}{emotion_info}: {msg.content}")

        return "\n".join(formatted)

    def _format_student_background(self) -> str:
        """格式化学生背景信息"""
        if not self.student_background:
            return "学生背景信息未提供"

        issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
            self.student_background.current_psychological_issue, {}
        )

        return f"""
- 基本信息：{self.student_background.name}，{self.student_background.age}岁，{self.student_background.grade}，{self.student_background.major}
- 性格特征：{", ".join(self.student_background.personality_traits)}
- 核心问题：{issue_data.get("name", "未知问题")}
- 症状描述：{self.student_background.symptom_description[:100]}...
- 家庭背景：{self.student_background.family_background[:100]}...
"""

    def _format_state_guide(self, current_state: CounselorState) -> str:
        """格式化当前状态的转换指导"""
        guide = STATE_TRANSITION_GUIDE.get(current_state, {})

        return f"""
当前阶段：{current_state.value}
- 关键目标：{", ".join(guide.get("key_goals", []))}
- 退出条件：{guide.get("exit_condition", "未定义")}
- 转换指标：{", ".join(guide.get("transition_indicators", []))}
- 典型持续时间：{guide.get("typical_duration", "未定义")}
"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            parsed_data = self._safe_json_parse(response)

            # 验证必需的字段
            required_fields = ["round_analysis", "state_transition", "risk_assessment"]
            for field in required_fields:
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")

            # 验证状态转换字段
            transition_data = parsed_data["state_transition"]
            if "need_transition" not in transition_data:
                raise ValueError("Missing need_transition field")

            # 验证风险评估字段
            risk_data = parsed_data["risk_assessment"]
            required_risk_fields = [
                "overall_risk_level",
                "suicide_risk",
                "self_harm_risk",
                "harm_others_risk",
            ]
            for field in required_risk_fields:
                if field not in risk_data:
                    risk_data[field] = 0

            return parsed_data

        except Exception as e:
            raise ValueError(f"Failed to parse flow control response: {str(e)}")

    async def evaluate_round(
        self,
        conversation_history: List[ConversationMessage],
        current_state: CounselorState,
        round_number: int,
    ) -> Dict[str, Any]:
        """
        评估当前对话轮次

        Args:
            conversation_history: 对话历史
            current_state: 当前咨询师状态
            round_number: 当前轮次

        Returns:
            Dict: 评估结果
        """
        context = {
            "conversation_history": conversation_history,
            "current_state": current_state,
            "round_number": round_number,
        }

        # 执行评估
        result = await self.execute(**context)

        # 添加基于规则的风险评估
        rule_based_risk = self._rule_based_risk_assessment(conversation_history)

        # 合并风险评估结果
        llm_risk = result["risk_assessment"]
        final_risk = self._merge_risk_assessments(llm_risk, rule_based_risk)
        result["risk_assessment"] = final_risk

        # 记录评估历史
        self.evaluation_history.append(
            {
                "round": round_number,
                "evaluation": result,
                "timestamp": None,  # 可以添加时间戳
            }
        )

        return result

    def _rule_based_risk_assessment(
        self, conversation_history: List[ConversationMessage]
    ) -> RiskAssessment:
        """基于规则的风险评估"""
        if not conversation_history:
            return RiskAssessment()

        # 获取最近的学生消息
        recent_student_messages = [
            msg.content for msg in conversation_history[-5:] if msg.role == "student"
        ]

        all_content = " ".join(recent_student_messages)
        return self.assess_risk(all_content)

    def _merge_risk_assessments(
        self, llm_risk: Dict[str, Any], rule_risk: RiskAssessment
    ) -> Dict[str, Any]:
        """合并LLM和规则的风险评估结果"""
        # 取较高的风险等级
        merged_risk = {
            "overall_risk_level": max(
                llm_risk.get("overall_risk_level", 0), rule_risk.overall_risk
            ),
            "suicide_risk": max(
                llm_risk.get("suicide_risk", 0), rule_risk.suicide_risk
            ),
            "self_harm_risk": max(
                llm_risk.get("self_harm_risk", 0), rule_risk.self_harm_risk
            ),
            "harm_others_risk": max(
                llm_risk.get("harm_others_risk", 0), rule_risk.harm_others_risk
            ),
            "emergency_required": (
                llm_risk.get("emergency_required", False)
                or rule_risk.emergency_required
            ),
        }

        # 合并风险指标
        llm_indicators = llm_risk.get("risk_indicators", [])
        rule_indicators = rule_risk.risk_indicators
        merged_risk["risk_indicators"] = list(set(llm_indicators + rule_indicators))

        # 合并描述
        llm_desc = llm_risk.get("risk_description", "")
        rule_desc = (
            f"检测到风险关键词: {', '.join(rule_indicators)}" if rule_indicators else ""
        )

        descriptions = [desc for desc in [llm_desc, rule_desc] if desc]
        merged_risk["risk_description"] = "; ".join(descriptions)

        return merged_risk

    def should_terminate_session(self, risk_assessment: Dict[str, Any]) -> bool:
        """判断是否应该终止会话"""
        return (
            risk_assessment.get("emergency_required", False)
            or risk_assessment.get("overall_risk_level", 0)
            >= SystemConfig.RISK_THRESHOLD
        )

    def get_state_transition_recommendation(
        self, evaluation_result: Dict[str, Any]
    ) -> Optional[CounselorState]:
        """获取状态转换建议"""
        transition_data = evaluation_result.get("state_transition", {})

        if not transition_data.get("need_transition", False):
            return None

        recommended_state_str = transition_data.get("recommended_state", "")

        # 将字符串转换为枚举
        state_mapping = {
            "introduction": CounselorState.INTRODUCTION,
            "exploration": CounselorState.EXPLORATION,
            "assessment": CounselorState.ASSESSMENT,
            "scale_recommendation": CounselorState.SCALE_RECOMMENDATION,
        }

        return state_mapping.get(recommended_state_str)

    def get_round_statistics(self) -> Dict[str, Any]:
        """获取轮次统计信息"""
        if not self.evaluation_history:
            return {}

        total_rounds = len(self.evaluation_history)
        transitions = sum(
            1
            for eval_data in self.evaluation_history
            if eval_data["evaluation"]["state_transition"]["need_transition"]
        )

        risk_levels = [
            eval_data["evaluation"]["risk_assessment"]["overall_risk_level"]
            for eval_data in self.evaluation_history
        ]

        avg_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0
        max_risk = max(risk_levels) if risk_levels else 0

        return {
            "total_rounds_evaluated": total_rounds,
            "transitions_recommended": transitions,
            "average_risk_level": round(avg_risk, 2),
            "maximum_risk_level": max_risk,
            "emergency_alerts": sum(
                1
                for eval_data in self.evaluation_history
                if eval_data["evaluation"]["risk_assessment"]["emergency_required"]
            ),
        }

    def get_conversation_quality_metrics(self) -> Dict[str, Any]:
        """获取对话质量指标"""
        if not self.evaluation_history:
            return {}

        recent_evaluations = self.evaluation_history[-5:]  # 最近5轮

        # 计算平均信任度和开放度趋势
        trust_levels = []
        openness_levels = []
        effectiveness_scores = []

        for eval_data in recent_evaluations:
            round_analysis = eval_data["evaluation"]["round_analysis"]

            # 将文本转换为数值
            trust_mapping = {
                "很低": 0.1,
                "较低": 0.3,
                "适中": 0.5,
                "较高": 0.7,
                "很高": 0.9,
            }
            openness_mapping = {
                "很低": 0.1,
                "较低": 0.3,
                "适中": 0.5,
                "较高": 0.7,
                "很高": 0.9,
            }
            effectiveness_mapping = {
                "需要改进": 0.2,
                "一般": 0.4,
                "良好": 0.7,
                "优秀": 1.0,
            }

            trust_levels.append(
                trust_mapping.get(
                    round_analysis.get("student_trust_level", "适中"), 0.5
                )
            )
            openness_levels.append(
                openness_mapping.get(
                    round_analysis.get("student_openness", "适中"), 0.5
                )
            )
            effectiveness_scores.append(
                effectiveness_mapping.get(
                    round_analysis.get("counselor_effectiveness", "良好"), 0.7
                )
            )

        return {
            "trust_trend": "improving"
            if len(trust_levels) > 1 and trust_levels[-1] > trust_levels[0]
            else "stable",
            "openness_trend": "improving"
            if len(openness_levels) > 1 and openness_levels[-1] > openness_levels[0]
            else "stable",
            "average_counselor_effectiveness": round(
                sum(effectiveness_scores) / len(effectiveness_scores), 2
            )
            if effectiveness_scores
            else 0.7,
            "current_trust_level": trust_levels[-1] if trust_levels else 0.5,
            "current_openness_level": openness_levels[-1] if openness_levels else 0.5,
        }

    def generate_session_insights(self) -> List[str]:
        """生成会话洞察"""
        if not self.evaluation_history:
            return ["暂无评估数据"]

        insights = []
        stats = self.get_round_statistics()
        quality = self.get_conversation_quality_metrics()

        # 基于统计数据生成洞察
        if stats.get("maximum_risk_level", 0) >= 3:
            insights.append("⚠️ 检测到较高风险级别，需要密切关注学生状态")

        if quality.get("trust_trend") == "improving":
            insights.append("✅ 学生信任度呈上升趋势，咨询关系建立良好")

        if quality.get("average_counselor_effectiveness", 0) >= 0.8:
            insights.append("✅ 咨询师表现优秀，专业技能运用恰当")
        elif quality.get("average_counselor_effectiveness", 0) < 0.5:
            insights.append("⚠️ 咨询师表现有待改进，建议调整咨询策略")

        if stats.get("transitions_recommended", 0) == 0:
            insights.append("📝 建议评估是否可以进入下一咨询阶段")

        return insights
