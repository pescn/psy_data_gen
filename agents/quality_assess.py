"""
质量评估Agent
对完整的咨询对话进行全面质量评估，生成结构化的评估报告
"""

from typing import Dict, List, Optional, Any
import statistics

from agents.base import Agent
from models import (
    ConversationMessage,
    QualityAssessment,
    CounselingTrajectory,
    StateTransition,
    BackgroundInfo,
    CounselorState,
)
from constants import PSYCHOLOGICAL_ISSUES_DATA, THERAPY_APPROACHES_DATA


class QualityAssessmentAgent(Agent):
    """
    质量评估Agent
    对完整的咨询会话进行综合质量评估
    """

    def _init_config(self, **kwargs):
        """初始化质量评估Agent配置"""
        # LLM配置（子类需要设置）
        # self.llm_client = kwargs.get('llm_client')
        # self.model = kwargs.get('model', 'gpt-4')

        # 构建提示词模板
        self._build_prompt_template()

    def _build_prompt_template(self):
        """构建提示词模板"""
        self.prompt_template = """你是一名资深的心理咨询督导专家，请对以下完整的心理咨询对话进行全面的质量评估。

背景信息：
{background_info}

完整对话记录：
{conversation_history}

咨询轨迹信息：
{counseling_trajectory}

评估任务：
请从以下几个维度对这次咨询进行专业评估：

1. **核心问题识别**：咨询师是否准确识别了学生的核心心理问题
2. **咨询轨迹分析**：状态转换是否合理，各阶段是否达到预期目标
3. **咨询技巧评估**：咨询师的专业技能运用是否恰当
4. **治疗关系质量**：是否建立了良好的咨询关系
5. **问题解决效果**：是否帮助学生获得洞察或改善
6. **一致性检查**：最终结果是否与初始设定的心理问题一致

请以JSON格式返回详细的评估结果：
{{
    "core_issue_identification": {{
        "identified_issue": "咨询师识别出的核心问题",
        "original_issue": "{original_issue}",
        "accuracy_score": 0-10的评分,
        "consistency_check": true/false,
        "analysis": "准确性分析"
    }},
    "counseling_trajectory": {{
        "state_transitions": [
            {{
                "from_state": "起始状态",
                "to_state": "目标状态", 
                "transition_round": 转换轮次,
                "appropriateness": "very_appropriate/appropriate/questionable/inappropriate",
                "reason": "转换是否合理的分析"
            }}
        ],
        "phase_effectiveness": {{
            "introduction_phase": {{
                "rounds_used": 轮次数,
                "effectiveness_score": 0-10评分,
                "key_achievements": ["达成的目标1", "达成的目标2"],
                "missed_opportunities": ["错失的机会1", "错失的机会2"]
            }},
            "exploration_phase": {{
                "rounds_used": 轮次数,
                "effectiveness_score": 0-10评分,
                "information_depth": "superficial/moderate/deep",
                "key_achievements": ["达成的目标1", "达成的目标2"],
                "missed_opportunities": ["错失的机会1", "错失的机会2"]
            }},
            "assessment_phase": {{
                "rounds_used": 轮次数,
                "effectiveness_score": 0-10评分,
                "diagnosis_quality": "poor/fair/good/excellent",
                "key_achievements": ["达成的目标1", "达成的目标2"],
                "missed_opportunities": ["错失的机会1", "错失的机会2"]
            }},
            "scale_recommendation_phase": {{
                "rounds_used": 轮次数,
                "effectiveness_score": 0-10评分,
                "recommendation_appropriateness": "poor/fair/good/excellent",
                "key_achievements": ["达成的目标1", "达成的目标2"],
                "missed_opportunities": ["错失的机会1", "错失的机会2"]
            }}
        }}
    }},
    "counseling_techniques": {{
        "overall_score": 0-10的总体技巧评分,
        "technique_analysis": {{
            "empathy_skills": {{
                "score": 0-10评分,
                "examples": ["体现共情的具体表达1", "体现共情的具体表达2"],
                "improvement_areas": ["需要改进的方面"]
            }},
            "questioning_skills": {{
                "score": 0-10评分,
                "open_questions_ratio": 开放式问题占比(0-1),
                "question_quality": "poor/fair/good/excellent",
                "examples": ["好问题示例1", "好问题示例2"]
            }},
            "reflection_skills": {{
                "score": 0-10评分,
                "reflection_frequency": "rare/occasional/frequent/optimal",
                "reflection_accuracy": "poor/fair/good/excellent",
                "examples": ["反映技术示例1", "反映技术示例2"]
            }},
            "therapeutic_approach": {{
                "approach_consistency": "咨询师是否一致地运用了声明的治疗流派",
                "approach_appropriateness": "流派选择是否适合学生问题",
                "technique_mastery": "对流派技术的掌握程度评估"
            }}
        }},
        "professional_boundaries": {{
            "maintained_boundaries": true/false,
            "boundary_issues": ["边界问题描述"],
            "professionalism_score": 0-10评分
        }}
    }},
    "therapeutic_relationship": {{
        "trust_building": {{
            "initial_trust": 0-10评分,
            "final_trust": 0-10评分,
            "trust_progression": "deteriorated/stagnant/steady/excellent",
            "trust_building_techniques": ["建立信任的有效技术"]
        }},
        "rapport_quality": {{
            "score": 0-10评分,
            "rapport_indicators": ["良好关系的指标"],
            "relationship_challenges": ["关系建立中的挑战"]
        }},
        "client_engagement": {{
            "initial_engagement": 0-10评分,
            "final_engagement": 0-10评分,
            "engagement_progression": "deteriorated/stagnant/steady/excellent",
            "engagement_strategies": ["提高参与度的策略"]
        }}
    }},
    "outcome_assessment": {{
        "client_insight_gained": {{
            "score": 0-10评分,
            "insights": ["学生获得的洞察1", "学生获得的洞察2"],
            "self_awareness_improvement": "学生自我觉察的提升程度"
        }},
        "problem_understanding": {{
            "initial_understanding": "学生对问题的初始理解水平",
            "final_understanding": "学生对问题的最终理解水平",
            "understanding_improvement": "理解水平的提升程度"
        }},
        "hope_and_motivation": {{
            "hope_level": 0-10评分,
            "motivation_to_change": 0-10评分,
            "future_orientation": "学生对未来的态度变化"
        }}
    }},
    "overall_quality": {{
        "total_score": 0-10的总体质量评分,
        "quality_level": "poor/fair/good/excellent",
        "strengths": ["咨询的主要优点1", "咨询的主要优点2", "咨询的主要优点3"],
        "weaknesses": ["需要改进的方面1", "需要改进的方面2"],
        "critical_incidents": ["关键事件或转折点"],
        "missed_opportunities": ["错失的重要机会"]
    }},
    "recommendations": {{
        "immediate_actions": ["立即需要采取的行动"],
        "future_sessions": ["后续咨询建议"],
        "counselor_development": ["咨询师发展建议"],
        "supervision_focus": ["督导重点建议"]
    }},
    "consistency_check": {{
        "issue_consistency": true/false,
        "consistency_analysis": "最终诊断与初始问题设定的一致性分析",
        "consistency_score": 0-10评分
    }}
}}

评估标准说明：
- 评分范围：0-10分，其中0-3为差，4-6为一般，7-8为良好，9-10为优秀
- 重点关注咨询的专业性、有效性和伦理性
- 考虑学生的具体背景和问题特点
- 评估要公正客观，既要指出优点也要指出不足
- 特别关注咨询师是否遵循了其声明的治疗流派的原则和技术

请基于专业的心理咨询标准进行评估，确保评估结果的客观性和建设性。
"""

    def _build_prompt(self, **context) -> str:
        """构建具体的提示词"""
        # 获取上下文信息
        background_info = context.get("background_info")
        conversation_history = context.get("conversation_history", [])
        counseling_trajectory = context.get("counseling_trajectory", {})

        # 格式化背景信息
        background_text = self._format_background_info(background_info)

        # 格式化对话历史
        conversation_text = self._format_conversation_history(conversation_history)

        # 格式化咨询轨迹
        trajectory_text = self._format_counseling_trajectory(counseling_trajectory)

        # 获取原始问题
        original_issue = ""
        if background_info and background_info.student_info:
            issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
                background_info.student_info.current_psychological_issue, {}
            )
            original_issue = issue_data.get("name", "未知问题")

        return self.prompt_template.format(
            background_info=background_text,
            conversation_history=conversation_text,
            counseling_trajectory=trajectory_text,
            original_issue=original_issue,
        )

    def _format_background_info(self, background_info: BackgroundInfo) -> str:
        """格式化背景信息"""
        if not background_info:
            return "背景信息缺失"

        student = background_info.student_info
        counselor = background_info.counselor_info

        issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
            student.current_psychological_issue, {}
        )
        approach_data = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {})

        return f"""
学生背景：
- 基本信息：{student.name}，{student.age}岁，{student.grade}，{student.major}
- 性格特征：{", ".join(student.personality_traits)}
- 家庭背景：{student.family_background}
- 心理问题：{issue_data.get("name", "未知")} - {issue_data.get("description", "")}
- 症状描述：{student.symptom_description}
- 深层信息：{student.hidden_personal_info}

咨询师背景：
- 基本信息：{counselor.name}，{counselor.experience_years}年经验
- 咨询流派：{approach_data.get("name", "未知")} - {approach_data.get("description", "")}
- 沟通风格：{counselor.communication_style}
- 专业领域：{", ".join(counselor.specialization)}
"""

    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """格式化对话历史"""
        if not history:
            return "无对话记录"

        formatted = []
        for i, msg in enumerate(history, 1):
            role_name = "学生" if msg.role == "student" else "咨询师"
            state_info = f"[{msg.state}]" if msg.state else ""
            emotion_info = f"(情绪: {msg.emotion.value})" if msg.emotion else ""

            formatted.append(
                f"{i:2d}. {role_name}{state_info}{emotion_info}: {msg.content}"
            )

        return "\n".join(formatted)

    def _format_counseling_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """格式化咨询轨迹"""
        if not trajectory:
            return "无轨迹信息"

        formatted = []

        # 状态转换信息
        if "state_transitions" in trajectory:
            formatted.append("状态转换记录：")
            for i, transition in enumerate(trajectory["state_transitions"], 1):
                formatted.append(
                    f"  {i}. 第{transition.get('round', '?')}轮: "
                    f"{transition.get('from_state', '?')} → {transition.get('to_state', '?')} "
                    f"(原因: {transition.get('reason', '未知')})"
                )

        # 各状态持续轮数
        if "rounds_per_state" in trajectory:
            formatted.append("\n各阶段持续轮数：")
            for state, rounds in trajectory["rounds_per_state"].items():
                formatted.append(f"  - {state}: {rounds}轮")

        # 总轮数
        if "total_rounds" in trajectory:
            formatted.append(f"\n总对话轮数：{trajectory['total_rounds']}")

        return "\n".join(formatted) if formatted else "轨迹信息不完整"

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            parsed_data = self._safe_json_parse(response)

            # 验证必需的字段
            required_fields = [
                "core_issue_identification",
                "counseling_trajectory",
                "counseling_techniques",
                "overall_quality",
            ]
            for field in required_fields:
                if field not in parsed_data:
                    parsed_data[field] = {}

            return parsed_data

        except Exception as e:
            raise ValueError(f"Failed to parse quality assessment response: {str(e)}")

    async def assess_conversation_quality(
        self,
        background_info: BackgroundInfo,
        conversation_history: List[ConversationMessage],
        counseling_trajectory: Optional[Dict[str, Any]] = None,
    ) -> QualityAssessment:
        """
        评估对话质量

        Args:
            background_info: 背景信息
            conversation_history: 完整对话历史
            counseling_trajectory: 咨询轨迹信息

        Returns:
            QualityAssessment: 质量评估结果
        """
        # 分析咨询轨迹
        if not counseling_trajectory:
            counseling_trajectory = self._analyze_counseling_trajectory(
                conversation_history
            )

        context = {
            "background_info": background_info,
            "conversation_history": conversation_history,
            "counseling_trajectory": counseling_trajectory,
        }

        # 执行评估
        result = await self.execute(**context)

        # 计算量化指标
        quantitative_metrics = self._calculate_quantitative_metrics(
            conversation_history, result
        )

        # 构建质量评估对象
        quality_assessment = self._build_quality_assessment(
            result, quantitative_metrics, counseling_trajectory
        )

        return quality_assessment

    def _analyze_counseling_trajectory(
        self, conversation_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """分析咨询轨迹"""
        if not conversation_history:
            return {}

        # 统计各状态的轮数
        state_rounds = {}
        current_state = None
        state_transitions = []

        for i, msg in enumerate(conversation_history):
            if msg.role == "counselor" and msg.state:
                if current_state != msg.state:
                    if current_state:
                        # 记录状态转换
                        state_transitions.append(
                            {
                                "from_state": current_state,
                                "to_state": msg.state,
                                "round": i + 1,
                                "reason": "自动检测",
                            }
                        )
                    current_state = msg.state

                # 统计轮数
                state_rounds[msg.state] = state_rounds.get(msg.state, 0) + 1

        return {
            "state_transitions": state_transitions,
            "rounds_per_state": state_rounds,
            "total_rounds": len(conversation_history),
        }

    def _calculate_quantitative_metrics(
        self,
        conversation_history: List[ConversationMessage],
        assessment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """计算量化指标"""
        if not conversation_history:
            return {}

        # 基础统计
        total_messages = len(conversation_history)
        student_messages = [
            msg for msg in conversation_history if msg.role == "student"
        ]
        counselor_messages = [
            msg for msg in conversation_history if msg.role == "counselor"
        ]

        # 消息长度分析
        student_lengths = [len(msg.content) for msg in student_messages]
        counselor_lengths = [len(msg.content) for msg in counselor_messages]

        # 情绪变化分析
        emotions = [msg.emotion.value for msg in student_messages if msg.emotion]
        unique_emotions = len(set(emotions)) if emotions else 0

        # 问句分析（简单统计）
        counselor_questions = sum(
            1
            for msg in counselor_messages
            if "?" in msg.content
            or any(
                word in msg.content
                for word in ["什么", "怎么", "为什么", "能说", "可以"]
            )
        )

        return {
            "total_messages": total_messages,
            "student_message_count": len(student_messages),
            "counselor_message_count": len(counselor_messages),
            "avg_student_message_length": statistics.mean(student_lengths)
            if student_lengths
            else 0,
            "avg_counselor_message_length": statistics.mean(counselor_lengths)
            if counselor_lengths
            else 0,
            "emotional_range": unique_emotions,
            "counselor_question_ratio": counselor_questions / len(counselor_messages)
            if counselor_messages
            else 0,
            "conversation_balance": len(student_messages) / len(counselor_messages)
            if counselor_messages
            else 0,
        }

    def _build_quality_assessment(
        self,
        llm_result: Dict[str, Any],
        metrics: Dict[str, Any],
        trajectory: Dict[str, Any],
    ) -> QualityAssessment:
        """构建质量评估对象"""
        # 提取核心信息
        core_issue = llm_result.get("core_issue_identification", {}).get(
            "identified_issue", "未识别"
        )

        # 提取关键转折点
        key_transitions = []
        overall_quality = llm_result.get("overall_quality", {})
        if "critical_incidents" in overall_quality:
            key_transitions.extend(overall_quality["critical_incidents"])

        # 构建咨询轨迹
        state_transitions = []
        if trajectory.get("state_transitions"):
            for trans in trajectory["state_transitions"]:
                state_transitions.append(
                    StateTransition(
                        from_state=CounselorState(
                            trans.get("from_state", "introduction")
                        ),
                        to_state=CounselorState(trans.get("to_state", "introduction")),
                        transition_round=trans.get("round", 0),
                        reason=trans.get("reason", "未知"),
                    )
                )

        counseling_trajectory = CounselingTrajectory(
            state_transitions=state_transitions,
            rounds_per_state=trajectory.get("rounds_per_state", {}),
            total_rounds=trajectory.get("total_rounds", 0),
        )

        # 提取评分
        counseling_techniques_score = float(
            llm_result.get("counseling_techniques", {}).get("overall_score", 7.0)
        )

        # 提取最终结果
        final_result = self._extract_final_result(llm_result)

        # 一致性检查
        consistency_check = llm_result.get("consistency_check", {})
        issue_consistency = consistency_check.get("issue_consistency", True)

        # 总体质量评分
        overall_score = float(overall_quality.get("total_score", 7.0))

        # 改进建议
        improvement_suggestions = []
        if "recommendations" in llm_result:
            recommendations = llm_result["recommendations"]
            improvement_suggestions.extend(
                recommendations.get("counselor_development", [])
            )
            improvement_suggestions.extend(recommendations.get("supervision_focus", []))

        return QualityAssessment(
            core_issue=core_issue,
            key_transitions=key_transitions,
            counseling_trajectory=counseling_trajectory,
            counseling_techniques_score=counseling_techniques_score,
            final_result=final_result,
            issue_consistency=issue_consistency,
            overall_quality_score=overall_score,
            improvement_suggestions=improvement_suggestions,
        )

    def _extract_final_result(self, llm_result: Dict[str, Any]) -> str:
        """提取最终结果描述"""
        components = []

        # 核心问题识别
        core_issue = llm_result.get("core_issue_identification", {})
        if core_issue.get("identified_issue"):
            components.append(f"识别核心问题：{core_issue['identified_issue']}")

        # 治疗效果
        outcome = llm_result.get("outcome_assessment", {})
        if outcome.get("client_insight_gained", {}).get("insights"):
            insights = outcome["client_insight_gained"]["insights"]
            components.append(f"学生获得洞察：{'; '.join(insights[:2])}")

        # 总体质量
        overall = llm_result.get("overall_quality", {})
        quality_level = overall.get("quality_level", "一般")
        components.append(f"咨询质量：{quality_level}")

        # 主要优势
        if overall.get("strengths"):
            strengths = overall["strengths"][:2]
            components.append(f"主要优势：{'; '.join(strengths)}")

        return "; ".join(components) if components else "评估完成"

    def generate_assessment_summary(
        self, assessment: QualityAssessment
    ) -> Dict[str, Any]:
        """生成评估摘要"""
        return {
            "overall_rating": self._get_rating_label(assessment.overall_quality_score),
            "techniques_rating": self._get_rating_label(
                assessment.counseling_techniques_score
            ),
            "consistency_status": "一致" if assessment.issue_consistency else "不一致",
            "total_rounds": assessment.counseling_trajectory.total_rounds,
            "state_transitions_count": len(
                assessment.counseling_trajectory.state_transitions
            ),
            "key_strengths": assessment.improvement_suggestions[:2]
            if assessment.improvement_suggestions
            else [],
            "improvement_areas": assessment.improvement_suggestions[2:]
            if len(assessment.improvement_suggestions) > 2
            else [],
            "recommendation_priority": "high"
            if assessment.overall_quality_score < 6
            else "medium"
            if assessment.overall_quality_score < 8
            else "low",
        }

    def _get_rating_label(self, score: float) -> str:
        """将数值评分转换为文字标签"""
        if score >= 9:
            return "优秀"
        elif score >= 7:
            return "良好"
        elif score >= 5:
            return "一般"
        else:
            return "需要改进"

    def compare_with_standards(self, assessment: QualityAssessment) -> Dict[str, Any]:
        """与标准进行对比"""
        # 预设的质量标准
        standards = {
            "minimum_rounds": 15,
            "maximum_rounds": 50,
            "minimum_techniques_score": 6.0,
            "minimum_overall_score": 6.0,
            "required_state_transitions": 2,
        }

        comparison = {}

        # 轮数检查
        total_rounds = assessment.counseling_trajectory.total_rounds
        comparison["rounds_check"] = {
            "actual": total_rounds,
            "standard": f"{standards['minimum_rounds']}-{standards['maximum_rounds']}",
            "meets_standard": standards["minimum_rounds"]
            <= total_rounds
            <= standards["maximum_rounds"],
        }

        # 技巧评分检查
        comparison["techniques_check"] = {
            "actual": assessment.counseling_techniques_score,
            "standard": f">={standards['minimum_techniques_score']}",
            "meets_standard": assessment.counseling_techniques_score
            >= standards["minimum_techniques_score"],
        }

        # 总体质量检查
        comparison["quality_check"] = {
            "actual": assessment.overall_quality_score,
            "standard": f">={standards['minimum_overall_score']}",
            "meets_standard": assessment.overall_quality_score
            >= standards["minimum_overall_score"],
        }

        # 状态转换检查
        transitions_count = len(assessment.counseling_trajectory.state_transitions)
        comparison["transitions_check"] = {
            "actual": transitions_count,
            "standard": f">={standards['required_state_transitions']}",
            "meets_standard": transitions_count
            >= standards["required_state_transitions"],
        }

        # 一致性检查
        comparison["consistency_check"] = {
            "actual": assessment.issue_consistency,
            "standard": True,
            "meets_standard": assessment.issue_consistency,
        }

        # 总体合规性
        all_checks = [check["meets_standard"] for check in comparison.values()]
        comparison["overall_compliance"] = {
            "passes_all_standards": all(all_checks),
            "compliance_rate": sum(all_checks) / len(all_checks),
            "failed_standards": [
                key
                for key, check in comparison.items()
                if not check["meets_standard"] and key != "overall_compliance"
            ],
        }

        return comparison
