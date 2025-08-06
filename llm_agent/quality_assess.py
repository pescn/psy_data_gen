"""
质量评估Agent
对完整的咨询对话进行全面质量评估，生成结构化的评估报告
"""

from typing import Dict, List, Literal, Optional, Any

from traceloop.sdk.decorators import agent

from .base import Agent
from models import (
    ConversationMessage,
    BackgroundInfo,
)
from constants import PSYCHOLOGICAL_ISSUES_DATA, THERAPY_APPROACHES_DATA
from pydantic import BaseModel, Field


class QualityAssessmentContext(BaseModel):
    """质量评估上下文"""

    background_info: BackgroundInfo = Field(..., description="背景信息")
    conversation_history: List[ConversationMessage] = Field(
        ..., description="完整对话历史"
    )
    counseling_trajectory: Optional[Dict[str, Any]] = Field(
        None, description="咨询轨迹信息"
    )


class CoreIssueIdentification(BaseModel):
    """核心问题识别"""

    identified_issue: str = Field(..., description="咨询师识别出的核心问题")
    original_issue: str = Field(..., description="原始设定的心理问题")
    accuracy_score: float = Field(..., ge=0, le=10, description="准确性评分 (0-10)")
    consistency_check: bool = Field(..., description="是否一致")
    analysis: str = Field(..., description="准确性分析")


class StateTransition(BaseModel):
    """状态转换"""

    from_state: str = Field(..., description="起始状态")
    to_state: str = Field(..., description="目标状态")
    transition_round: int = Field(..., description="转换轮次")
    appropriateness: Literal[
        "very_appropriate", "appropriate", "questionable", "inappropriate"
    ]
    reason: str = Field(..., description="转换是否合理的分析")


class IntroductionPhase(BaseModel):
    """介绍阶段"""

    rounds_used: int
    effectiveness_score: float = Field(..., ge=0, le=10, description="0-10评分")
    key_achievements: List[str]
    missed_opportunities: List[str]


class ExplorationPhase(BaseModel):
    """探索阶段"""

    rounds_used: int
    effectiveness_score: float = Field(..., ge=0, le=10)
    information_depth: Literal["superficial", "moderate", "deep"]
    key_achievements: List[str]
    missed_opportunities: List[str]


class AssessmentPhase(BaseModel):
    """评估阶段"""

    rounds_used: int
    effectiveness_score: float = Field(..., ge=0, le=10)
    diagnosis_quality: Literal["poor", "fair", "good", "excellent"]
    key_achievements: List[str]
    missed_opportunities: List[str]


class ScaleRecommendationPhase(BaseModel):
    """量表推荐阶段"""

    rounds_used: int
    effectiveness_score: float = Field(..., ge=0, le=10)
    recommendation_appropriateness: Literal["poor", "fair", "good", "excellent"]
    key_achievements: List[str]
    missed_opportunities: List[str]


class PhaseEffectiveness(BaseModel):
    """各阶段效果"""

    introduction_phase: IntroductionPhase
    exploration_phase: ExplorationPhase
    assessment_phase: AssessmentPhase
    scale_recommendation_phase: ScaleRecommendationPhase


class CounselingTrajectory(BaseModel):
    """咨询轨迹"""

    state_transitions: List[StateTransition]
    phase_effectiveness: PhaseEffectiveness


class EmpathySkills(BaseModel):
    """共情技巧"""

    score: float = Field(..., ge=0, le=10, description="0-10评分")
    examples: List[str] = Field(..., description="体现共情的具体表达")
    improvement_areas: List[str]


class QuestioningSkills(BaseModel):
    """提问技巧"""

    score: float = Field(..., ge=0, le=10)
    open_questions_ratio: float = Field(
        ..., ge=0, le=1, description="开放式问题占比(0-1)"
    )
    question_quality: Literal["poor", "fair", "good", "excellent"]
    examples: List[str]


class ReflectionSkills(BaseModel):
    """反映技巧"""

    score: float = Field(..., ge=0, le=10)
    reflection_frequency: Literal["rare", "occasional", "frequent", "optimal"]
    reflection_accuracy: Literal["poor", "fair", "good", "excellent"]
    examples: List[str]


class TherapeuticApproach(BaseModel):
    """治疗流派"""

    approach_consistency: str = Field(
        ..., description="咨询师是否一致地运用了声明的治疗流派"
    )
    approach_appropriateness: str = Field(..., description="流派选择是否适合学生问题")
    technique_mastery: str = Field(..., description="对流派技术的掌握程度评估")


class TechniqueAnalysis(BaseModel):
    """技巧分析"""

    empathy_skills: EmpathySkills
    questioning_skills: QuestioningSkills
    reflection_skills: ReflectionSkills
    therapeutic_approach: TherapeuticApproach


class ProfessionalBoundaries(BaseModel):
    """专业边界"""

    maintained_boundaries: bool
    boundary_issues: List[str]
    professionalism_score: float = Field(..., ge=0, le=10, description="0-10评分")


class CounselingTechniques(BaseModel):
    """咨询技巧"""

    overall_score: float = Field(..., ge=0, le=10, description="0-10的总体技巧评分")
    technique_analysis: TechniqueAnalysis
    professional_boundaries: ProfessionalBoundaries


class TrustBuilding(BaseModel):
    """信任建立"""

    initial_trust: float = Field(..., ge=0, le=10, description="0-10评分")
    final_trust: float = Field(..., ge=0, le=10)
    trust_progression: Literal["deteriorated", "stagnant", "steady", "excellent"]
    trust_building_techniques: List[str]


class RapportQuality(BaseModel):
    """关系质量"""

    score: float = Field(..., ge=0, le=10)
    rapport_indicators: List[str]
    relationship_challenges: List[str]


class ClientEngagement(BaseModel):
    """来访者参与度"""

    initial_engagement: float = Field(..., ge=0, le=10)
    final_engagement: float = Field(..., ge=0, le=10)
    engagement_progression: Literal["deteriorated", "stagnant", "steady", "excellent"]
    engagement_strategies: List[str]


class TherapeuticRelationship(BaseModel):
    """治疗关系"""

    trust_building: TrustBuilding
    rapport_quality: RapportQuality
    client_engagement: ClientEngagement


class ClientInsightGained(BaseModel):
    """来访者获得的洞察"""

    score: float = Field(..., ge=0, le=10, description="0-10评分")
    insights: List[str] = Field(..., description="学生获得的洞察")
    self_awareness_improvement: str


class ProblemUnderstanding(BaseModel):
    """问题理解"""

    initial_understanding: str
    final_understanding: str
    understanding_improvement: str


class HopeAndMotivation(BaseModel):
    """希望与动机"""

    hope_level: float = Field(..., ge=0, le=10, description="0-10评分")
    motivation_to_change: float = Field(..., ge=0, le=10)
    future_orientation: str


class OutcomeAssessment(BaseModel):
    """结果评估"""

    client_insight_gained: ClientInsightGained
    problem_understanding: ProblemUnderstanding
    hope_and_motivation: HopeAndMotivation


class OverallQuality(BaseModel):
    """整体质量"""

    total_score: float = Field(..., ge=0, le=10, description="0-10的总体质量评分")
    quality_level: Literal["poor", "fair", "good", "excellent"]
    strengths: List[str] = Field(..., description="咨询的主要优点")
    weaknesses: List[str] = Field(..., description="需要改进的方面")
    critical_incidents: List[str] = Field(..., description="关键事件或转折点")
    missed_opportunities: List[str] = Field(..., description="错失的重要机会")


class Recommendations(BaseModel):
    """建议"""

    immediate_actions: List[str] = Field(..., description="立即需要采取的行动")
    future_sessions: List[str] = Field(..., description="后续咨询建议")
    counselor_development: List[str] = Field(..., description="咨询师发展建议")
    supervision_focus: List[str] = Field(..., description="督导重点建议")


class ConsistencyCheckModel(BaseModel):
    """一致性检查"""

    issue_consistency: bool
    consistency_analysis: str = Field(
        ..., description="最终诊断与初始问题设定的一致性分析"
    )
    consistency_score: float = Field(..., ge=0, le=10, description="0-10评分")


class QualityAssessmentResult(BaseModel):
    """
    咨询质量评估结果的完整数据模型
    """

    core_issue_identification: CoreIssueIdentification
    counseling_trajectory: CounselingTrajectory
    counseling_techniques: CounselingTechniques
    therapeutic_relationship: TherapeuticRelationship
    outcome_assessment: OutcomeAssessment
    overall_quality: OverallQuality
    recommendations: Recommendations
    consistency_check: ConsistencyCheckModel


quality_assessment_format_prompt = """# 输出格式
请返回一个完整的质量评估JSON对象，下面是其Interface结构：

```typescript
interface QualityAssessmentResult {
    core_issue_identification: {
        identified_issue: string; // 咨询师识别出的核心问题
        original_issue: string; // 原始设定的心理问题
        accuracy_score: number; // 准确性评分 (0-10)
        consistency_check: boolean; // 是否一致
        analysis: string; // 准确性分析
    };
    counseling_trajectory: {
        state_transitions: Array<{
            from_state: string; // 起始状态
            to_state: string; // 目标状态
            transition_round: number; // 转换轮次
            appropriateness: "very_appropriate" | "appropriate" | "questionable" | "inappropriate";
            reason: string; // 转换是否合理的分析
        }>;
        phase_effectiveness: {
            introduction_phase: {
                rounds_used: number;
                effectiveness_score: number; // 0-10评分
                key_achievements: string[];
                missed_opportunities: string[];
            };
            exploration_phase: {
                rounds_used: number;
                effectiveness_score: number;
                information_depth: "superficial" | "moderate" | "deep";
                key_achievements: string[];
                missed_opportunities: string[];
            };
            assessment_phase: {
                rounds_used: number;
                effectiveness_score: number;
                diagnosis_quality: "poor" | "fair" | "good" | "excellent";
                key_achievements: string[];
                missed_opportunities: string[];
            };
            scale_recommendation_phase: {
                rounds_used: number;
                effectiveness_score: number;
                recommendation_appropriateness: "poor" | "fair" | "good" | "excellent";
                key_achievements: string[];
                missed_opportunities: string[];
            };
        };
    };
    counseling_techniques: {
        overall_score: number; // 0-10的总体技巧评分
        technique_analysis: {
            empathy_skills: {
                score: number; // 0-10评分
                examples: string[]; // 体现共情的具体表达
                improvement_areas: string[];
            };
            questioning_skills: {
                score: number;
                open_questions_ratio: number; // 开放式问题占比(0-1)
                question_quality: "poor" | "fair" | "good" | "excellent";
                examples: string[];
            };
            reflection_skills: {
                score: number;
                reflection_frequency: "rare" | "occasional" | "frequent" | "optimal";
                reflection_accuracy: "poor" | "fair" | "good" | "excellent";
                examples: string[];
            };
            therapeutic_approach: {
                approach_consistency: string; // 咨询师是否一致地运用了声明的治疗流派
                approach_appropriateness: string; // 流派选择是否适合学生问题
                technique_mastery: string; // 对流派技术的掌握程度评估
            };
        };
        professional_boundaries: {
            maintained_boundaries: boolean;
            boundary_issues: string[];
            professionalism_score: number; // 0-10评分
        };
    };
    therapeutic_relationship: {
        trust_building: {
            initial_trust: number; // 0-10评分
            final_trust: number;
            trust_progression: "deteriorated" | "stagnant" | "steady" | "excellent";
            trust_building_techniques: string[];
        };
        rapport_quality: {
            score: number;
            rapport_indicators: string[];
            relationship_challenges: string[];
        };
        client_engagement: {
            initial_engagement: number;
            final_engagement: number;
            engagement_progression: "deteriorated" | "stagnant" | "steady" | "excellent";
            engagement_strategies: string[];
        };
    };
    outcome_assessment: {
        client_insight_gained: {
            score: number; // 0-10评分
            insights: string[]; // 学生获得的洞察
            self_awareness_improvement: string;
        };
        problem_understanding: {
            initial_understanding: string;
            final_understanding: string;
            understanding_improvement: string;
        };
        hope_and_motivation: {
            hope_level: number; // 0-10评分
            motivation_to_change: number;
            future_orientation: string;
        };
    };
    overall_quality: {
        total_score: number; // 0-10的总体质量评分
        quality_level: "poor" | "fair" | "good" | "excellent";
        strengths: string[]; // 咨询的主要优点
        weaknesses: string[]; // 需要改进的方面
        critical_incidents: string[]; // 关键事件或转折点
        missed_opportunities: string[]; // 错失的重要机会
    };
    recommendations: {
        immediate_actions: string[]; // 立即需要采取的行动
        future_sessions: string[]; // 后续咨询建议
        counselor_development: string[]; // 咨询师发展建议
        supervision_focus: string[]; // 督导重点建议
    };
    consistency_check: {
        issue_consistency: boolean;
        consistency_analysis: string; // 最终诊断与初始问题设定的一致性分析
        consistency_score: number; // 0-10评分
    };
}
```

请确保生成的 JSON 对象符合上述结构，并且所有字段都包含有效内容。
"""


@agent(name="质量评估 Agent", method_name="execute")
class QualityAssessmentAgent(Agent[QualityAssessmentContext, QualityAssessmentResult]):
    """
    质量评估Agent
    对完整的咨询会话进行综合质量评估
    """

    context_class = QualityAssessmentContext
    result_class = QualityAssessmentResult

    def prompt(self, context: QualityAssessmentContext) -> str:
        """
        构建质量评估的提示词
        """
        base_prompt = """# Role: 心理咨询质量评估专家
你是一名资深的心理咨询督导专家，请对以下完整的心理咨询对话进行全面的质量评估。请基于专业的心理咨询标准进行评估，分析咨询师在各个阶段的表现，确保评估结果的客观性和建设性。

## 评估任务：
请从以下几个维度对这次咨询进行专业评估：

1. **核心问题识别**：咨询师是否准确识别了学生的核心心理问题
2. **咨询轨迹分析**：状态转换是否合理，各阶段是否达到预期目标
3. **咨询技巧评估**：咨询师的专业技能运用是否恰当
4. **治疗关系质量**：是否建立了良好的咨询关系
5. **问题解决效果**：是否帮助学生获得洞察或改善
6. **一致性检查**：最终结果是否与初始设定的心理问题一致

## 评估标准说明：
- 评分范围：0-10分，其中0-3为差，4-6为一般，7-8为良好，9-10为优秀
- 重点关注咨询的专业性、有效性和伦理性
- 考虑学生的具体背景和问题特点
- 评估要公正客观，既要指出优点也要指出不足
"""

        # 构建完整的提示词
        prompt = (
            base_prompt
            + quality_assessment_format_prompt
            + self._format_background_info(context.background_info)
            + self._format_conversation_history(context.conversation_history)
            + self._format_counseling_trajectory(context.counseling_trajectory)
        )
        return prompt

    def _format_background_info(self, background_info: BackgroundInfo) -> str:
        """格式化背景信息"""
        if not background_info:
            return "## 背景信息：\n背景信息缺失"

        student = background_info.student_info
        counselor = background_info.counselor_info

        issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
            student.current_psychological_issue, {}
        )
        issue_str = (
            f"{issue_data.get('name', '未知')} - {issue_data.get('description', '')}"
            if issue_data
            else student.current_psychological_issue
        )
        approach_data = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {})

        return f"""## 背景信息：
### 学生背景：
- 基本信息：{student.age}岁，{student.grade}，{student.major}
- 性格特征：{", ".join(student.personality_traits)}
- 家庭背景：{student.family_background}
- 心理问题：{issue_str}
- 症状描述：{student.symptom_description}
- 深层信息：{student.hidden_personal_info}

### 咨询师背景：
- 咨询流派：{approach_data.get("name", "未知")} - {approach_data.get("description", "")}
- 沟通风格：{counselor.communication_style}
- 专业领域：{", ".join(counselor.specialization)}

### 原始设定心理问题：
{issue_data.get("name", "未知问题")}
"""

    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """格式化对话历史"""
        if not history:
            return "## 完整对话记录：\n无对话记录"

        formatted = []
        for i, msg in enumerate(history):
            role_name = "学生" if msg.role == "student" else "咨询师"
            state_info = (
                f"[{msg.state}]" if msg.state and msg.role == "counselor" else ""
            )
            emotion_info = (
                f"(情绪: {msg.emotion.value})"
                if msg.emotion and msg.role == "counselor"
                else ""
            )

            formatted.append(
                f"{(i + 1):2d}. {role_name}{state_info}{emotion_info}: {msg.content}"
            )

        return "## 完整对话记录：\n" + "\n".join(formatted) + "\n\n"

    def _format_counseling_trajectory(
        self, trajectory: Optional[Dict[str, Any]]
    ) -> str:
        """格式化咨询轨迹"""
        if not trajectory:
            return "## 咨询轨迹信息：\n无轨迹信息"

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

        return (
            "## 咨询轨迹信息：\n"
            + ("\n".join(formatted) if formatted else "轨迹信息不完整")
            + "\n\n"
        )


if __name__ == "__main__":
    import asyncio
    import json

    # 示例背景信息和对话历史
    background_info = BackgroundInfo(
        student_info={
            "age": 20,
            "gender": "女",
            "grade": "大二",
            "major": "心理学",
            "family_background": "父母离异，和母亲同住",
            "personality_traits": ["内向", "敏感", "完美主义"],
            "psychological_profile": "有轻度焦虑和抑郁倾向",
            "hidden_personal_info": "曾经历过校园欺凌，导致自尊心受损",
            "symptom_description": "经常感到焦虑，难以集中注意，晚上失眠",
            "current_psychological_issue": "抑郁情绪",
        },
        counselor_info={
            "therapy_approach": "认知行为疗法",
            "communication_style": "温和而坚定，善于倾听",
            "specialization": ["焦虑症", "抑郁症", "人际关系问题"],
        },
        initial_question="我最近总是感到很焦虑，不知道该怎么办。",
    )
    conversation_history = [
        ConversationMessage(
            role="student",
            content="我最近总是感到很焦虑，不知道该怎么办。",
            state="introduction",
            emotion="anxious",
            round_number=1,
        ),
        ConversationMessage(
            role="counselor",
            content="可以告诉我更多关于你的焦虑吗？",
            state="introduction",
            emotion="calm",
            round_number=1,
        ),
        ConversationMessage(
            role="student",
            content="我觉得自己总是担心考试和未来的事情。",
            state="exploration",
            emotion="anxious",
            round_number=2,
        ),
    ]
    counseling_trajectory = {
        "state_transitions": [
            {
                "from_state": "introduction",
                "to_state": "exploration",
                "transition_round": 2,
                "appropriateness": "very_appropriate",
                "reason": "成功引导学生深入探讨焦虑问题",
            }
        ],
        "rounds_per_state": {"introduction": 1, "exploration": 1},
        "total_rounds": 2,
    }
    context = QualityAssessmentContext(
        background_info=background_info,
        conversation_history=conversation_history,
        counseling_trajectory=counseling_trajectory,
    )
    agent = QualityAssessmentAgent()
    # 执行评估
    assessment_result = asyncio.run(agent.execute(context))
    print(json.dumps(assessment_result.model_dump(), ensure_ascii=False, indent=2))
