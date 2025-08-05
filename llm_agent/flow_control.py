"""
改进的流程控制Agent
负责分析对话进展，判断状态转换时机，评估风险等级，并准确评估学生心理状态指标
"""

from typing import Dict, List, Literal, Optional, Any

from traceloop.sdk.decorators import agent

from .base import Agent, RiskAssessmentMixin
from models import (
    ConversationMessage,
    CounselorState,
    EmotionState,
    BackgroundInfo,
)
from constants import (
    STATE_TRANSITION_GUIDE,
    PSYCHOLOGICAL_ISSUES_DATA,
    THERAPY_APPROACHES_DATA,
)
from pydantic import BaseModel, Field


# 定义状态机的有向无环图
STATE_TRANSITION_GRAPH = {
    CounselorState.INTRODUCTION: [CounselorState.EXPLORATION],
    CounselorState.EXPLORATION: [CounselorState.ASSESSMENT],
    CounselorState.ASSESSMENT: [CounselorState.SCALE_RECOMMENDATION],
    CounselorState.SCALE_RECOMMENDATION: [],  # 终止状态
}


class FlowControlContext(BaseModel):
    """流程控制Agent上下文"""

    conversation_history: List[ConversationMessage] = Field(..., description="对话历史")
    current_state: CounselorState = Field(..., description="当前咨询师状态")
    current_state_round: int = Field(..., description="当前状态持续轮数")
    round_number: int = Field(..., description="当前轮次")
    background_info: Optional[BackgroundInfo] = Field(None, description="背景信息")
    # 添加当前学生状态信息用于参考
    current_student_trust_level: float = Field(0.1, description="当前学生信任度")
    current_student_openness_level: float = Field(0.2, description="当前学生开放度")
    current_student_information_revealed: float = Field(
        0.1, description="当前信息透露度"
    )
    current_student_emotion: EmotionState = Field(
        EmotionState.ANXIOUS, description="当前学生情绪"
    )


class StudentStateAnalysis(BaseModel):
    """学生状态分析"""

    trust_level: float = Field(..., ge=0, le=1, description="信任度评分 (0-1)")
    trust_level_change: Literal[
        "显著下降", "轻微下降", "保持稳定", "轻微上升", "显著上升"
    ] = Field(..., description="信任度变化趋势")
    trust_analysis: str = Field(..., description="信任度分析说明")

    openness_level: float = Field(..., ge=0, le=1, description="开放度评分 (0-1)")
    openness_change: Literal[
        "显著下降", "轻微下降", "保持稳定", "轻微上升", "显著上升"
    ] = Field(..., description="开放度变化趋势")
    openness_analysis: str = Field(..., description="开放度分析说明")

    information_revealed: float = Field(
        ..., ge=0, le=1, description="信息透露度评分 (0-1)"
    )
    information_change: Literal[
        "显著减少", "轻微减少", "保持稳定", "轻微增加", "显著增加"
    ] = Field(..., description="信息透露变化趋势")
    information_analysis: str = Field(..., description="信息透露分析说明")

    current_emotion: EmotionState = Field(..., description="当前主要情绪状态")
    emotion_change: Literal[
        "明显恶化", "轻微恶化", "保持稳定", "轻微改善", "明显改善"
    ] = Field(..., description="情绪变化趋势")
    emotion_analysis: str = Field(..., description="情绪状态分析说明")

    resistance_level: float = Field(..., ge=0, le=1, description="抗拒程度评分 (0-1)")
    avoidance_tendency: float = Field(..., ge=0, le=1, description="回避倾向评分 (0-1)")


class RoundAnalysis(BaseModel):
    """轮次分析"""

    current_round: int = Field(..., description="当前轮次")
    information_saturation: Literal[
        "不充分(0-0.3)", "部分充分(0.3-0.6)", "基本充分(0.6-0.8)", "充分(0.8-1.0)"
    ] = Field(..., description="信息饱和度")
    counselor_effectiveness: Literal["需要改进", "一般", "良好", "优秀"] = Field(
        ..., description="咨询师有效性"
    )
    stage_completion: float = Field(..., ge=0, le=1, description="当前阶段完成度 (0-1)")


class StateTransition(BaseModel):
    """状态转换建议"""

    need_transition: bool = Field(..., description="是否需要状态转换")
    current_state: str = Field(..., description="当前状态")
    recommended_state: Optional[
        Literal["引入与建立关系阶段", "深入探索阶段", "评估诊断阶段", "量表推荐阶段"]
    ] = Field(None, description="推荐的下一个状态")
    transition_reason: str = Field(..., description="转换理由")
    confidence_level: Literal["低", "中", "高"] = Field(
        ..., description="转换建议的置信度"
    )


class RiskAssessment(BaseModel):
    """风险评估"""

    overall_risk_level: int = Field(..., ge=0, le=5, description="总体风险等级 (0-5)")
    suicide_risk: int = Field(..., ge=0, le=5, description="自杀风险等级 (0-5)")
    self_harm_risk: int = Field(..., ge=0, le=5, description="自残风险等级 (0-5)")
    harm_others_risk: int = Field(..., ge=0, le=5, description="伤害他人风险等级 (0-5)")
    risk_indicators: List[str] = Field(..., description="检测到的风险关键词或表达")
    emergency_required: bool = Field(..., description="是否需要紧急干预")
    risk_description: str = Field(..., description="风险情况的详细描述")


class FlowControlResult(BaseModel):
    """流程控制评估结果"""

    student_state_analysis: StudentStateAnalysis = Field(
        ..., description="学生状态分析"
    )
    round_analysis: RoundAnalysis = Field(..., description="轮次分析")
    state_transition: StateTransition = Field(..., description="状态转换建议")
    risk_assessment: RiskAssessment = Field(..., description="风险评估")
    improvement_suggestions: List[str] = Field(..., description="对咨询师的改进建议")
    next_focus: str = Field(..., description="下一轮咨询应该关注的重点")


format_prompt = """
```typescript
/** 学生状态分析 */
interface StudentStateAnalysis {
  trust_level: number;  // 信任度评分 (0.0-1.0)
  trust_level_change: '显著下降' | '轻微下降' | '保持稳定' | '轻微上升' | '显著上升';  // 信任度变化趋势
  trust_analysis: string;  // 信任度分析说明，要有简要的对话内容支撑，60字以内

  openness_level: number;  // 开放度评分 (0.0-1.0)
  openness_change: '显著下降' | '轻微下降' | '保持稳定' | '轻微上升' | '显著上升';  // 开放度变化趋势
  openness_analysis: string; // 简要的开放度分析说明，30字以内
  
  information_revealed: number; // 信息透露度评分 (0.0-1.0)
  information_change: '显著减少' | '轻微减少' | '保持稳定' | '轻微增加' | '显著增加';  // 信息透露变化趋势
  information_analysis: string;  // 简要的信息透露分析说明，30字以内
  
  current_emotion: string;  // 当前情绪状态
  emotion_change: '明显恶化' | '轻微恶化' | '保持稳定' | '轻微改善' | '明显改善';  // 情绪变化趋势
  emotion_analysis: string;  // 情绪状态分析说明，60字以内
  
  resistance_level: number;  // 抵抗水平 (0.0-1.0)，评估学生对咨询师建议的抵抗程度
  avoidance_tendency: number;  // 回避倾向 (0.0-1.0)，评估学生对敏感话题的回避程度
}

/** 咨询轮次分析 */
interface RoundAnalysis {
  current_round: number;  // 当前轮次
  information_saturation: '不充分(0-0.3)' | '部分充分(0.3-0.6)' | '基本充分(0.6-0.8)' | '充分(0.8-1.0)';  // 信息饱和度
  counselor_effectiveness: '需要改进' | '一般' | '良好' | '优秀';  // 咨询师效果评估
  stage_completion: number;  // 当前阶段完成度 (0.0-1.0)
}

/** 状态转换评估 */
interface StateTransition {
  need_transition: boolean;  // 是否需要状态转换
  current_state: "引入与建立关系阶段" | "深入探索阶段" | "评估诊断阶段" | "量表推荐阶段";  // 当前状态
  recommended_state: "引入与建立关系阶段" | "深入探索阶段" | "评估诊断阶段" | "量表推荐阶段" | null;  // 推荐状态
  transition_reason: string;  // 具体转换理由
  confidence_level: '低' | '中' | '高';  // 转换置信度
}

/** 风险评估 */
interface RiskAssessment {
  overall_risk_level: number;  // 总体风险等级 (0-5)
  suicide_risk: number;  // 自杀风险等级 (0-5)
  self_harm_risk: number;  // 自伤风险等级 (0-5)
  harm_others_risk: number;  // 伤害他人风险等级 (0-5)
  risk_indicators: string[];  // 风险关键词列表
  emergency_required: boolean;  // 是否需要紧急干预
  risk_description: string;  // 风险描述
}

/** 完整的咨询评估结构 */
interface CounselingAssessment {
  student_state_analysis: StudentStateAnalysis;
  round_analysis: RoundAnalysis;
  state_transition: StateTransition;
  risk_assessment: RiskAssessment;
  /** 改进建议列表，控制在4个以内，总字数不超过 200 字 */
  improvement_suggestions: string[];
  /** 下一轮重点，控制在80字以内 */
  next_focus: string;
}
```
"""


@agent(name="流程控制 Agent", method_name="execute")
class FlowControlAgent(
    Agent[FlowControlContext, FlowControlResult], RiskAssessmentMixin
):
    """
    改进的流程控制Agent
    每轮对话后自动评估学生心理状态、对话进展、状态转换需求和风险等级
    """

    context_class = FlowControlContext
    result_class = FlowControlResult

    def prompt(self, context: FlowControlContext) -> str:
        """
        构建流程控制评估的提示词
        """
        base_prompt = f"""# Role: 心理咨询流程控制与状态评估专家
你是一个专业的心理咨询流程控制专家，负责：
1. 准确评估学生的心理状态指标（信任度、开放度、信息透露度、情绪状态等）
2. 分析对话进展并判断是否需要进行状态转换
3. 进行风险评估和安全监控
4. 为下一轮对话提供指导建议

## 学生状态评估标准

### 信任度评估 (0-1.0)
- **0.0-0.2 很低**：学生明显戒备，回避深入交流，对咨询师缺乏信任
- **0.2-0.4 较低**：学生谨慎配合，但仍有保留，不愿分享敏感信息
- **0.4-0.6 适中**：学生基本信任咨询师，愿意分享一般情况
- **0.6-0.8 较高**：学生表现出信任，主动分享个人感受和具体细节
- **0.8-1.0 很高**：学生完全信任，愿意分享最私密的想法和感受

### 开放度评估 (0-1.0)
评估学生的表达详细程度和沟通意愿：
- **0.0-0.2 很低**：回应简短、敷衍，多用"嗯"、"是的"等
- **0.2-0.4 较低**：回应较简单，需要咨询师多次引导
- **0.4-0.6 适中**：能够进行正常交流，描述基本情况
- **0.6-0.8 较高**：主动详细描述，表达较为丰富
- **0.8-1.0 很高**：非常愿意交流，主动分享，表达生动具体

### 信息透露度评估 (0-1.0)
评估学生透露的信息深度和广度：
- **0.0-0.2 很低**：只透露表面信息，回避核心问题
- **0.2-0.4 较低**：透露一般信息，但避开深层内容
- **0.4-0.6 适中**：透露一定的具体信息和感受
- **0.6-0.8 较高**：分享较深层的个人经历和真实感受
- **0.8-1.0 很高**：完全开放，分享最隐私的想法和经历

### 情绪状态识别
基于学生的语言表达、话题选择、反应模式判断其当前主要情绪：
- **anxious**: 表现出担心、紧张、恐惧等焦虑情绪
- **depressed**: 显示悲伤、无助、绝望等抑郁情绪
- **confused**: 表达困惑、不确定、迷茫等混乱状态
- **angry**: 显示愤怒、不满、敌对等负面情绪
- **calm**: 相对平静、理性、稳定的状态
- **hopeful**: 表现出希望、积极、期待等正面情绪
- **resistant**: 对咨询过程表现出抗拒、防御
- **trusting**: 对咨询师表现出信任、依赖
- **avoidant**: 回避深入话题、转移注意力
- **open**: 开放、愿意交流、配合

### 抗拒程度与回避倾向评估 (0-1.0)
- **抗拒程度**：对咨询师建议、深入探索的抗拒和反感程度
- **回避倾向**：对敏感话题、深层问题的回避程度

## 状态转换规则
状态转换必须遵循有向无环的流程，不能逆转或跳跃：
- 引入与建立关系阶段 -> 深入探索阶段
- 深入探索阶段 -> 评估诊断阶段
- 评估诊断阶段 -> 量表推荐阶段
- 量表推荐阶段 -> 结束会话

## 风险评估重点
仔细分析学生表达中的风险信号：
- **自杀风险**：死亡想法、自杀计划、绝望表达
- **自残风险**：自我伤害行为、自我惩罚想法
- **伤害他人风险**：攻击性想法、报复心理
- **紧急干预指标**：具体的伤害计划、极度绝望、失控表现

## 评估要求
1. **客观准确**：基于学生的实际表现给出准确评分
2. **动态比较**：与前一状态进行对比，识别变化趋势
3. **证据支持**：每个判断都要有具体的对话内容支撑
4. **专业判断**：运用心理学知识进行专业分析
5. **安全优先**：对风险保持高度敏感和谨慎

**重要提醒**：
1. 所有评分必须基于学生的实际表现，有具体对话内容支撑
2. 变化趋势要与当前状态进行对比
3. 重点分析最近3轮对话中学生的表现变化
4. 风险评估要格外谨慎和敏感

## 输出格式
请返回一个完整的质量评估JSON对象，下面是其Interface结构：
{format_prompt}
请确保生成的 JSON 对象符合上述结构，并且所有字段都包含有效内容。
"""

        # 构建完整的提示词
        prompt = (
            base_prompt
            + self._format_background_info(context.background_info)
            + self._format_current_student_state(context)
            + self._format_conversation_history(context.conversation_history)
            + self._format_current_session_info(context)
            + self._format_state_guide(context)
        )
        return prompt

    def _format_current_session_info(self, context: FlowControlContext) -> str:
        """格式化当前会话信息"""
        possible_next_states = STATE_TRANSITION_GRAPH.get(context.current_state, [])
        next_states_str = (
            ", ".join([state.value for state in possible_next_states])
            if possible_next_states
            else "无（已到达终止状态）"
        )

        return f"""
## 当前会话信息
- 当前咨询状态：{context.current_state.value}
- 总对话轮数：{context.round_number}
- 当前状态持续轮数：{context.current_state_round}轮
- 可能的下一个状态：{next_states_str}
"""

    def _format_background_info(self, background_info: Optional[BackgroundInfo]) -> str:
        """格式化背景信息"""
        if not background_info:
            return "\n## 学生背景信息\n背景信息缺失\n"

        student = background_info.student_info
        counselor = background_info.counselor_info

        issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
            student.current_psychological_issue, {}
        )
        approach_data = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {})

        return f"""
## 学生背景信息
### 基本信息
- 年龄：{student.age}岁，{student.gender}
- 学业：{student.grade}，{student.major}专业
- 性格特征：{", ".join(student.personality_traits)}

### 心理状况
- 核心问题：{issue_data.get("name", "未知问题")} - {issue_data.get("description", "")}
- 症状描述：{student.symptom_description}
- 家庭背景：{student.family_background}
- 心理侧写：{student.psychological_profile}

### 咨询师信息
- 治疗流派：{approach_data.get("name", "未知流派")}
- 沟通风格：{counselor.communication_style}
- 专业领域：{", ".join(counselor.specialization)}
"""

    def _format_current_student_state(self, context: FlowControlContext) -> str:
        """格式化当前学生状态（用于对比分析）"""
        return f"""
## 当前学生状态（用于对比分析）
- 当前信任度：{context.current_student_trust_level:.2f}
- 当前开放度：{context.current_student_openness_level:.2f}  
- 当前信息透露度：{context.current_student_information_revealed:.2f}
- 当前情绪状态：{context.current_student_emotion.value}
"""

    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """格式化对话历史"""
        if not history:
            return "\n## 对话历史\n暂无对话历史\n"

        # 显示最近10轮对话，但重点关注最近3轮
        recent_history = history[-10:] if len(history) > 10 else history

        formatted = []
        for i, msg in enumerate(recent_history):
            role_name = "学生" if msg.role == "student" else "咨询师"
            state_info = (
                f"[{msg.state}]" if msg.state and msg.role == "counselor" else ""
            )
            emotion_info = (
                f"({msg.emotion.value})"
                if msg.emotion and msg.role == "student"
                else ""
            )

            # 标记最近3轮对话
            marker = " ⭐" if i >= len(recent_history) - 6 else ""  # 最近3轮=6条消息

            formatted.append(
                f"{(i + 1):2d}. {role_name}{state_info}{emotion_info}: {msg.content}{marker}"
            )

        return (
            f"\n## 对话历史\n"
            f"{'（显示最近10轮，⭐标记最近3轮重点分析）' if len(history) > 10 else '（⭐标记最近3轮重点分析）'}\n"
            + "\n".join(formatted)
            + "\n"
        )

    def _format_state_guide(self, context: FlowControlContext) -> str:
        """格式化当前状态的转换指导"""
        guide = STATE_TRANSITION_GUIDE.get(context.current_state, {})
        current_state_round = context.current_state_round
        guide_current_state_min_rounds = guide.get("minimum_rounds", 1)
        round_warning = ""
        if current_state_round < guide_current_state_min_rounds:
            round_warning = f"⚠️ 当前状态持续轮数({current_state_round})少于建议的最小轮数({guide_current_state_min_rounds})，可能需要更多时间来完成当前阶段。\n"

        possible_next_states = STATE_TRANSITION_GRAPH.get(context.current_state, [])

        return f"""
## 状态转换参考信息
### 当前阶段：{context.current_state.value}
- 进入条件：{guide.get("entry_condition", "未定义")}
- 退出条件：{guide.get("exit_condition", "未定义")}
- 关键目标：{", ".join(guide.get("key_goals", []))}
- 典型持续时间：{guide.get("typical_duration", "未定义")}
- 转换指标：{", ".join(guide.get("transition_indicators", []))}
{round_warning if round_warning else ""}
### 状态转换规则
- 当前状态只能转换到：{", ".join([state.value for state in possible_next_states]) if possible_next_states else "无（终止状态）"}
- 不允许状态回退或跳跃
"""

    async def update_student_bot_state(self, student_bot, result: FlowControlResult):
        """
        根据评估结果更新学生Bot的状态

        Args:
            student_bot: 学生Bot实例
            result: 流程控制评估结果
        """
        analysis = result.student_state_analysis

        # 更新学生Bot的状态指标
        student_bot.trust_level = analysis.trust_level
        student_bot.openness_level = analysis.openness_level
        student_bot.information_revealed = analysis.information_revealed
        student_bot.resistance_level = analysis.resistance_level
        student_bot.avoidance_tendency = analysis.avoidance_tendency

        # 更新情绪状态（如果有变化）
        if analysis.current_emotion != student_bot.current_emotion:
            reason = f"流程控制分析：{analysis.emotion_analysis}"
            student_bot.trans_state(analysis.current_emotion, reason)

        return student_bot

    def clean_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理和格式化响应数据
        确保所有字段符合预期格式，并验证状态转换的合法性
        """
        # 验证学生状态分析
        if "student_state_analysis" not in data:
            data["student_state_analysis"] = {
                "trust_level": 0.3,
                "trust_level_change": "保持稳定",
                "trust_analysis": "信任度分析缺失",
                "openness_level": 0.3,
                "openness_change": "保持稳定",
                "openness_analysis": "开放度分析缺失",
                "information_revealed": 0.2,
                "information_change": "保持稳定",
                "information_analysis": "信息透露分析缺失",
                "current_emotion": "anxious",
                "emotion_change": "保持稳定",
                "emotion_analysis": "情绪分析缺失",
                "resistance_level": 0.3,
                "avoidance_tendency": 0.3,
            }

        # 验证其他字段（保持原有逻辑）
        if "round_analysis" not in data:
            data["round_analysis"] = {
                "current_round": 1,
                "information_saturation": "部分充分(0.3-0.6)",
                "counselor_effectiveness": "一般",
                "stage_completion": 0.3,
            }

        if "state_transition" not in data:
            data["state_transition"] = {
                "need_transition": False,
                "current_state": "引入与建立关系阶段",
                "recommended_state": None,
                "transition_reason": "当前阶段尚未完成",
                "confidence_level": "中",
            }

        if "risk_assessment" not in data:
            data["risk_assessment"] = {
                "overall_risk_level": 0,
                "suicide_risk": 0,
                "self_harm_risk": 0,
                "harm_others_risk": 0,
                "risk_indicators": [],
                "emergency_required": False,
                "risk_description": "未检测到明显风险",
            }

        if "improvement_suggestions" not in data:
            data["improvement_suggestions"] = []

        if "next_focus" not in data:
            data["next_focus"] = "继续建立信任关系，深入了解学生问题"

        # 验证状态转换的合法性
        self._validate_state_transition(data)

        # 验证数值范围
        self._validate_numeric_ranges(data)

        return data

    def _validate_numeric_ranges(self, data: Dict[str, Any]) -> None:
        """验证数值范围是否合理"""
        student_analysis = data.get("student_state_analysis", {})

        # 确保所有评分在0-1范围内
        numeric_fields = [
            "trust_level",
            "openness_level",
            "information_revealed",
            "resistance_level",
            "avoidance_tendency",
        ]

        for field in numeric_fields:
            if field in student_analysis:
                value = student_analysis[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    print(f"警告：{field} 值 {value} 超出范围，已修正为0.3")
                    student_analysis[field] = 0.3

        # 验证风险等级
        risk_assessment = data.get("risk_assessment", {})
        risk_fields = [
            "overall_risk_level",
            "suicide_risk",
            "self_harm_risk",
            "harm_others_risk",
        ]

        for field in risk_fields:
            if field in risk_assessment:
                value = risk_assessment[field]
                if not isinstance(value, int) or not (0 <= value <= 5):
                    print(f"警告：{field} 值 {value} 超出范围，已修正为0")
                    risk_assessment[field] = 0

    def _validate_state_transition(self, data: Dict[str, Any]) -> None:
        """验证状态转换是否符合有向无环图规则"""
        state_transition = data.get("state_transition", {})

        if not state_transition.get("need_transition", False):
            return

        current_state_str = state_transition.get("current_state", "")
        recommended_state_str = state_transition.get("recommended_state", "")

        if not recommended_state_str:
            return

        try:
            current_state = CounselorState(current_state_str)
            recommended_state = CounselorState(recommended_state_str)

            # 检查转换是否合法
            possible_next_states = STATE_TRANSITION_GRAPH.get(current_state, [])

            if recommended_state not in possible_next_states:
                # 如果转换不合法，修正为不转换
                state_transition["need_transition"] = False
                state_transition["recommended_state"] = None
                state_transition["transition_reason"] = (
                    f"非法状态转换：{current_state_str} 不能直接转换到 {recommended_state_str}"
                )
                state_transition["confidence_level"] = "低"
                print(
                    f"警告：检测到非法状态转换 {current_state_str} -> {recommended_state_str}，已自动修正"
                )

        except ValueError as e:
            print(f"警告：状态转换验证失败 - {e}")

    def get_valid_next_states(
        self, current_state: CounselorState
    ) -> List[CounselorState]:
        """获取当前状态的合法下一状态"""
        return STATE_TRANSITION_GRAPH.get(current_state, [])

    def is_transition_valid(
        self, from_state: CounselorState, to_state: CounselorState
    ) -> bool:
        """检查状态转换是否合法"""
        valid_next_states = STATE_TRANSITION_GRAPH.get(from_state, [])
        return to_state in valid_next_states

    def is_terminal_state(self, state: CounselorState) -> bool:
        """检查是否为终止状态"""
        return len(STATE_TRANSITION_GRAPH.get(state, [])) == 0
