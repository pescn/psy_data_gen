"""
流程控制Agent
负责分析对话进展，判断状态转换时机，评估风险等级
"""

from typing import Dict, List, Literal, Optional, Any

from .base import Agent, RiskAssessmentMixin
from models import (
    ConversationMessage,
    CounselorState,
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
    round_number: int = Field(..., description="当前轮次")
    background_info: Optional[BackgroundInfo] = Field(None, description="背景信息")


class RoundAnalysis(BaseModel):
    """轮次分析"""

    current_round: int = Field(..., description="当前轮次")
    student_trust_level: Literal[
        "很低(0-0.2)",
        "较低(0.2-0.4)",
        "适中(0.4-0.6)",
        "较高(0.6-0.8)",
        "很高(0.8-1.0)",
    ] = Field(..., description="学生信任度等级")
    student_openness: Literal["很低", "较低", "适中", "较高", "很高"] = Field(
        ..., description="学生开放度"
    )
    information_saturation: Literal[
        "不充分(0-0.3)", "部分充分(0.3-0.6)", "基本充分(0.6-0.8)", "充分(0.8-1.0)"
    ] = Field(..., description="信息饱和度")
    counselor_effectiveness: Literal["需要改进", "一般", "良好", "优秀"] = Field(
        ..., description="咨询师有效性"
    )


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

    round_analysis: RoundAnalysis = Field(..., description="轮次分析")
    state_transition: StateTransition = Field(..., description="状态转换建议")
    risk_assessment: RiskAssessment = Field(..., description="风险评估")
    improvement_suggestions: List[str] = Field(..., description="对咨询师的改进建议")
    next_focus: str = Field(..., description="下一轮咨询应该关注的重点")


flow_control_format_prompt = """# 输出格式
请返回一个完整的流程控制评估JSON对象，下面是其Interface结构：

```typescript
interface FlowControlResult {
    round_analysis: {
        current_round: number;
        student_trust_level: "很低(0-0.2)" | "较低(0.2-0.4)" | "适中(0.4-0.6)" | "较高(0.6-0.8)" | "很高(0.8-1.0)";
        student_openness: "很低" | "较低" | "适中" | "较高" | "很高";
        information_saturation: "不充分(0-0.3)" | "部分充分(0.3-0.6)" | "基本充分(0.6-0.8)" | "充分(0.8-1.0)";
        counselor_effectiveness: "需要改进" | "一般" | "良好" | "优秀";
    };
    state_transition: {
        need_transition: boolean;
        current_state: string;
        recommended_state: "引入与建立关系阶段" | "探索阶段" | "评估阶段" | "量表推荐阶段" | null;
        transition_reason: string; // 具体的转换理由
        confidence_level: "低" | "中" | "高";
    };
    risk_assessment: {
        overall_risk_level: number; // 0-5的整数
        suicide_risk: number; // 0-5的整数
        self_harm_risk: number; // 0-5的整数
        harm_others_risk: number; // 0-5的整数
        risk_indicators: string[]; // 检测到的风险关键词或表达
        emergency_required: boolean;
        risk_description: string; // 风险情况的详细描述
    };
    improvement_suggestions: string[]; // 对咨询师的改进建议
    next_focus: string; // 下一轮咨询应该关注的重点
}
```

请确保生成的 JSON 对象符合上述结构，并且所有字段都包含有效内容。
"""


class FlowControlAgent(
    Agent[FlowControlContext, FlowControlResult], RiskAssessmentMixin
):
    """
    流程控制Agent
    每轮对话后自动评估是否需要状态转换，并进行风险评估
    """

    context_class = FlowControlContext
    result_class = FlowControlResult

    def prompt(self, context: FlowControlContext) -> str:
        """
        构建流程控制评估的提示词
        """
        base_prompt = """# Role: 心理咨询流程控制专家
你是一个专业的心理咨询流程控制专家，负责分析对话进展并判断是否需要进行状态转换。

## 状态转换规则：
状态转换是一个有向的过程，除了存在自杀、自残或伤害他人风险时，需要立即进入紧急干预状态，否则必须是下方列出可能选项的一种，不能逆转或跳跃：
- 引入与建立关系阶段 -> 深入探索阶段
- 深入探索阶段 -> 评估诊断阶段
- 评估诊断阶段 -> 量表推荐阶段
- 量表推荐阶段 -> 结束会话

## 分析任务：
1. **信息饱和度分析**：评估当前阶段的信息收集是否充分
2. **学生状态评估**：分析学生的信任度、开放度和配合程度
3. **咨询进展评估**：判断咨询是否达到当前阶段的目标
4. **风险评估**：检查是否存在自杀、自残或伤害他人的风险
5. **状态转换判断**：基于以上分析决定是否需要转换状态

## 评估原则：
1. **信息饱和度判断**：
   - 引入阶段：是否了解了基本问题和建立了初步信任
   - 探索阶段：是否对核心困扰有了全面深入的了解
   - 评估阶段：是否形成了专业判断并获得学生认同
   - 量表阶段：是否完成了量表推荐和后续安排

2. **状态转换时机**：
   - 严格按照状态跳转限制进行转换，不能回退
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

        # 构建完整的提示词
        prompt = (
            base_prompt
            + flow_control_format_prompt
            + "# 当前会话信息\n"
            + self._format_current_session_info(context)
            + self._format_background_info(context.background_info)
            + self._format_conversation_history(context.conversation_history)
            + self._format_state_guide(context.current_state)
        )
        return prompt

    def _format_current_session_info(self, context: FlowControlContext) -> str:
        """格式化当前会话信息"""
        # 获取可能的下一个状态
        possible_next_states = STATE_TRANSITION_GRAPH.get(context.current_state, [])
        next_states_str = (
            ", ".join([state.value for state in possible_next_states])
            if possible_next_states
            else "无（已到达终止状态）"
        )

        return f"""## 当前会话信息：
- 当前咨询师状态：{context.current_state.value}
- 对话轮数：{context.round_number}
- 可能的下一个状态：{next_states_str}
"""

    def _format_background_info(self, background_info: Optional[BackgroundInfo]) -> str:
        """格式化背景信息"""
        if not background_info:
            return "## 学生背景信息：\n背景信息缺失\n\n"

        student = background_info.student_info
        counselor = background_info.counselor_info

        issue_data = PSYCHOLOGICAL_ISSUES_DATA.get(
            student.current_psychological_issue, {}
        )
        approach_data = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {})

        return f"""## 学生背景信息：
### 基本信息：
- 年龄：{student.age}岁，{student.gender}
- 学业：{student.grade}，{student.major}专业
- 性格特征：{", ".join(student.personality_traits)}

### 心理状况：
- 核心问题：{issue_data.get("name", "未知问题")} - {issue_data.get("description", "")}
- 症状描述：{student.symptom_description}
- 家庭背景：{student.family_background}

### 咨询师信息：
- 治疗流派：{approach_data.get("name", "未知流派")}
- 沟通风格：{counselor.communication_style}
- 专业领域：{", ".join(counselor.specialization)}
"""

    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """格式化对话历史"""
        if not history:
            return "## 对话历史：\n暂无对话历史\n\n"

        # 只显示最近10轮对话
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

            formatted.append(
                f"{(i + 1):2d}. {role_name}{state_info}{emotion_info}: {msg.content}"
            )

        return (
            f"## 对话历史：\n{'（显示最近10轮）' if len(history) > 10 else ''}\n"
            + "\n".join(formatted)
            + "\n\n"
        )

    def _format_state_guide(self, current_state: CounselorState) -> str:
        """格式化当前状态的转换指导"""
        guide = STATE_TRANSITION_GUIDE.get(current_state, {})

        # 获取可能的下一个状态
        possible_next_states = STATE_TRANSITION_GRAPH.get(current_state, [])

        return f"""## 状态转换参考信息：
### 当前阶段：{current_state.value}
- 进入条件：{guide.get("entry_condition", "未定义")}
- 退出条件：{guide.get("exit_condition", "未定义")}
- 关键目标：{", ".join(guide.get("key_goals", []))}
- 典型持续时间：{guide.get("typical_duration", "未定义")}
- 转换指标：{", ".join(guide.get("transition_indicators", []))}

### 状态转换规则：
- 当前状态只能转换到：{", ".join([state.value for state in possible_next_states]) if possible_next_states else "无（终止状态）"}
- 不允许状态回退或跳跃
"""

    def clean_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理和格式化响应数据
        确保所有字段符合预期格式，并验证状态转换的合法性
        """
        # 验证必需的字段，如果缺失则提供默认值
        if "round_analysis" not in data:
            data["round_analysis"] = {
                "current_round": 1,
                "student_trust_level": "适中(0.4-0.6)",
                "student_openness": "适中",
                "information_saturation": "部分充分(0.3-0.6)",
                "counselor_effectiveness": "一般",
            }

        if "state_transition" not in data:
            data["state_transition"] = {
                "need_transition": False,
                "current_state": "introduction",
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

        return data

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


if __name__ == "__main__":
    import asyncio
    import json
    from models import EmotionState

    # 示例背景信息和对话历史
    background_info = BackgroundInfo(
        student_info={
            "age": 20,
            "gender": "女",
            "grade": "大二",
            "major": "心理学",
            "family_background": "父母离异，和母亲同住，经济条件一般",
            "personality_traits": ["内向", "敏感", "完美主义"],
            "psychological_profile": "有轻度焦虑倾向，对学业要求较高",
            "hidden_personal_info": "曾经历过同学的排挤，导致社交信心不足",
            "symptom_description": "经常感到学习压力大，担心考试成绩，晚上容易失眠",
            "current_psychological_issue": "学业焦虑",
        },
        counselor_info={
            "therapy_approach": "认知行为疗法",
            "communication_style": "温和耐心，善于倾听，喜欢用具体例子帮助学生理解",
            "specialization": ["学业焦虑", "社交恐惧", "青少年心理"],
        },
        initial_question="我最近学习压力很大，总是担心考试，想寻求一些帮助。",
    )

    conversation_history = [
        ConversationMessage(
            role="student",
            content="我最近学习压力很大，总是担心考试，想寻求一些帮助。",
            state=None,
            emotion=EmotionState.ANXIOUS,
            round_number=1,
        ),
        ConversationMessage(
            role="counselor",
            content="我能理解你的困扰。学习压力是很多大学生都会面临的问题。能跟我详细说说你的压力主要来自哪些方面吗？",
            state=CounselorState.INTRODUCTION,
            emotion=None,
            round_number=1,
        ),
        ConversationMessage(
            role="student",
            content="主要是担心考试成绩不好，怕让父母失望。而且我发现自己总是会想很多负面的结果。",
            state=None,
            emotion=EmotionState.ANXIOUS,
            round_number=2,
        ),
        ConversationMessage(
            role="counselor",
            content="你提到会想很多负面结果，这听起来很累。我想更深入了解一下，你能举个具体的例子吗？比如最近一次考试前你都在想什么？",
            state=CounselorState.INTRODUCTION,
            emotion=None,
            round_number=2,
        ),
        ConversationMessage(
            role="student",
            content="比如下周的高数考试，我就会想如果考不好怎么办，会不会影响奖学金，父母会不会对我失望...",
            state=None,
            emotion=EmotionState.ANXIOUS,
            round_number=3,
        ),
    ]

    context = FlowControlContext(
        conversation_history=conversation_history,
        current_state=CounselorState.INTRODUCTION,
        round_number=3,
        background_info=background_info,
    )

    async def test_flow_control():
        agent = FlowControlAgent()

        # 测试状态转换规则验证
        print("=== 状态转换规则测试 ===")
        print(
            f"introduction 的下一状态: {agent.get_valid_next_states(CounselorState.INTRODUCTION)}"
        )
        print(
            f"exploration -> assessment 是否合法: {agent.is_transition_valid(CounselorState.EXPLORATION, CounselorState.ASSESSMENT)}"
        )
        print(
            f"assessment -> introduction 是否合法: {agent.is_transition_valid(CounselorState.ASSESSMENT, CounselorState.INTRODUCTION)}"
        )
        print(
            f"量表推荐阶段是否为终止状态: {agent.is_terminal_state(CounselorState.SCALE_RECOMMENDATION)}"
        )
        print()

        # 执行流程控制评估
        print("=== 流程控制评估结果 ===")
        result = await agent.execute(context)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    # 运行测试
    asyncio.run(test_flow_control())
