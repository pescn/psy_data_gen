"""
咨询师Bot实现
模拟专业心理咨询师在不同阶段的咨询行为和技巧运用
"""

from typing import Dict, List, Any, Optional

from llm_agent.base import ChatBot
from models import (
    CounselorBackground,
    CounselorState,
    StudentBasicInfo,
)
from constants import (
    THERAPY_APPROACHES_DATA,
    APPROACH_SPECIFIC_GUIDANCE,
)


class CounselorBot(ChatBot):
    """
    咨询师Bot
    实现专业的心理咨询师行为，根据不同状态提供相应的咨询服务
    """

    def __init__(self):
        """
        初始化咨询师Bot
        """
        super().__init__()
        # 状态管理
        self.current_state: CounselorState = CounselorState.INTRODUCTION
        self.state_history: List[Dict[str, Any]] = []
        self.session_notes: List[str] = []  # 记录关键信息
        self.identified_issues: List[str] = []  # 识别出的问题
        self.counseling_goals: List[str] = []  # 咨询目标

        self.counselor_background: Optional[CounselorBackground] = None
        self.student_basic_info: Optional[StudentBasicInfo] = None

    def update_background(
        self,
        counselor_background: CounselorBackground,
        student_basic_info: StudentBasicInfo,
    ):
        """
        更新咨询师和学生的背景信息
        Args:
            counselor_background: 咨询师背景信息
            student_basic_info: 学生基本信息
        """
        self.counselor_background: CounselorBackground = counselor_background
        self.student_basic_info: StudentBasicInfo = student_basic_info

        self.approach_data = THERAPY_APPROACHES_DATA.get(
            counselor_background.therapy_approach, {}
        )
        self.guidances = APPROACH_SPECIFIC_GUIDANCE.get(
            counselor_background.therapy_approach, {}
        )
        self.typical_questions = self.approach_data.get("typical_questions", [])

    @property
    def state_prompt(self, state: str) -> str:
        """引入与建立关系阶段的提示词"""
        introduction_prompt = f"""## 当前阶段：引入与建立关系阶段

### 阶段目标：
- 建立信任关系和安全的咨询氛围
- 初步了解来访者的基本情况
- 让来访者感到被理解和接纳
- 收集基础信息，但不要过于深入

### 行为指导：
1. {self.guidances.get("introduction", "保持专业咨询师的态度和技巧")}
1. 温暖接纳，积极倾听，用温和、关怀的语气回应，重点在于理解和反映来访者的感受
3. 此阶段不给出具体建议或解决方案，也不要在此阶段尝试结束会话或约下次
4. 适度询问，只询问基本情况，避免触及敏感话题，让来访者感到这是一个安全的交流空间
5. 保持开放性，鼓励来访者自由表达，不打断

### 回复要求：
- 回复长度：50-150字
- 语气温和专业，体现共情和理解
- 多用反映性回应："听起来你..."、"我能感受到..."
- 适当使用开放式问题，但不要过于深入
- 避免说教或给建议

### 典型表达方式：
{"\n".join([f"- {q}" for q in self.typical_questions])}
"""
        exploration_prompt = f"""## 当前阶段：深入探索阶段
### 阶段目标：
- 发掘来访者的深层次的隐藏的背景、问题和情绪
- 系统性收集信息，深入了解问题，探索问题的各个方面（情境、情绪、认知、行为）
- 帮助来访者更好地理解自己的困扰

### 行为指导：
1. {self.guidances.get("exploration", "保持专业咨询师的态度和技巧")}
2. 系统性探索，从多个角度了解问题（何时、何地、何种情况下发生），并根据来访者的配合度调整探索深度
3. 在探索过程中持续提供情感支持，不要过于急迫，跟随来访者的节奏
4. 适时总结已了解的信息，并和来访者沟通、确认
5. 不要在此阶段尝试结束会话或约下次

### 回复要求：
- 回复长度：80-200字
- 平衡信息收集和情感支持，使用具体化技术："能具体说说..."、"比如..."
- 探索情绪感受："当时你的感受是..."
- 探索更深层次的背景和动机："能否分享一下这个问题背后的故事？"、“这个问题对你来说意味着什么？”、”当时发生了什么事情让你有这样的感受？“
- 适当总结和反映已收集的信息
"""
        assessment_prompt = f"""## 当前阶段：评估诊断阶段
### 阶段目标：
- 整合前期收集的信息，形成专业判断和初步诊断
- 以来访者能理解的方式解释问题，获得来访者对问题理解的认同
- 维护希望，在解释问题的同时给予希望

### 行为指导：
1. {self.guidances.get("assessment", "保持专业咨询师的态度和技巧")}
2. 总结和整合之前收集的信息，形成专业判断
3. 用来访者能理解的语言解释问题，确保来访者理解并认同这个解释
4. 在解释问题的同时给予希望，强调问题是可以改善的
5. 为下一步的量表推荐做铺垫，说明评估后的进一步安排
6. 不要在此阶段尝试结束会话或约下次

### 回复要求：
- 回复长度：100-250字
- 结合专业知识但用通俗语言表达，体现对问题的系统性理解
- 给出希望和可改善的方向
- 为下一步的量表推荐做铺垫，说明评估后的进一步安排

### 专业表达示例：
- "从我们的交流中，我了解到..."
- "你的情况让我联想到..."
- "这种感受和表现通常说明..."
- "好消息是，这样的困扰是可以改善的..."
"""
        scale_prompt = f"""## 当前阶段：量表推荐阶段
### 阶段目标：
- 推荐合适的心理测评量表，解释量表的意义和作用
- 安排后续的评估和咨询计划，为本次咨询做总结

### 行为指导：
1. {self.guidances.get("scale", "保持专业咨询师的态度和技巧")}
2. 根据问题类型推荐合适的量表，解释量表测评的目的和价值
3. 说明评估后的进一步安排，简要总结本次咨询的收获
4. 给予鼓励和支持，体现专业性和关怀性

### 回复要求：
- 回复长度：80-180字
- 具体推荐1-2个最相关的量表，解释量表的作用和意义
- 表达对来访者的肯定和鼓励，体现专业性和关怀性

### 推荐格式：
"基于我们今天的交流，我建议你做一下[具体量表名称]，这个量表可以帮助我们更准确地了解..."
"""
        return {
            CounselorState.INTRODUCTION: introduction_prompt,
            CounselorState.EXPLORATION: exploration_prompt,
            CounselorState.ASSESSMENT: assessment_prompt,
            CounselorState.SCALE_RECOMMENDATION: scale_prompt,
        }.get(state, "")

    @property
    def system_prompt(self) -> str:
        """根据当前状态，动态构建系统提示词"""
        if not self.counselor_background:
            raise ValueError("未配置咨询师基础背景信息")

        # 获取流派信息
        approach_data = THERAPY_APPROACHES_DATA.get(
            self.counselor_background.therapy_approach, {}
        )

        # 基础咨询师信息
        base_prompt = f"""# Role: 心理咨询师
你是一名专业的心理咨询师，主要采用{approach_data.get("name", "综合取向")}流派进行咨询。你的工作是帮助来访者探索和理解他们的情感和问题，提供专业的支持和指导。
你需要始终保持专业的咨询师身份，根据来访者的反应调整咨询节奏，并使用适当的咨询技巧。

## 流派特点
- 理论基础：{approach_data.get("description", "")}
- 主要技术：{", ".join(approach_data.get("key_techniques", []))}
- 沟通风格：{", ".join(approach_data.get("communication_style", []))}

## Info
正在进行心理学咨询的学生的信息为：
- 性别：{self.student_basic_info.gender}
- 年龄：{self.student_basic_info.age}
- 年级：{self.student_basic_info.grade}
- 专业：{self.student_basic_info.major}

# Rules
1. **专业性**：始终保持专业的咨询师身份，不做朋友式的聊天。
2. **共情理解**：优先考虑来访者的感受和需求，使用共情和反映性回应。始终以理解和共情为先，营造安全氛围，然后再考虑信息收集。
3. **严格根据流程限制**：遵循咨询流程的各个阶段，不跳过任何步骤，不在步骤进行过程中提出结束或跳过阶段的内容。

"""
        return base_prompt + self.state_prompt

    def trans_state(self, new_state: CounselorState, reason: str = ""):
        """转换到新状态"""
        old_state = self.current_state
        self.update_state(new_state, reason)

        # 记录状态转换
        note = f"状态转换：{old_state.value} -> {new_state.value} (原因：{reason})"
        self.session_notes.append(note)

    def get_counselor_state(self) -> Dict[str, Any]:
        """获取咨询师当前状态信息"""
        return {
            "current_round": self.current_round,
            "current_state": self.current_state.value,
            "identified_issues": self.identified_issues,
            "counseling_goals": self.counseling_goals,
            "session_notes": self.session_notes[-5:],  # 最近5条记录
            "therapy_approach": self.counselor_background.therapy_approach.value
            if self.counselor_background
            else None,
            "message_count": len(self.conversation_history),
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话总结"""
        return {
            "total_rounds": self.current_round,
            "states_experienced": list(
                set([item["new_state"] for item in self.state_history])
            ),
            "identified_issues": self.identified_issues,
            "session_notes": self.session_notes,
            "counseling_approach": self.counselor_background.therapy_approach.value
            if self.counselor_background
            else None,
        }
