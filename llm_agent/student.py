"""
学生Bot实现
模拟真实学生在心理咨询中的表现，包括情绪变化、信息透露等自然行为
"""

import random
from typing import Dict, List, Any, Optional

from llm_agent.base import ChatBot, RiskAssessmentMixin, convert_history_for_student
from models import (
    PsychologicalIssue,
    StudentBackground,
    EmotionState,
    ConversationMessage,
)


class StudentBot(ChatBot, RiskAssessmentMixin):
    """
    学生Bot
    模拟学生在心理咨询中的真实表现
    """

    def __init__(self):
        """
        初始化学生Bot
        """
        super().__init__()
        # 情绪状态管理
        self.current_emotion: EmotionState = EmotionState.ANXIOUS
        self.emotion_history: List[Dict[str, Any]] = []
        self.session_notes: List[str] = []  # 记录关键信息

        # 行为特征配置
        self.trust_level = 0.1  # 初始信任度很低
        self.openness_level = 0.2  # 初始开放度很低
        self.information_revealed = 0.1  # 已透露的信息比例

        # 个性化参数
        self.avoidance_tendency = random.uniform(0.3, 0.7)  # 回避倾向
        self.resistance_level = random.uniform(0.2, 0.6)  # 抗拒程度
        self.chattiness = random.uniform(0.4, 0.8)  # 健谈程度

        # 学生背景信息（待设置）
        self.student_background: Optional[StudentBackground] = None

    def update_background(self, student_background: StudentBackground):
        """
        更新学生背景信息

        Args:
            student_background: 学生背景信息
        """
        self.student_background = student_background
        self.current_emotion = self._determine_initial_emotion()

        # 根据背景调整个性化参数
        self._adjust_personality_parameters()

    def _determine_initial_emotion(self) -> EmotionState:
        """根据心理问题确定初始情绪状态"""
        if not self.student_background:
            return EmotionState.ANXIOUS

        emotion_mapping = {
            PsychologicalIssue.ACADEMIC_ANXIETY: EmotionState.ANXIOUS,
            PsychologicalIssue.SOCIAL_PHOBIA: EmotionState.ANXIOUS,
            PsychologicalIssue.DEPRESSION: EmotionState.DEPRESSED,
            PsychologicalIssue.PROCRASTINATION: EmotionState.CONFUSED,
            PsychologicalIssue.OCD_SYMPTOMS: EmotionState.ANXIOUS,
            PsychologicalIssue.ADAPTATION_ISSUES: EmotionState.CONFUSED,
            PsychologicalIssue.RELATIONSHIP_ISSUES: EmotionState.DEPRESSED,
            PsychologicalIssue.FAMILY_CONFLICTS: EmotionState.ANGRY,
            PsychologicalIssue.IDENTITY_CONFUSION: EmotionState.CONFUSED,
            PsychologicalIssue.SLEEP_PROBLEMS: EmotionState.ANXIOUS,
        }

        return emotion_mapping.get(
            self.student_background.current_psychological_issue, EmotionState.ANXIOUS
        )

    def _adjust_personality_parameters(self):
        """根据性格特征调整个性化参数"""
        if not self.student_background:
            return

        traits = self.student_background.personality_traits

        for trait in traits:
            if "内向" in trait:
                self.openness_level *= 0.8
                self.chattiness *= 0.7
            elif "外向" in trait:
                self.openness_level *= 1.2
                self.chattiness *= 1.3
            elif "敏感" in trait:
                self.resistance_level *= 1.3
                self.avoidance_tendency *= 1.2
            elif "完美主义" in trait:
                self.avoidance_tendency *= 1.2
                self.resistance_level *= 1.1

    def convert_history_to_messages(
        self, conversation_history: List[ConversationMessage]
    ) -> List[Dict[str, str]]:
        """
        将对话历史转换为学生Bot的视角
        学生Bot视角：assistant=学生, user=咨询师
        """
        return convert_history_for_student(conversation_history)

    @property
    def system_prompt(self) -> str:
        """根据当前状态动态构建系统提示词"""
        if not self.student_background:
            raise ValueError("未配置学生背景信息")

        base_prompt = f"""# Role: 心理咨询来访者（大学生）
你是一名正在接受心理咨询的大学生，你需要根据自己的背景和心理问题，真实地表达自己的感受和困扰。

## Rules
1. **渐进式信息透露**：根据信任度逐步透露更多信息，不要一次性说出所有问题
2. **真实的学生语言**：使用口语化、非专业的表达方式，符合大学生身份
3. **情绪真实性**：严格根据当前情绪状态调整语气和表达方式
4. **适度回避**：对敏感问题可能会回避、转移话题或说"不知道"
5. **质疑与试探**：当咨询师触及核心问题时，可能会质疑咨询师的意图
6. **自然对话**：保持大学生的说话习惯，可以适当跑题但要有逻辑
7. **保持角色一致性**：体现当前的信任度和情绪状态
8. **回复长度控制**：50-200字之间，根据开放度调整详细程度

## 回复要求
- 仅返回语句内容，不要有行为或表情的描述
- 可以适当通过语句内容表现出犹豫、不确定、回避等真实反应
- 使用符合当前情绪状态的语气和词汇

# 学生背景信息

## 基本信息
- 姓名：{self.student_background.name}
- 年龄：{self.student_background.age}岁
- 性别：{self.student_background.gender}
- 年级：{self.student_background.grade}
- 专业：{self.student_background.major}

## 家庭背景
{self.student_background.family_background}

## 性格特征
{", ".join(self.student_background.personality_traits)}

## 心理侧写
{self.student_background.psychological_profile}

## 当前困扰
{self.student_background.symptom_description}

## 深层信息（只有在高度信任后才会透露）
{self.student_background.hidden_personal_info}


# 当前状态
- 对话轮数：{self.current_round}
- 情绪状态：{self.current_emotion.value}
- 信任度：{self.trust_level:.1f}/1.0
- 开放度：{self.openness_level:.1f}/1.0
- 信息透露度：{self.information_revealed:.1f}/1.0

## 情绪状态指导
{self._get_emotion_guidance()}

## 行为调整建议
{self._get_behavior_guidance()}
"""
        return base_prompt

    def _get_emotion_guidance(self) -> str:
        """获取当前情绪的行为指导"""
        emotion_guides = {
            EmotionState.ANXIOUS: "表现出紧张、担心，语速可能较快，容易转移话题，用词谨慎",
            EmotionState.DEPRESSED: "语调低沉，回应较少，可能表达无助感，缺乏动力",
            EmotionState.CONFUSED: "表现出困惑、不确定，经常说'不知道'、'可能'、'也许'",
            EmotionState.ANGRY: "语气可能较冲，容易情绪化，可能对建议有抗拒",
            EmotionState.CALM: "相对平静，能够理性交流，语气平和",
            EmotionState.HOPEFUL: "积极一些，愿意尝试建议，对未来有期待",
            EmotionState.RESISTANT: "对咨询师的话有质疑，可能不太配合，表现出防御",
            EmotionState.TRUSTING: "更愿意分享，语气较为放松，主动提供信息",
            EmotionState.AVOIDANT: "回避深入话题，可能转移话题，不愿深入",
            EmotionState.OPEN: "比较愿意交流，会分享更多细节，表达较为直接",
            EmotionState.OTHER: "根据具体情况灵活表现",
        }
        return emotion_guides.get(self.current_emotion, "保持自然的情绪表达")

    def _get_behavior_guidance(self) -> str:
        """获取行为指导建议"""
        guidance = []

        if self.trust_level < 0.3:
            guidance.append("信任度较低：保持谨慎，只分享表面信息，可能有戒备心理")
        elif self.trust_level < 0.6:
            guidance.append("信任度适中：可以分享一些具体细节，但避免最敏感的内容")
        else:
            guidance.append("信任度较高：可以分享深层信息和真实感受")

        if self.openness_level < 0.4:
            guidance.append("开放度较低：回应较简短，可能需要咨询师多次引导")
        else:
            guidance.append("开放度较高：愿意详细描述情况和感受")

        if self.avoidance_tendency > 0.6:
            guidance.append("回避倾向强：对敏感话题容易转移话题或回避")

        if self.resistance_level > 0.5:
            guidance.append("抗拒程度高：对咨询师的建议可能表现出质疑或不配合")

        return "；".join(guidance) if guidance else "保持自然的交流状态"

    def should_reveal_information(self, information_level: str) -> bool:
        """
        判断是否应该透露某个级别的信息

        Args:
            information_level: "surface", "moderate", "deep"

        Returns:
            bool: 是否应该透露
        """
        thresholds = {"surface": 0.1, "moderate": 0.4, "deep": 0.7}
        threshold = thresholds.get(information_level, 0.5)

        return (
            self.trust_level >= threshold
            and random.random() < (self.trust_level + self.openness_level) / 2
        )

    def trans_state(self, new_emotion: EmotionState, reason: str = ""):
        """转换到新状态"""
        current_emotion = self.current_emotion
        self.update_state(new_emotion, reason)

        # 记录状态转换
        note = (
            f"状态转换：{current_emotion.value} -> {new_emotion.value} (原因：{reason})"
        )
        self.session_notes.append(note)

    def get_student_state(self) -> Dict[str, Any]:
        """获取学生当前状态信息"""
        return {
            "current_round": self.current_round,
            "current_emotion": self.current_emotion.value,
            "trust_level": self.trust_level,
            "openness_level": self.openness_level,
            "information_revealed": self.information_revealed,
            "avoidance_tendency": self.avoidance_tendency,
            "resistance_level": self.resistance_level,
            "psychological_issue": self.student_background.current_psychological_issue.value
            if self.student_background
            else None,
            "message_count": len(self.conversation_history)
            if hasattr(self, "conversation_history")
            else 0,
        }

    def get_emotion_summary(self) -> Dict[str, Any]:
        """获取情绪变化总结"""
        return {
            "current_emotion": self.current_emotion.value,
            "emotion_transitions": len(self.emotion_history),
            "emotion_history": self.emotion_history[-5:],  # 最近5次情绪变化
            "trust_evolution": self.trust_level,
            "openness_evolution": self.openness_level,
        }
