"""
学生Bot实现
模拟真实学生在心理咨询中的表现，包括情绪变化、信息透露等自然行为
"""

import random
from typing import Dict, List, Any

from llm_agent.base import ChatBot, RiskAssessmentMixin
from models import (
    PsychologicalIssue,
    StudentBackground,
    EmotionState,
    ConversationMessage,
)
from constants import EMOTION_TRANSITIONS


class StudentBot(ChatBot, RiskAssessmentMixin):
    """
    学生Bot
    模拟学生在心理咨询中的真实表现
    """

    def _init_config(
        self,
        student_background: StudentBackground,
    ):
        """初始化学生Bot配置"""
        # 学生背景信息
        self.student_background: StudentBackground = student_background

        # 初始化情绪状态
        self.current_emotion = self._determine_initial_emotion()

        # 行为特征配置
        self.trust_level = 0.1  # 初始信任度很低
        self.openness_level = 0.2  # 初始开放度很低
        self.information_revealed = 0.1  # 已透露的信息比例

        # 个性化参数
        self.avoidance_tendency = random.uniform(0.3, 0.7)  # 回避倾向
        self.resistance_level = random.uniform(0.2, 0.6)  # 抗拒程度
        self.chattiness = random.uniform(0.4, 0.8)  # 健谈程度

        # 构建系统提示词模板
        self._build_system_prompts()

    def _determine_initial_emotion(self) -> EmotionState:
        """根据心理问题确定初始情绪状态"""
        if not self.student_background:
            return EmotionState.ANXIOUS
        return {
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
        }[self.student_background.current_psychological_issue]

    def _build_system_prompts(self):
        """构建系统提示词模板"""
        if not self.student_background:
            return

        base_prompt = f"""你是一名正在接受心理咨询的大学生，具有以下背景信息：

基本信息：
- 姓名：{self.student_background.name}
- 年龄：{self.student_background.age}岁
- 性别：{self.student_background.gender}
- 年级：{self.student_background.grade}
- 专业：{self.student_background.major}

家庭背景：
{self.student_background.family_background}

性格特征：
{", ".join(self.student_background.personality_traits)}

心理侧写：
{self.student_background.psychological_profile}

当前困扰：
{self.student_background.symptom_description}

深层信息（只有在高度信任后才会透露）：
{self.student_background.hidden_personal_info}

重要行为指导原则：
1. **渐进式信息透露**：根据信任度逐步透露更多信息，不要一次性说出所有问题
2. **真实的学生语言**：使用口语化、非专业的表达方式
3. **情绪真实性**：根据当前情绪状态调整语气和表达
4. **适度回避**：对敏感问题可能会回避、转移话题或说"不知道"
5. **质疑与试探**：当咨询师触及核心问题时，可能会质疑咨询师的意图
6. **自然对话**：保持大学生的说话习惯，可以适当跑题但要有逻辑

当前状态：
- 信任度：{self.trust_level:.1f}/1.0
- 开放度：{self.openness_level:.1f}/1.0
- 情绪状态：{self.current_emotion.value}
"""

        self.system_prompts["base"] = base_prompt

    def _build_system_prompt(self, **context) -> str:
        """构建当前的系统提示词"""
        base_prompt = self.system_prompts.get("base", "")

        # 添加当前状态信息
        current_state_info = f"""

当前对话状态：
- 对话轮数：{self.current_round}
- 当前情绪：{self.current_emotion.value}
- 信任度：{self.trust_level:.1f}/1.0（影响信息透露程度）
- 开放度：{self.openness_level:.1f}/1.0（影响回应详细程度）
- 信息透露度：{self.information_revealed:.1f}/1.0

情绪状态指导：
{self._get_emotion_guidance()}

行为调整建议：
{self._get_behavior_guidance()}

回复要求：
- 保持角色一致性，体现当前的信任度和情绪状态
- 根据咨询师的话语自然调整情绪和开放程度
- 使用符合大学生身份的语言表达
- 回复长度控制在50-200字之间
- 可以适当通过语句内容表现出犹豫、不确定、回避等真实反应
- 仅返回语句内容，不要有行为或表情的描述
"""

        return base_prompt + current_state_info

    def _get_emotion_guidance(self) -> str:
        """获取当前情绪的行为指导"""
        emotion_guides = {
            EmotionState.ANXIOUS: "表现出紧张、担心，语速可能较快，容易转移话题",
            EmotionState.DEPRESSED: "语调低沉，回应较少，可能表达无助感",
            EmotionState.CONFUSED: "表现出困惑、不确定，经常说'不知道'",
            EmotionState.ANGRY: "语气可能较冲，容易情绪化，可能对建议有抗拒",
            EmotionState.CALM: "相对平静，能够理性交流",
            EmotionState.HOPEFUL: "积极一些，愿意尝试建议",
            EmotionState.RESISTANT: "对咨询师的话有质疑，可能不太配合",
            EmotionState.TRUSTING: "更愿意分享，语气较为放松",
            EmotionState.AVOIDANT: "回避深入话题，可能转移话题",
            EmotionState.OPEN: "比较愿意交流，会分享更多细节",
            EmotionState.OTHER: "根据具体情况灵活表现",
        }

        return emotion_guides.get(self.current_emotion, "保持自然的情绪表达")

    def _get_behavior_guidance(self) -> str:
        """获取行为指导建议"""
        guidance = []

        if self.trust_level < 0.3:
            guidance.append("信任度较低：保持谨慎，只分享表面信息")
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

    def convert_history_to_messages(
        self, conversation_history: List[ConversationMessage]
    ) -> List[Dict[str, str]]:
        converted_history = []
        for msg in conversation_history:
            if msg.role == "counselor":
                # 咨询师的消息作为assistant
                converted_history.append({"role": "assistant", "content": msg.content})
            elif msg.role == "student":
                # 学生的消息作为user
                converted_history.append({"role": "user", "content": msg.content})
        return converted_history

    async def generate_student_response(
        self, conversation_history: List[ConversationMessage]
    ) -> str:
        """
        生成学生的回复

        Args:
            conversation_history: 完整的对话历史

        Returns:
            str: 学生的回复
        """
        # 更新内部对话历史
        self.conversation_history = conversation_history
        self.current_round = (
            len([msg for msg in conversation_history if msg.role == "student"]) + 1
        )

        # 分析咨询师的最后回复，调整状态
        if conversation_history:
            last_counselor_msg = None
            for msg in reversed(conversation_history):
                if msg.role == "counselor":
                    last_counselor_msg = msg
                    break

            if last_counselor_msg:
                await self._analyze_and_adjust_state(last_counselor_msg.content)

        # 生成回复
        response = await self.generate_response(
            for_role="student", conversation_history=conversation_history
        )

        # 评估风险
        risk_assessment = self.assess_risk(response)
        if risk_assessment.emergency_required:
            # 如果风险过高，生成更安全的回复
            response = await self._generate_safer_response()

        # 记录情绪变化
        self._update_conversation_metadata()

        return response

    async def _analyze_and_adjust_state(self, counselor_response: str):
        """
        分析咨询师回复并调整学生状态

        Args:
            counselor_response: 咨询师的回复内容
        """
        # 基于咨询师回复的特征调整状态
        response_lower = counselor_response.lower()

        # 信任度调整
        trust_increase_indicators = [
            "理解",
            "感受",
            "不容易",
            "我能感受到",
            "听起来",
            "对你来说",
            "你的感受",
            "这很困难",
        ]
        trust_decrease_indicators = ["应该", "必须", "你需要", "建议你", "你要", "最好"]

        trust_change = 0
        for indicator in trust_increase_indicators:
            if indicator in response_lower:
                trust_change += 0.05

        for indicator in trust_decrease_indicators:
            if indicator in response_lower:
                trust_change -= 0.02

        self.trust_level = max(0, min(1, self.trust_level + trust_change))

        # 开放度调整
        if any(word in response_lower for word in ["具体", "详细", "能说说", "比如"]):
            self.openness_level = min(1, self.openness_level + 0.03)

        # 信息透露度调整
        if self.trust_level > 0.6:
            self.information_revealed = min(1, self.information_revealed + 0.1)
        elif self.trust_level > 0.3:
            self.information_revealed = min(1, self.information_revealed + 0.05)

        # 情绪状态调整
        await self._adjust_emotion_state(counselor_response)

    async def _adjust_emotion_state(self, counselor_response: str):
        """调整情绪状态"""
        response_lower = counselor_response.lower()

        # 积极因素
        positive_indicators = ["理解", "正常的", "可以理解", "不用担心", "我们一起"]
        # 中性因素
        neutral_indicators = ["能说说", "具体", "什么时候", "怎么样"]
        # 可能引起抗拒的因素
        resistance_indicators = ["为什么", "原因", "你觉得", "有没有想过"]

        current_transitions = EMOTION_TRANSITIONS.get(self.current_emotion, [])

        if any(indicator in response_lower for indicator in positive_indicators):
            # 倾向于积极情绪转换
            positive_emotions = [
                EmotionState.CALM,
                EmotionState.TRUSTING,
                EmotionState.HOPEFUL,
            ]
            available_positive = [
                e for e in current_transitions if e in positive_emotions
            ]
            if available_positive:
                new_emotion = random.choice(available_positive)
                self.update_state(new_emotion, "咨询师表现出理解和支持")

        elif any(indicator in response_lower for indicator in resistance_indicators):
            # 可能引起轻微抗拒或回避
            if random.random() < self.resistance_level:
                negative_emotions = [EmotionState.RESISTANT, EmotionState.AVOIDANT]
                available_negative = [
                    e for e in current_transitions if e in negative_emotions
                ]
                if available_negative:
                    new_emotion = random.choice(available_negative)
                    self.update_state(new_emotion, "对咨询师的深入询问感到抗拒")

    async def _generate_safer_response(self) -> str:
        """生成更安全的回复（当检测到风险时）"""
        safe_responses = [
            "我...我只是有时候会这样想，可能不是什么大问题。",
            "嗯，我觉得我需要时间想想。",
            "可能我说得有点过了，其实也没那么严重。",
            "我不太确定该怎么表达，就是感觉很难受。",
        ]

        return random.choice(safe_responses)

    def _update_conversation_metadata(self):
        """更新对话元数据"""
        if self.conversation_history:
            last_msg = self.conversation_history[-1]
            if last_msg.role == "student":
                last_msg.emotion = self.current_emotion

    def get_student_state(self) -> Dict[str, Any]:
        """获取学生当前状态信息"""
        return {
            "bot_id": self.bot_id,
            "current_round": self.current_round,
            "current_emotion": self.current_emotion.value,
            "trust_level": self.trust_level,
            "openness_level": self.openness_level,
            "information_revealed": self.information_revealed,
            "avoidance_tendency": self.avoidance_tendency,
            "resistance_level": self.resistance_level,
            "message_count": len(self.conversation_history),
        }

    def simulate_personality_traits(self) -> Dict[str, float]:
        """模拟个性特征影响"""
        if not self.student_background:
            return {}

        traits = self.student_background.personality_traits
        trait_effects = {}

        for trait in traits:
            if "内向" in trait:
                trait_effects["introversion"] = 0.7
                self.openness_level *= 0.8
            elif "外向" in trait:
                trait_effects["extraversion"] = 0.7
                self.chattiness *= 1.2
            elif "敏感" in trait:
                trait_effects["sensitivity"] = 0.8
                self.resistance_level *= 1.3
            elif "完美主义" in trait:
                trait_effects["perfectionism"] = 0.6
                self.avoidance_tendency *= 1.2

        return trait_effects

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
