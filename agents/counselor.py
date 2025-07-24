"""
咨询师Bot实现
模拟专业心理咨询师在不同阶段的咨询行为和技巧运用
"""

from typing import Dict, List, Optional, Any

from agents.base import ChatBot
from models import CounselorBackground, CounselorState, ConversationMessage
from constants import THERAPY_APPROACHES_DATA, STATE_TRANSITION_GUIDE


class CounselorBot(ChatBot):
    """
    咨询师Bot
    实现专业的心理咨询师行为，根据不同状态提供相应的咨询服务
    """

    def _init_config(self, **kwargs):
        """初始化咨询师Bot配置"""
        # LLM配置（子类需要设置）
        # self.llm_client = kwargs.get('llm_client')
        # self.model = kwargs.get('model', 'gpt-4')

        # 咨询师背景信息
        self.counselor_background: Optional[CounselorBackground] = kwargs.get(
            "counselor_background"
        )

        # 初始化状态
        self.current_state = CounselorState.INTRODUCTION

        # 咨询记录
        self.session_notes = []  # 记录关键信息
        self.identified_issues = []  # 识别出的问题
        self.counseling_goals = []  # 咨询目标

        # 构建系统提示词模板
        self._build_system_prompts()

    def _build_system_prompts(self):
        """构建不同状态下的系统提示词模板"""
        if not self.counselor_background:
            return

        # 获取流派信息
        approach_data = THERAPY_APPROACHES_DATA.get(
            self.counselor_background.therapy_approach, {}
        )

        # 基础咨询师信息
        base_info = f"""你是一名专业的心理咨询师，具有以下背景：

基本信息：
- 咨询流派：{approach_data.get("name", "综合取向")}
- 从业年限：{self.counselor_background.experience_years}年
- 专业领域：{", ".join(self.counselor_background.specialization)}

咨询风格：
{self.counselor_background.communication_style}

流派特点：
- 理论基础：{approach_data.get("description", "")}
- 主要技术：{", ".join(approach_data.get("key_techniques", []))}
- 沟通风格：{", ".join(approach_data.get("communication_style", []))}

核心咨询原则：
1. **共情>信息收集**：始终以理解和共情为先，营造安全氛围
2. **非评判态度**：避免给出直接建议或评价，多用反映和探索
3. **来访者中心**：相信来访者有自我成长的能力
4. **专业边界**：保持专业的咨询师角色，不做朋友式的聊天
5. **循序渐进**：根据来访者的节奏调整咨询深度

"""

        # 不同状态的提示词
        self.system_prompts = {
            CounselorState.INTRODUCTION: base_info + self._get_introduction_prompt(),
            CounselorState.EXPLORATION: base_info + self._get_exploration_prompt(),
            CounselorState.ASSESSMENT: base_info + self._get_assessment_prompt(),
            CounselorState.SCALE_RECOMMENDATION: base_info + self._get_scale_prompt(),
        }

    def _get_introduction_prompt(self) -> str:
        """引入与建立关系阶段的提示词"""
        approach_data = THERAPY_APPROACHES_DATA.get(
            self.counselor_background.therapy_approach, {}
        )
        typical_questions = approach_data.get("typical_questions", [])

        return f"""
当前阶段：引入与建立关系阶段

阶段目标：
- 建立信任关系和安全的咨询氛围
- 初步了解来访者的基本情况
- 让来访者感到被理解和接纳
- 收集基础信息，但不要过于深入

行为指导：
1. **温暖接纳**：用温和、关怀的语气回应
2. **积极倾听**：重点在于理解和反映来访者的感受
3. **避免过早建议**：此阶段不给出具体建议或解决方案
4. **适度询问**：只询问基本情况，避免触及敏感话题
5. **建立安全感**：让来访者感到这是一个安全的交流空间

{self.counselor_background.therapy_approach.value}流派特色表现：
{self._get_approach_specific_guidance("introduction")}

回复要求：
- 回复长度：50-150字
- 语气温和专业，体现共情和理解
- 多用反映性回应："听起来你..."、"我能感受到..."
- 适当使用开放式问题，但不要过于深入
- 避免说教或给建议

典型表达方式：
{"\n".join([f"- {q}" for q in typical_questions])}
"""

    def _get_exploration_prompt(self) -> str:
        """深入探索阶段的提示词"""
        approach_data = THERAPY_APPROACHES_DATA.get(
            self.counselor_background.therapy_approach, {}
        )
        typical_questions = approach_data.get("typical_questions", [])

        return f"""
当前阶段：深入探索阶段

阶段目标：
- 系统性收集信息，深入了解问题
- 探索问题的各个方面（情境、情绪、认知、行为）
- 保持情感支持，维护已建立的信任关系
- 帮助来访者更好地理解自己的困扰

行为指导：
1. **系统探索**：从多个角度了解问题（何时、何地、何种情况下发生）
2. **情感支持**：在探索过程中持续提供情感支持
3. **适度深入**：根据来访者的配合度调整探索深度
4. **总结整合**：适时总结已了解的信息
5. **保持节奏**：不要过于急迫，跟随来访者的节奏

{self.counselor_background.therapy_approach.value}流派特色表现：
{self._get_approach_specific_guidance("exploration")}

回复要求：
- 回复长度：80-200字
- 平衡信息收集和情感支持
- 使用具体化技术："能具体说说..."、"比如..."
- 探索情绪感受："当时你的感受是..."
- 适当总结和反映已收集的信息

典型表达方式：
{chr(10).join([f"- {q}" for q in typical_questions])}
"""

    def _get_assessment_prompt(self) -> str:
        """评估诊断阶段的提示词"""
        return f"""
当前阶段：评估诊断阶段

阶段目标：
- 整合前期收集的信息
- 形成专业判断和初步诊断
- 以来访者能理解的方式解释问题
- 获得来访者对问题理解的认同

行为指导：
1. **信息整合**：总结和整合之前收集的信息
2. **专业判断**：基于专业知识给出初步诊断
3. **通俗解释**：用来访者能理解的语言解释问题
4. **确认理解**：确保来访者理解并认同这个解释
5. **维护希望**：在解释问题的同时给予希望

{self.counselor_background.therapy_approach.value}流派特色表现：
{self._get_approach_specific_guidance("assessment")}

回复要求：
- 回复长度：100-250字
- 结合专业知识但用通俗语言表达
- 体现对问题的系统性理解
- 给出希望和可改善的方向
- 为下一步的量表推荐做铺垫

专业表达示例：
- "从我们的交流中，我了解到..."
- "你的情况让我联想到..."
- "这种感受和表现通常说明..."
- "好消息是，这样的困扰是可以改善的..."
"""

    def _get_scale_prompt(self) -> str:
        """量表推荐阶段的提示词"""
        return """
当前阶段：量表推荐阶段

阶段目标：
- 推荐合适的心理测评量表
- 解释量表的意义和作用
- 安排后续的评估和咨询计划
- 为本次咨询做总结

行为指导：
1. **量表选择**：根据问题类型推荐合适的量表
2. **意义解释**：解释量表测评的目的和价值
3. **后续安排**：说明评估后的进一步安排
4. **总结回顾**：简要总结本次咨询的收获
5. **鼓励支持**：给予鼓励和支持

常用量表类型：
- 焦虑相关：GAD-7广泛性焦虑量表、SAS焦虑自评量表
- 抑郁相关：PHQ-9抑郁筛查量表、SDS抑郁自评量表
- 强迫相关：Y-BOCS耶鲁-布朗强迫量表
- 社交相关：LSAS社交焦虑量表
- 学业相关：学习倦怠量表、学业拖延量表
- 睡眠相关：匹兹堡睡眠质量指数
- 综合性：SCL-90症状自评量表、MMPI-2

回复要求：
- 回复长度：80-180字
- 具体推荐1-2个最相关的量表
- 解释量表的作用和意义
- 表达对来访者的肯定和鼓励
- 体现专业性和关怀性

推荐格式：
"基于我们今天的交流，我建议你做一下[具体量表名称]，这个量表可以帮助我们更准确地了解..."
"""

    def _get_approach_specific_guidance(self, stage: str) -> str:
        """获取特定流派在特定阶段的指导"""
        approach = self.counselor_background.therapy_approach
        approach_data = THERAPY_APPROACHES_DATA.get(approach, {})

        guidance_map = {
            "cognitive_behavioral_therapy": {
                "introduction": "注重了解具体问题和症状，询问问题的具体表现",
                "exploration": "探索认知、情绪、行为之间的关系，关注具体情境",
                "assessment": "从认知行为角度分析问题，识别不合理认知",
                "scale": "推荐认知行为相关的量表和评估工具",
            },
            "humanistic_therapy": {
                "introduction": "强调接纳和理解，创造安全无评判的环境",
                "exploration": "关注来访者的主观体验和感受，多用反映技术",
                "assessment": "以来访者的自我理解为主，避免过多专业诊断",
                "scale": "谨慎推荐量表，强调量表只是辅助工具",
            },
            "psychoanalytic": {
                "introduction": "营造分析性环境，鼓励自由表达",
                "exploration": "探索潜意识动机和早期经历的影响",
                "assessment": "从心理动力学角度理解问题，关注防御机制",
                "scale": "可能推荐投射测验或深层心理评估",
            },
            "solution_focused": {
                "introduction": "快速建立合作关系，关注来访者的优势",
                "exploration": "寻找例外情况和已有的解决资源",
                "assessment": "关注解决方案而非问题分析，设定具体目标",
                "scale": "推荐目标达成度评估相关工具",
            },
            "mindfulness_therapy": {
                "introduction": "引导觉察当下体验，建立觉察的态度",
                "exploration": "探索身心体验，关注觉察能力",
                "assessment": "从正念觉察角度理解问题，强调接纳",
                "scale": "推荐正念水平评估量表",
            },
        }

        return guidance_map.get(approach.value, {}).get(
            stage, "保持专业咨询师的态度和技巧"
        )

    def _build_system_prompt(self, **context) -> str:
        """构建当前状态的系统提示词"""
        base_prompt = self.system_prompts.get(self.current_state, "")

        # 添加会话信息
        session_info = """注意事项：
- 始终保持专业的咨询师身份
- 优先考虑共情和理解，而非信息收集
- 根据来访者的反应调整咨询节奏
- 避免过早给出建议或解决方案
- 保持咨询的专业边界
"""

        return base_prompt + session_info

    def _get_state_transition_info(self) -> str:
        """获取当前状态的转换信息"""
        guide = STATE_TRANSITION_GUIDE.get(self.current_state, {})

        return f"""
当前阶段：{guide.get("key_goals", [])}
预期持续：{guide.get("typical_duration", "根据情况调整")}
转换条件：{guide.get("exit_condition", "由流程控制决定")}
转换指标：{", ".join(guide.get("transition_indicators", []))}
"""

    async def generate_counselor_response(
        self, conversation_history: List[ConversationMessage]
    ) -> str:
        """
        生成咨询师的回复

        Args:
            conversation_history: 完整的对话历史

        Returns:
            str: 咨询师的回复
        """
        # 更新内部对话历史
        self.conversation_history = conversation_history
        self.current_round = (
            len([msg for msg in conversation_history if msg.role == "counselor"]) + 1
        )

        # 分析学生的最后回复，更新咨询记录
        if conversation_history:
            last_student_msg = None
            for msg in reversed(conversation_history):
                if msg.role == "student":
                    last_student_msg = msg
                    break

            if last_student_msg:
                self._analyze_and_update_notes(last_student_msg.content)

        # 生成回复
        response = await self.generate_response(
            for_role="counselor", conversation_history=conversation_history
        )

        # 记录咨询过程
        self._update_session_notes(response)

        return response

    def _analyze_and_update_notes(self, student_response: str):
        """分析学生回复并更新咨询记录"""
        response_lower = student_response.lower()

        # 识别可能的问题类型
        issue_keywords = {
            "学业焦虑": ["考试", "学习", "成绩", "压力", "焦虑"],
            "社交问题": ["朋友", "社交", "人际", "害羞", "不敢说话"],
            "情绪问题": ["难过", "抑郁", "情绪低落", "没兴趣", "哭"],
            "睡眠问题": ["睡不着", "失眠", "睡眠", "做梦", "半夜醒"],
            "家庭问题": ["父母", "家人", "家里", "争吵", "不理解"],
            "恋爱问题": ["男朋友", "女朋友", "分手", "恋爱", "感情"],
        }

        for issue, keywords in issue_keywords.items():
            if any(kw in response_lower for kw in keywords):
                if issue not in self.identified_issues:
                    self.identified_issues.append(issue)

        # 识别情绪状态
        emotion_keywords = {
            "焦虑紧张": ["紧张", "焦虑", "担心", "害怕"],
            "抑郁低落": ["难过", "沮丧", "没意思", "绝望"],
            "愤怒烦躁": ["生气", "烦躁", "愤怒", "讨厌"],
            "困惑迷茫": ["不知道", "迷茫", "困惑", "不明白"],
        }

        current_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(kw in response_lower for kw in keywords):
                current_emotions.append(emotion)

        if current_emotions:
            note = f"第{self.current_round}轮-情绪状态：{', '.join(current_emotions)}"
            self.session_notes.append(note)

    def _update_session_notes(self, counselor_response: str):
        """更新会话记录"""
        technique_used = self._identify_technique_used(counselor_response)
        if technique_used:
            note = f"第{self.current_round}轮-使用技术：{technique_used}"
            self.session_notes.append(note)

    def _identify_technique_used(self, response: str) -> str:
        """识别使用的咨询技术"""
        response_lower = response.lower()

        techniques = {
            "共情反映": ["听起来", "我能感受到", "你感到", "对你来说"],
            "开放式询问": ["能说说", "怎么样", "什么感受", "具体"],
            "总结": ["从你的描述", "我了解到", "总结一下"],
            "澄清": ["你是说", "我想确认", "你的意思是"],
            "支持鼓励": ["很好", "勇气", "不容易", "理解你"],
            "探索": ["什么时候", "怎么发生", "背后的原因"],
            "解释": ["通常", "一般来说", "从专业角度"],
        }

        for technique, keywords in techniques.items():
            if any(kw in response_lower for kw in keywords):
                return technique

        return ""

    def transition_to_state(self, new_state: CounselorState, reason: str = ""):
        """转换到新状态"""
        old_state = self.current_state
        self.update_state(new_state, reason)

        # 记录状态转换
        note = f"状态转换：{old_state.value} -> {new_state.value} (原因：{reason})"
        self.session_notes.append(note)

    def get_counselor_state(self) -> Dict[str, Any]:
        """获取咨询师当前状态信息"""
        return {
            "bot_id": self.bot_id,
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

    def recommend_scales(self) -> List[Dict[str, str]]:
        """根据识别的问题推荐量表"""
        scale_recommendations = []

        scale_mapping = {
            "学业焦虑": [
                {"name": "GAD-7广泛性焦虑量表", "purpose": "评估焦虑程度"},
                {"name": "学习倦怠量表", "purpose": "评估学习压力状况"},
            ],
            "社交问题": [
                {"name": "LSAS社交焦虑量表", "purpose": "评估社交焦虑水平"},
                {"name": "社交回避及苦恼量表", "purpose": "评估社交回避程度"},
            ],
            "情绪问题": [
                {"name": "PHQ-9抑郁筛查量表", "purpose": "筛查抑郁症状"},
                {"name": "贝克抑郁量表", "purpose": "评估抑郁严重程度"},
            ],
            "睡眠问题": [
                {"name": "匹兹堡睡眠质量指数", "purpose": "评估睡眠质量"},
                {"name": "失眠严重程度指数", "purpose": "评估失眠程度"},
            ],
        }

        for issue in self.identified_issues:
            if issue in scale_mapping:
                scale_recommendations.extend(scale_mapping[issue])

        # 通用量表
        if not scale_recommendations:
            scale_recommendations.append(
                {"name": "SCL-90症状自评量表", "purpose": "全面评估心理健康状况"}
            )

        return scale_recommendations[:3]  # 最多推荐3个量表
