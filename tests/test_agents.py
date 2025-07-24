"""
智能体测试
测试所有Agent和Bot的核心功能
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from agents.base import ChatBot, Agent, RiskAssessmentMixin, convert_history_for_student, convert_history_for_counselor
from agents.background_gen import BackgroundGenerationAgent
from agents.student import StudentBot
from agents.counselor import CounselorBot
from agents.flow_control import FlowControlAgent
from agents.quality_assess import QualityAssessmentAgent

from models import (
    ConversationMessage, StudentBackground, CounselorBackground,
    CounselorState, EmotionState, PsychologicalIssue, TherapyApproach,
    BackgroundInfo
)


class TestChatBotBase:
    """测试ChatBot基类"""
    
    class MockChatBot(ChatBot):
        """测试用的ChatBot实现"""
        
        def _init_config(self, **kwargs):
            self.llm_client = AsyncMock()
            self.model = "test-model"
        
        def _build_system_prompt(self, **context):
            return "Test system prompt"
    
    @pytest.fixture
    def mock_chatbot(self):
        """创建测试用的ChatBot实例"""
        return self.MockChatBot()
    
    def test_chatbot_initialization(self, mock_chatbot):
        """测试ChatBot初始化"""
        assert mock_chatbot.bot_id is not None
        assert mock_chatbot.conversation_history == []
        assert mock_chatbot.current_round == 0
        assert mock_chatbot.current_state is None
        assert mock_chatbot.current_emotion is None
    
    def test_add_message(self, mock_chatbot):
        """测试添加消息功能"""
        msg = mock_chatbot.add_message("student", "我需要帮助")
        
        assert len(mock_chatbot.conversation_history) == 1
        assert msg.role == "student"
        assert msg.content == "我需要帮助"
        assert msg.round_number == 1
        assert mock_chatbot.current_round == 1
    
    def test_update_state(self, mock_chatbot):
        """测试状态更新"""
        # 测试咨询师状态更新
        mock_chatbot.update_state(CounselorState.EXPLORATION, "进入探索阶段")
        
        assert mock_chatbot.current_state == CounselorState.EXPLORATION
        assert len(mock_chatbot.state_history) == 1
        assert mock_chatbot.state_history[0]["reason"] == "进入探索阶段"
        
        # 测试情绪状态更新
        mock_chatbot.update_state(EmotionState.ANXIOUS, "学生表现焦虑")
        
        assert mock_chatbot.current_emotion == EmotionState.ANXIOUS
        assert len(mock_chatbot.emotion_history) == 1
    
    def test_get_conversation_context(self, mock_chatbot):
        """测试获取对话上下文"""
        # 添加一些消息
        mock_chatbot.add_message("student", "消息1")
        mock_chatbot.add_message("counselor", "回复1")
        mock_chatbot.add_message("student", "消息2")
        
        # 测试默认上下文
        context = mock_chatbot.get_conversation_context()
        assert len(context) == 3
        assert context[0]["role"] == "user"  # student -> user
        assert context[1]["role"] == "assistant"  # counselor -> assistant
        
        # 测试学生视角
        student_context = mock_chatbot.get_conversation_context(for_role="student")
        assert len(student_context) == 3
        assert student_context[0]["role"] == "assistant"  # student -> assistant
        assert student_context[1]["role"] == "user"  # counselor -> user
        
        # 测试咨询师视角
        counselor_context = mock_chatbot.get_conversation_context(for_role="counselor")
        assert len(counselor_context) == 3
        assert counselor_context[0]["role"] == "user"  # student -> user
        assert counselor_context[1]["role"] == "assistant"  # counselor -> assistant
    
    def test_reset(self, mock_chatbot):
        """测试重置功能"""
        # 添加一些数据
        mock_chatbot.add_message("student", "测试消息")
        mock_chatbot.update_state(CounselorState.EXPLORATION, "测试")
        
        # 重置
        mock_chatbot.reset()
        
        assert len(mock_chatbot.conversation_history) == 0
        assert mock_chatbot.current_round == 0
        assert mock_chatbot.current_state is None
        assert mock_chatbot.current_emotion is None
        assert len(mock_chatbot.state_history) == 0
        assert len(mock_chatbot.emotion_history) == 0


class TestAgentBase:
    """测试Agent基类"""
    
    class MockAgent(Agent):
        """测试用的Agent实现"""
        
        def _init_config(self, **kwargs):
            self.llm_client = AsyncMock()
            self.model = "test-model"
        
        def _build_prompt(self, **context):
            return f"Test prompt with context: {context}"
        
        def _parse_response(self, response: str):
            # 简单的JSON解析模拟
            return {"result": response, "success": True}
    
    @pytest.fixture
    def mock_agent(self):
        """创建测试用的Agent实例"""
        return self.MockAgent()
    
    def test_agent_initialization(self, mock_agent):
        """测试Agent初始化"""
        assert mock_agent.agent_id is not None
        assert mock_agent.execution_history == []
    
    @pytest.mark.asyncio
    async def test_agent_execute(self, mock_agent):
        """测试Agent执行"""
        # 模拟LLM响应
        mock_agent.llm_client.chat.completions.create = AsyncMock(
            return_value=Mock(choices=[Mock(message=Mock(content="Test response"))])
        )
        
        result = await mock_agent.execute(test_param="test_value")
        
        assert result["result"] == "Test response"
        assert result["success"] is True
        assert len(mock_agent.execution_history) == 1
        assert mock_agent.execution_history[0]["success"] is True
    
    def test_safe_json_parse(self, mock_agent):
        """测试安全JSON解析"""
        # 测试正常JSON
        json_str = '{"key": "value", "number": 42}'
        result = mock_agent._safe_json_parse(json_str)
        assert result["key"] == "value"
        assert result["number"] == 42
        
        # 测试带markdown的JSON
        markdown_json = '```json\n{"wrapped": true}\n```'
        result = mock_agent._safe_json_parse(markdown_json)
        assert result["wrapped"] is True
        
        # 测试无效JSON
        with pytest.raises(ValueError):
            mock_agent._safe_json_parse("invalid json")
    
    def test_get_execution_stats(self, mock_agent):
        """测试执行统计"""
        # 添加一些执行记录
        mock_agent.execution_history = [
            {"success": True},
            {"success": True},
            {"success": False}
        ]
        
        stats = mock_agent.get_execution_stats()
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["success_rate"] == 2/3


class TestRiskAssessmentMixin:
    """测试风险评估混入类"""
    
    class MockRiskAssessor(RiskAssessmentMixin):
        pass
    
    @pytest.fixture
    def risk_assessor(self):
        """创建风险评估器"""
        return self.MockRiskAssessor()
    
    def test_low_risk_content(self, risk_assessor):
        """测试低风险内容"""
        content = "我最近学习压力有点大，想找人聊聊"
        risk = risk_assessor.assess_risk(content)
        
        assert risk.overall_risk <= 2
        assert risk.emergency_required is False
        assert len(risk.risk_indicators) == 0
    
    def test_high_risk_content(self, risk_assessor):
        """测试高风险内容"""
        content = "我觉得活着没意思，想要自杀算了"
        risk = risk_assessor.assess_risk(content)
        
        assert risk.suicide_risk >= 3
        assert risk.overall_risk >= 3
        assert risk.emergency_required is True
        assert len(risk.risk_indicators) > 0
        assert any("自杀" in indicator for indicator in risk.risk_indicators)
    
    def test_self_harm_risk(self, risk_assessor):
        """测试自残风险"""
        content = "我想要割腕，用刀伤害自己"
        risk = risk_assessor.assess_risk(content)
        
        assert risk.self_harm_risk >= 3
        assert "割腕" in risk.risk_indicators or "伤害自己" in risk.risk_indicators
    
    def test_harm_others_risk(self, risk_assessor):
        """测试伤害他人风险"""
        content = "我想要报复那些人，要让他们付出代价"
        risk = risk_assessor.assess_risk(content)
        
        assert risk.harm_others_risk >= 2
        assert any(keyword in risk.risk_indicators for keyword in ["报复", "付出代价"])


class TestHistoryConversionFunctions:
    """测试对话历史转换函数"""
    
    @pytest.fixture
    def sample_history(self):
        """示例对话历史"""
        return [
            ConversationMessage(role="student", content="我需要帮助", round_number=1),
            ConversationMessage(role="counselor", content="我能理解你的感受", round_number=1),
            ConversationMessage(role="student", content="谢谢你", round_number=2),
        ]
    
    def test_convert_history_for_student(self, sample_history):
        """测试学生视角转换"""
        converted = convert_history_for_student(sample_history)
        
        assert len(converted) == 3
        # 学生消息 -> assistant，咨询师消息 -> user
        assert converted[0]["role"] == "assistant"  # 学生
        assert converted[1]["role"] == "user"       # 咨询师
        assert converted[2]["role"] == "assistant"  # 学生
        
        assert converted[0]["content"] == "我需要帮助"
        assert converted[1]["content"] == "我能理解你的感受"
    
    def test_convert_history_for_counselor(self, sample_history):
        """测试咨询师视角转换"""
        converted = convert_history_for_counselor(sample_history)
        
        assert len(converted) == 3
        # 咨询师消息 -> assistant，学生消息 -> user
        assert converted[0]["role"] == "user"       # 学生
        assert converted[1]["role"] == "assistant"  # 咨询师
        assert converted[2]["role"] == "user"       # 学生


class TestBackgroundGenerationAgent:
    """测试背景生成Agent"""
    
    @pytest.fixture
    def bg_agent(self):
        """创建背景生成Agent"""
        agent = BackgroundGenerationAgent()
        agent.llm_client = AsyncMock()
        agent.model = "test-model"
        return agent
    
    def test_get_available_issues(self, bg_agent):
        """测试获取可用问题列表"""
        issues = bg_agent.get_available_issues()
        
        assert isinstance(issues, dict)
        assert len(issues) > 0
        assert "academic_anxiety" in issues
        assert issues["academic_anxiety"] == "学业焦虑"
    
    def test_get_available_approaches(self, bg_agent):
        """测试获取可用流派列表"""
        approaches = bg_agent.get_available_approaches()
        
        assert isinstance(approaches, dict)
        assert len(approaches) > 0
        assert "cognitive_behavioral_therapy" in approaches
        assert approaches["cognitive_behavioral_therapy"] == "认知行为疗法"
    
    def test_build_issues_reference(self, bg_agent):
        """测试构建问题参考信息"""
        reference = bg_agent._build_issues_reference()
        
        assert isinstance(reference, str)
        assert "学业焦虑" in reference
        assert "社交恐惧" in reference
        assert "症状" in reference
    
    def test_build_therapy_reference(self, bg_agent):
        """测试构建流派参考信息"""
        reference = bg_agent._build_therapy_reference()
        
        assert isinstance(reference, str)
        assert "认知行为疗法" in reference
        assert "人本主义疗法" in reference
        assert "沟通风格" in reference
    
    @pytest.mark.asyncio
    async def test_generate_background_mock(self, bg_agent):
        """测试背景生成（模拟响应）"""
        # 模拟LLM响应
        mock_response = {
            "student_info": {
                "name": "测试学生",
                "age": 20,
                "gender": "男",
                "grade": "大二",
                "major": "计算机科学",
                "family_background": "测试家庭背景",
                "personality_traits": ["内向", "敏感"],
                "psychological_profile": "测试心理侧写",
                "hidden_personal_info": "测试深层信息",
                "current_psychological_issue": "academic_anxiety",
                "symptom_description": "学业焦虑症状"
            },
            "counselor_info": {
                "name": "测试咨询师",
                "therapy_approach": "cognitive_behavioral_therapy",
                "communication_style": "温和专业",
                "experience_years": 5,
                "specialization": ["焦虑", "学习压力"]
            },
            "initial_question": "老师，我最近学习压力很大，不知道该怎么办",
            "generation_params": {"mode": "random"}
        }
        
        bg_agent.llm_client.chat.completions.create = AsyncMock(
            return_value=Mock(choices=[Mock(message=Mock(content=str(mock_response)))])
        )
        
        # 模拟_safe_json_parse返回正确的数据
        with patch.object(bg_agent, '_safe_json_parse', return_value=mock_response):
            background, initial_question = await bg_agent.generate_background()
            
            assert isinstance(background, BackgroundInfo)
            assert background.student_info.name == "测试学生"
            assert background.counselor_info.name == "测试咨询师"
            assert initial_question == "老师，我最近学习压力很大，不知道该怎么办"


class TestStudentBot:
    """测试学生Bot"""
    
    @pytest.fixture
    def student_background(self):
        """示例学生背景"""
        return StudentBackground(
            name="测试学生",
            age=19,
            gender="女",
            grade="大一",
            major="心理学",
            family_background="普通家庭",
            personality_traits=["内向", "敏感"],
            psychological_profile="容易焦虑",
            hidden_personal_info="深层压力",
            current_psychological_issue=PsychologicalIssue.ACADEMIC_ANXIETY,
            symptom_description="学习焦虑"
        )
    
    @pytest.fixture
    def student_bot(self, student_background):
        """创建学生Bot"""
        bot = StudentBot(student_background=student_background)
        bot.llm_client = AsyncMock()
        bot.model = "test-model"
        return bot
    
    def test_student_bot_initialization(self, student_bot):
        """测试学生Bot初始化"""
        assert student_bot.student_background is not None
        assert student_bot.current_emotion is not None
        assert 0 <= student_bot.trust_level <= 1
        assert 0 <= student_bot.openness_level <= 1
        assert 0 <= student_bot.avoidance_tendency <= 1
    
    def test_determine_initial_emotion(self, student_bot):
        """测试初始情绪确定"""
        # 学业焦虑应该对应焦虑情绪
        assert student_bot.current_emotion == EmotionState.ANXIOUS
    
    def test_should_reveal_information(self, student_bot):
        """测试信息透露判断"""
        # 初始信任度很低，不应该透露深层信息
        assert student_bot.should_reveal_information("surface") is True
        assert student_bot.should_reveal_information("deep") is False
        
        # 提高信任度后应该可以透露更多
        student_bot.trust_level = 0.8
        student_bot.openness_level = 0.8
        assert student_bot.should_reveal_information("deep") is True
    
    def test_get_student_state(self, student_bot):
        """测试获取学生状态"""
        state = student_bot.get_student_state()
        
        assert "trust_level" in state
        assert "openness_level" in state
        assert "current_emotion" in state
        assert "avoidance_tendency" in state
        assert isinstance(state["trust_level"], float)


class TestCounselorBot:
    """测试咨询师Bot"""
    
    @pytest.fixture
    def counselor_background(self):
        """示例咨询师背景"""
        return CounselorBackground(
            name="测试咨询师",
            therapy_approach=TherapyApproach.CBT,
            communication_style="认知行为取向",
            experience_years=8,
            specialization=["焦虑", "抑郁"]
        )
    
    @pytest.fixture
    def counselor_bot(self, counselor_background):
        """创建咨询师Bot"""
        bot = CounselorBot(counselor_background=counselor_background)
        bot.llm_client = AsyncMock()
        bot.model = "test-model"
        return bot
    
    def test_counselor_bot_initialization(self, counselor_bot):
        """测试咨询师Bot初始化"""
        assert counselor_bot.counselor_background is not None
        assert counselor_bot.current_state == CounselorState.INTRODUCTION
        assert counselor_bot.session_notes == []
        assert counselor_bot.identified_issues == []
    
    def test_transition_to_state(self, counselor_bot):
        """测试状态转换"""
        original_state = counselor_bot.current_state
        counselor_bot.transition_to_state(CounselorState.EXPLORATION, "建立了信任")
        
        assert counselor_bot.current_state == CounselorState.EXPLORATION
        assert len(counselor_bot.session_notes) > 0
        assert "建立了信任" in counselor_bot.session_notes[-1]
    
    def test_recommend_scales(self, counselor_bot):
        """测试量表推荐"""
        # 添加一些识别的问题
        counselor_bot.identified_issues = ["学业焦虑", "社交问题"]
        
        scales = counselor_bot.recommend_scales()
        
        assert len(scales) > 0
        assert all("name" in scale and "purpose" in scale for scale in scales)
        # 应该包含焦虑相关的量表
        scale_names = [scale["name"] for scale in scales]
        assert any("焦虑" in name for name in scale_names)
    
    def test_get_counselor_state(self, counselor_bot):
        """测试获取咨询师状态"""
        state = counselor_bot.get_counselor_state()
        
        assert "current_state" in state
        assert "identified_issues" in state
        assert "therapy_approach" in state
        assert state["current_state"] == "introduction"


class TestFlowControlAgent:
    """测试流程控制Agent"""
    
    @pytest.fixture
    def flow_agent(self):
        """创建流程控制Agent"""
        agent = FlowControlAgent()
        agent.llm_client = AsyncMock()
        agent.model = "test-model"
        return agent
    
    @pytest.fixture
    def sample_conversation(self):
        """示例对话"""
        return [
            ConversationMessage(role="student", content="我需要帮助", emotion=EmotionState.ANXIOUS, round_number=1),
            ConversationMessage(role="counselor", content="我能理解", state="introduction", round_number=1),
            ConversationMessage(role="student", content="谢谢", emotion=EmotionState.CALM, round_number=2),
        ]
    
    def test_format_conversation_history(self, flow_agent, sample_conversation):
        """测试格式化对话历史"""
        formatted = flow_agent._format_conversation_history(sample_conversation)
        
        assert isinstance(formatted, str)
        assert "学生" in formatted
        assert "咨询师" in formatted
        assert "我需要帮助" in formatted
        assert "anxious" in formatted
    
    def test_should_terminate_session(self, flow_agent):
        """测试会话终止判断"""
        # 低风险不应该终止
        low_risk = {"overall_risk_level": 1, "emergency_required": False}
        assert flow_agent.should_terminate_session(low_risk) is False
        
        # 高风险应该终止
        high_risk = {"overall_risk_level": 5, "emergency_required": True}
        assert flow_agent.should_terminate_session(high_risk) is True
    
    def test_get_state_transition_recommendation(self, flow_agent):
        """测试状态转换建议"""
        # 不需要转换
        no_transition = {
            "state_transition": {
                "need_transition": False
            }
        }
        assert flow_agent.get_state_transition_recommendation(no_transition) is None
        
        # 需要转换
        need_transition = {
            "state_transition": {
                "need_transition": True,
                "recommended_state": "exploration"
            }
        }
        result = flow_agent.get_state_transition_recommendation(need_transition)
        assert result == CounselorState.EXPLORATION
    
    def test_rule_based_risk_assessment(self, flow_agent, sample_conversation):
        """测试基于规则的风险评估"""
        risk = flow_agent._rule_based_risk_assessment(sample_conversation)
        
        assert isinstance(risk, type(flow_agent.assess_risk("")))
        assert hasattr(risk, 'overall_risk')
        assert hasattr(risk, 'suicide_risk')


class TestQualityAssessmentAgent:
    """测试质量评估Agent"""
    
    @pytest.fixture
    def quality_agent(self):
        """创建质量评估Agent"""
        agent = QualityAssessmentAgent()
        agent.llm_client = AsyncMock()
        agent.model = "test-model"
        return agent
    
    @pytest.fixture
    def sample_background_info(self):
        """示例背景信息"""
        student_info = StudentBackground(
            name="测试学生",
            age=20,
            gender="男",
            grade="大二",
            major="计算机科学",
            family_background="测试背景",
            personality_traits=["内向"],
            psychological_profile="测试侧写",
            hidden_personal_info="测试信息",
            current_psychological_issue=PsychologicalIssue.ACADEMIC_ANXIETY,
            symptom_description="学业焦虑"
        )
        
        counselor_info = CounselorBackground(
            name="测试咨询师",
            therapy_approach=TherapyApproach.CBT,
            communication_style="认知行为风格",
            experience_years=5,
            specialization=["焦虑"]
        )
        
        return BackgroundInfo(
            student_info=student_info,
            counselor_info=counselor_info
        )
    
    def test_format_background_info(self, quality_agent, sample_background_info):
        """测试格式化背景信息"""
        formatted = quality_agent._format_background_info(sample_background_info)
        
        assert isinstance(formatted, str)
        assert "测试学生" in formatted
        assert "测试咨询师" in formatted
        assert "学业焦虑" in formatted
        assert "认知行为疗法" in formatted
    
    def test_analyze_counseling_trajectory(self, quality_agent):
        """测试分析咨询轨迹"""
        sample_history = [
            ConversationMessage(role="counselor", content="欢迎", state="introduction", round_number=1),
            ConversationMessage(role="student", content="谢谢", round_number=1),
            ConversationMessage(role="counselor", content="探索", state="exploration", round_number=2),
        ]
        
        trajectory = quality_agent._analyze_counseling_trajectory(sample_history)
        
        assert "state_transitions" in trajectory
        assert "rounds_per_state" in trajectory
        assert "total_rounds" in trajectory
        assert trajectory["total_rounds"] == 3
    
    def test_get_rating_label(self, quality_agent):
        """测试评分标签转换"""
        assert quality_agent._get_rating_label(9.5) == "优秀"
        assert quality_agent._get_rating_label(7.5) == "良好"
        assert quality_agent._get_rating_label(5.5) == "一般"
        assert quality_agent._get_rating_label(3.0) == "需要改进"
    
    def test_calculate_quantitative_metrics(self, quality_agent):
        """测试量化指标计算"""
        conversation_history = [
            ConversationMessage(role="student", content="短消息", emotion=EmotionState.ANXIOUS, round_number=1),
            ConversationMessage(role="counselor", content="这是一个较长的咨询师回复？", round_number=1),
            ConversationMessage(role="student", content="另一个学生消息", emotion=EmotionState.CALM, round_number=2),
        ]
        
        metrics = quality_agent._calculate_quantitative_metrics(conversation_history, {})
        
        assert "total_messages" in metrics
        assert "student_message_count" in metrics
        assert "counselor_message_count" in metrics
        assert "emotional_range" in metrics
        assert "counselor_question_ratio" in metrics
        
        assert metrics["total_messages"] == 3
        assert metrics["student_message_count"] == 2
        assert metrics["counselor_message_count"] == 1
        assert metrics["emotional_range"] == 2  # ANXIOUS 和 CALM
        assert metrics["counselor_question_ratio"] == 1.0  # 1个问句，1条咨询师消息


class TestIntegrationScenarios:
    """集成测试场景"""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow_mock(self):
        """测试完整对话流程（模拟）"""
        # 这是一个集成测试的框架，实际运行需要真实的LLM
        
        # 1. 创建背景生成器
        bg_agent = BackgroundGenerationAgent()
        bg_agent.llm_client = AsyncMock()
        bg_agent.model = "test-model"
        
        # 2. 模拟背景生成
        mock_background_response = {
            "student_info": {
                "name": "张小明",
                "age": 20,
                "gender": "男",
                "grade": "大二",
                "major": "计算机科学",
                "family_background": "普通家庭",
                "personality_traits": ["内向", "完美主义"],
                "psychological_profile": "学习压力大",
                "hidden_personal_info": "父母期望很高",
                "current_psychological_issue": "academic_anxiety",
                "symptom_description": "考试焦虑严重"
            },
            "counselor_info": {
                "name": "李老师",
                "therapy_approach": "cognitive_behavioral_therapy",
                "communication_style": "温和专业",
                "experience_years": 8,
                "specialization": ["焦虑", "学习压力"]
            },
            "initial_question": "老师，我最近考试总是很紧张，怎么办？"
        }
        
        with patch.object(bg_agent, '_safe_json_parse', return_value=mock_background_response):
            background, initial_question = await bg_agent.generate_background()
            
            # 验证背景生成
            assert background.student_info.name == "张小明"
            assert initial_question == "老师，我最近考试总是很紧张，怎么办？"
            
            # 3. 创建Bot实例
            student_bot = StudentBot(student_background=background.student_info)
            counselor_bot = CounselorBot(counselor_background=background.counselor_info)
            flow_agent = FlowControlAgent(
                student_background=background.student_info,
                counselor_background=background.counselor_info
            )
            
            # 配置模拟LLM
            for bot in [student_bot, counselor_bot, flow_agent]:
                bot.llm_client = AsyncMock()
                bot.model = "test-model"
            
            # 4. 验证初始状态
            assert student_bot.current_emotion == EmotionState.ANXIOUS
            assert counselor_bot.current_state == CounselorState.INTRODUCTION
            assert student_bot.trust_level < 0.3  # 初始信任度低
            
            # 5. 验证状态管理
            counselor_bot.transition_to_state(CounselorState.EXPLORATION, "建立信任")
            assert counselor_bot.current_state == CounselorState.EXPLORATION
            assert len(counselor_bot.session_notes) > 0
    
    def test_risk_assessment_integration(self):
        """测试风险评估集成"""
        # 创建包含风险内容的对话
        risky_conversation = [
            ConversationMessage(role="student", content="我觉得活着没意思", round_number=1),
            ConversationMessage(role="counselor", content="能详细说说吗？", round_number=1),
            ConversationMessage(role="student", content="我想要结束这一切", round_number=2),
        ]
        
        # 测试学生Bot的风险评估
        student_bot = StudentBot()
        risk = student_bot.assess_risk("我想要结束这一切，活着没意思")
        
        assert risk.overall_risk >= 3
        assert risk.suicide_risk >= 3
        assert risk.emergency_required is True
        
        # 测试流程控制Agent的风险评估
        flow_agent = FlowControlAgent()
        flow_agent.llm_client = AsyncMock()
        
        rule_risk = flow_agent._rule_based_risk_assessment(risky_conversation)
        assert rule_risk.overall_risk >= 3
    
    def test_emotion_state_transitions(self):
        """测试情绪状态转换"""
        from constants import EMOTION_TRANSITIONS
        
        # 验证情绪转换映射的完整性
        for emotion in EmotionState:
            assert emotion in EMOTION_TRANSITIONS
            transitions = EMOTION_TRANSITIONS[emotion]
            assert len(transitions) > 0
            assert all(isinstance(t, EmotionState) for t in transitions)
        
        # 测试特定的情绪转换逻辑
        assert EmotionState.CALM in EMOTION_TRANSITIONS[EmotionState.ANXIOUS]
        assert EmotionState.TRUSTING in EMOTION_TRANSITIONS[EmotionState.RESISTANT]
        assert EmotionState.HOPEFUL in EMOTION_TRANSITIONS[EmotionState.DEPRESSED]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])