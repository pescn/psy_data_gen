"""
主程序测试
测试StreamLit应用的核心功能和会话管理
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# 由于StreamLit的特殊性，我们主要测试核心逻辑类
from main import ConversationSession
from models import (
    BackgroundInfo, StudentBackground, CounselorBackground,
    ConversationMessage, CounselorState, EmotionState,
    PsychologicalIssue, TherapyApproach
)


class TestConversationSession:
    """测试对话会话管理器"""
    
    @pytest.fixture
    def session(self):
        """创建测试会话"""
        session = ConversationSession("test_session_123")
        
        # 模拟配置LLM客户端
        mock_client = AsyncMock()
        for agent in [session.background_agent, session.student_bot, 
                     session.counselor_bot, session.flow_control_agent, session.quality_agent]:
            agent.llm_client = mock_client
            agent.model = "test-model"
        
        return session
    
    @pytest.fixture
    def sample_background(self):
        """示例背景信息"""
        student_info = StudentBackground(
            name="测试学生",
            age=20,
            gender="男",
            grade="大二",
            major="计算机科学",
            family_background="普通家庭背景",
            personality_traits=["内向", "认真"],
            psychological_profile="学习压力较大",
            hidden_personal_info="父母期望很高",
            current_psychological_issue=PsychologicalIssue.ACADEMIC_ANXIETY,
            symptom_description="考试前总是很紧张"
        )
        
        counselor_info = CounselorBackground(
            name="测试咨询师",
            therapy_approach=TherapyApproach.CBT,
            communication_style="认知行为取向，温和专业",
            experience_years=5,
            specialization=["焦虑", "学习压力"]
        )
        
        return BackgroundInfo(
            student_info=student_info,
            counselor_info=counselor_info
        )
    
    def test_session_initialization(self, session):
        """测试会话初始化"""
        assert session.session_id == "test_session_123"
        assert session.background_info is None
        assert session.initial_question == ""
        assert session.conversation_history == []
        assert session.current_round == 0
        assert session.is_completed is False
        assert session.risk_alert is False
        
        # 验证Agent实例存在
        assert session.background_agent is not None
        assert session.student_bot is not None
        assert session.counselor_bot is not None
        assert session.flow_control_agent is not None
        assert session.quality_agent is not None
    
    def test_configure_llm_clients(self, session):
        """测试LLM客户端配置"""
        mock_client = Mock()
        llm_config = {
            'client': mock_client,
            'model': 'gpt-4'
        }
        
        session.configure_llm_clients(llm_config)
        
        # 验证所有agent都配置了LLM客户端
        for agent in [session.background_agent, session.student_bot,
                     session.counselor_bot, session.flow_control_agent, session.quality_agent]:
            assert agent.llm_client == mock_client
            assert agent.model == 'gpt-4'
    
    @pytest.mark.asyncio
    async def test_generate_background_mock(self, session):
        """测试背景生成（模拟）"""
        # 模拟背景生成Agent的返回
        mock_background = BackgroundInfo(
            student_info=StudentBackground(
                name="模拟学生",
                age=19,
                gender="女",
                grade="大一",
                major="心理学",
                family_background="模拟背景",
                personality_traits=["外向"],
                psychological_profile="模拟侧写",
                hidden_personal_info="模拟信息",
                current_psychological_issue=PsychologicalIssue.SOCIAL_PHOBIA,
                symptom_description="社交恐惧"
            ),
            counselor_info=CounselorBackground(
                name="模拟咨询师",
                therapy_approach=TherapyApproach.HUMANISTIC,
                communication_style="人本主义",
                experience_years=3,
                specialization=["社交"]
            )
        )
        
        mock_initial_question = "老师，我在社交场合总是很紧张"
        
        # 模拟background_agent的generate_background方法
        session.background_agent.generate_background = AsyncMock(
            return_value=(mock_background, mock_initial_question)
        )
        
        # 执行生成
        background, initial_question = await session.generate_background("random")
        
        # 验证结果
        assert background.student_info.name == "模拟学生"
        assert background.counselor_info.therapy_approach == TherapyApproach.HUMANISTIC
        assert initial_question == "老师，我在社交场合总是很紧张"
        
        # 验证会话状态更新
        assert session.background_info == background
        assert session.initial_question == initial_question
    
    @pytest.mark.asyncio
    async def test_start_conversation_mock(self, session, sample_background):
        """测试开始对话（模拟）"""
        # 设置背景信息
        session.background_info = sample_background
        session.initial_question = "老师，我需要帮助"
        
        # 模拟咨询师Bot的回复
        mock_counselor_response = "我能理解你的感受，能详细说说你遇到的困难吗？"
        session.counselor_bot.generate_counselor_response = AsyncMock(
            return_value=mock_counselor_response
        )
        
        # 开始对话
        counselor_response = await session.start_conversation()
        
        # 验证结果
        assert counselor_response == mock_counselor_response
        assert len(session.conversation_history) == 2  # 学生问题 + 咨询师回复
        assert session.current_round == 1
        
        # 验证消息内容
        student_msg = session.conversation_history[0]
        assert student_msg.role == "student"
        assert student_msg.content == "老师，我需要帮助"
        
        counselor_msg = session.conversation_history[1]
        assert counselor_msg.role == "counselor"
        assert counselor_msg.content == mock_counselor_response
    
    @pytest.mark.asyncio
    async def test_continue_conversation_mock(self, session, sample_background):
        """测试继续对话（模拟）"""
        # 设置初始状态
        session.background_info = sample_background
        session.conversation_history = [
            ConversationMessage(role="student", content="初始问题", round_number=1),
            ConversationMessage(role="counselor", content="初始回复", round_number=1)
        ]
        session.current_round = 1
        
        # 模拟各个组件的返回
        mock_student_response = "是的，我确实很焦虑"
        session.student_bot.generate_student_response = AsyncMock(
            return_value=mock_student_response
        )
        
        mock_evaluation = {
            "state_transition": {"need_transition": False},
            "risk_assessment": {"overall_risk_level": 1, "emergency_required": False}
        }
        session.flow_control_agent.evaluate_round = AsyncMock(
            return_value=mock_evaluation
        )
        session.flow_control_agent.should_terminate_session = Mock(return_value=False)
        session.flow_control_agent.get_state_transition_recommendation = Mock(return_value=None)
        
        mock_counselor_response = "我能感受到你的焦虑，这很正常"
        session.counselor_bot.generate_counselor_response = AsyncMock(
            return_value=mock_counselor_response
        )
        
        # 执行继续对话
        student_response = await session.continue_conversation()
        
        