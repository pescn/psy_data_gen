"""
数据模型测试
测试所有Pydantic模型的验证、序列化等功能
"""

import pytest
from pydantic import ValidationError
from models import (
    ConversationMessage, StudentBackground, CounselorBackground,
    BackgroundInfo, QualityAssessment, CounselingTrajectory,
    StateTransition, GenerationResult, RiskAssessment,
    PsychologicalIssue, TherapyApproach, CounselorState, EmotionState
)


class TestConversationMessage:
    """测试对话消息模型"""
    
    def test_valid_conversation_message(self):
        """测试有效的对话消息创建"""
        msg = ConversationMessage(
            role="student",
            content="我最近压力很大",
            state=None,
            emotion=EmotionState.ANXIOUS,
            round_number=1
        )
        
        assert msg.role == "student"
        assert msg.content == "我最近压力很大"
        assert msg.emotion == EmotionState.ANXIOUS
        assert msg.round_number == 1
    
    def test_conversation_message_serialization(self):
        """测试对话消息序列化"""
        msg = ConversationMessage(
            role="counselor",
            content="我能理解你的感受",
            state="introduction",
            emotion=None,
            round_number=2
        )
        
        msg_dict = msg.dict()
        assert msg_dict['role'] == "counselor"
        assert msg_dict['state'] == "introduction"
        assert msg_dict['emotion'] is None
    
    def test_invalid_round_number(self):
        """测试无效的轮数（应该为正数）"""
        with pytest.raises(ValidationError):
            ConversationMessage(
                role="student",
                content="测试",
                round_number=0  # 无效：应该>=1
            )


class TestStudentBackground:
    """测试学生背景模型"""
    
    @pytest.fixture
    def valid_student_data(self):
        """有效的学生数据"""
        return {
            "name": "张小明",
            "age": 20,
            "gender": "男",
            "grade": "大二",
            "major": "计算机科学与技术",
            "family_background": "来自普通工薪家庭，父母都是公务员，家庭关系和睦",
            "personality_traits": ["内向", "敏感", "完美主义"],
            "psychological_profile": "性格较为内向，对自己要求严格，容易因为学业压力而焦虑",
            "hidden_personal_info": "从小父母对其期望很高，导致其内心承受巨大压力",
            "current_psychological_issue": PsychologicalIssue.ACADEMIC_ANXIETY,
            "symptom_description": "最近总是担心考试成绩，晚上经常失眠"
        }
    
    def test_valid_student_background(self, valid_student_data):
        """测试有效的学生背景创建"""
        student = StudentBackground(**valid_student_data)
        
        assert student.name == "张小明"
        assert student.age == 20
        assert student.current_psychological_issue == PsychologicalIssue.ACADEMIC_ANXIETY
        assert len(student.personality_traits) == 3
    
    def test_invalid_age(self, valid_student_data):
        """测试无效年龄"""
        valid_student_data["age"] = 15  # 太小
        with pytest.raises(ValidationError):
            StudentBackground(**valid_student_data)
        
        valid_student_data["age"] = 30  # 太大
        with pytest.raises(ValidationError):
            StudentBackground(**valid_student_data)
    
    def test_empty_personality_traits(self, valid_student_data):
        """测试空的性格特征列表"""
        valid_student_data["personality_traits"] = []
        student = StudentBackground(**valid_student_data)
        assert student.personality_traits == []


class TestCounselorBackground:
    """测试咨询师背景模型"""
    
    @pytest.fixture
    def valid_counselor_data(self):
        """有效的咨询师数据"""
        return {
            "name": "李心语",
            "therapy_approach": TherapyApproach.CBT,
            "communication_style": "温和耐心，善于倾听，注重与来访者建立信任关系",
            "experience_years": 8,
            "specialization": ["焦虑障碍", "抑郁症", "学习压力"]
        }
    
    def test_valid_counselor_background(self, valid_counselor_data):
        """测试有效的咨询师背景创建"""
        counselor = CounselorBackground(**valid_counselor_data)
        
        assert counselor.name == "李心语"
        assert counselor.therapy_approach == TherapyApproach.CBT
        assert counselor.experience_years == 8
        assert len(counselor.specialization) == 3
    
    def test_invalid_experience_years(self, valid_counselor_data):
        """测试无效的从业年限"""
        valid_counselor_data["experience_years"] = -1
        with pytest.raises(ValidationError):
            CounselorBackground(**valid_counselor_data)


class TestBackgroundInfo:
    """测试背景信息汇总模型"""
    
    @pytest.fixture
    def sample_background_info(self):
        """示例背景信息"""
        student_info = StudentBackground(
            name="测试学生",
            age=19,
            gender="女",
            grade="大一",
            major="心理学",
            family_background="测试背景",
            personality_traits=["外向"],
            psychological_profile="测试侧写",
            hidden_personal_info="测试信息",
            current_psychological_issue=PsychologicalIssue.SOCIAL_PHOBIA,
            symptom_description="社交恐惧症状"
        )
        
        counselor_info = CounselorBackground(
            name="测试咨询师",
            therapy_approach=TherapyApproach.HUMANISTIC,
            communication_style="人本主义风格",
            experience_years=5,
            specialization=["社交恐惧"]
        )
        
        return BackgroundInfo(
            student_info=student_info,
            counselor_info=counselor_info
        )
    
    def test_background_info_creation(self, sample_background_info):
        """测试背景信息创建"""
        assert sample_background_info.student_info.name == "测试学生"
        assert sample_background_info.counselor_info.therapy_approach == TherapyApproach.HUMANISTIC
        assert isinstance(sample_background_info.generation_params, dict)


class TestStateTransition:
    """测试状态转换模型"""
    
    def test_valid_state_transition(self):
        """测试有效的状态转换"""
        transition = StateTransition(
            from_state=CounselorState.INTRODUCTION,
            to_state=CounselorState.EXPLORATION,
            transition_round=5,
            reason="学生开始信任，愿意分享更多信息"
        )
        
        assert transition.from_state == CounselorState.INTRODUCTION
        assert transition.to_state == CounselorState.EXPLORATION
        assert transition.transition_round == 5
        assert "信任" in transition.reason
    
    def test_invalid_transition_round(self):
        """测试无效的转换轮数"""
        with pytest.raises(ValidationError):
            StateTransition(
                from_state=CounselorState.INTRODUCTION,
                to_state=CounselorState.EXPLORATION,
                transition_round=0,  # 应该>=1
                reason="测试"
            )


class TestCounselingTrajectory:
    """测试咨询轨迹模型"""
    
    def test_counseling_trajectory(self):
        """测试咨询轨迹创建"""
        transitions = [
            StateTransition(
                from_state=CounselorState.INTRODUCTION,
                to_state=CounselorState.EXPLORATION, 
                transition_round=5,
                reason="建立信任"
            ),
            StateTransition(
                from_state=CounselorState.EXPLORATION,
                to_state=CounselorState.ASSESSMENT,
                transition_round=15,
                reason="信息收集充分"
            )
        ]
        
        trajectory = CounselingTrajectory(
            state_transitions=transitions,
            rounds_per_state={"introduction": 5, "exploration": 10, "assessment": 3},
            total_rounds=18
        )
        
        assert len(trajectory.state_transitions) == 2
        assert trajectory.total_rounds == 18
        assert trajectory.rounds_per_state["exploration"] == 10


class TestQualityAssessment:
    """测试质量评估模型"""
    
    def test_quality_assessment_scores(self):
        """测试质量评估评分范围"""
        # 测试有效评分
        assessment = QualityAssessment(
            core_issue="学业焦虑",
            key_transitions=["建立信任", "深入探索"],
            counseling_trajectory=CounselingTrajectory(
                state_transitions=[],
                rounds_per_state={},
                total_rounds=20
            ),
            counseling_techniques_score=8.5,
            final_result="咨询成功",
            issue_consistency=True,
            overall_quality_score=8.2,
            improvement_suggestions=["继续保持共情"]
        )
        
        assert 0 <= assessment.counseling_techniques_score <= 10
        assert 0 <= assessment.overall_quality_score <= 10
        assert assessment.issue_consistency is True
    
    def test_invalid_scores(self):
        """测试无效评分"""
        with pytest.raises(ValidationError):
            QualityAssessment(
                core_issue="测试",
                key_transitions=[],
                counseling_trajectory=CounselingTrajectory(
                    state_transitions=[],
                    rounds_per_state={},
                    total_rounds=1
                ),
                counseling_techniques_score=11.0,  # 超出范围
                final_result="测试",
                issue_consistency=True,
                overall_quality_score=5.0
            )


class TestRiskAssessment:
    """测试风险评估模型"""
    
    def test_valid_risk_assessment(self):
        """测试有效的风险评估"""
        risk = RiskAssessment(
            suicide_risk=2,
            self_harm_risk=1,
            harm_others_risk=0,
            overall_risk=2,
            risk_indicators=["压力大", "睡不着"],
            emergency_required=False
        )
        
        assert risk.suicide_risk == 2
        assert risk.overall_risk == 2
        assert len(risk.risk_indicators) == 2
        assert risk.emergency_required is False
    
    def test_risk_level_validation(self):
        """测试风险等级验证"""
        # 测试超出范围的风险等级
        with pytest.raises(ValidationError):
            RiskAssessment(
                suicide_risk=6,  # 超出0-5范围
                self_harm_risk=1,
                harm_others_risk=0,
                overall_risk=2
            )
    
    def test_emergency_required_logic(self):
        """测试紧急干预逻辑"""
        # 高风险应该需要紧急干预
        risk = RiskAssessment(
            suicide_risk=5,
            self_harm_risk=4,
            harm_others_risk=3,
            overall_risk=5,
            emergency_required=True
        )
        
        assert risk.emergency_required is True
        assert risk.overall_risk == 5


class TestEnumModels:
    """测试枚举模型"""
    
    def test_psychological_issue_enum(self):
        """测试心理问题枚举"""
        assert PsychologicalIssue.ACADEMIC_ANXIETY.value == "academic_anxiety"
        assert PsychologicalIssue.SOCIAL_PHOBIA.value == "social_phobia"
        
        # 测试枚举完整性
        expected_issues = [
            "academic_anxiety", "social_phobia", "depression", 
            "procrastination", "ocd_symptoms", "adaptation_issues",
            "relationship_issues", "family_conflicts", "identity_confusion", 
            "sleep_problems"
        ]
        
        actual_issues = [issue.value for issue in PsychologicalIssue]
        assert set(actual_issues) == set(expected_issues)
    
    def test_therapy_approach_enum(self):
        """测试咨询流派枚举"""
        assert TherapyApproach.CBT.value == "cognitive_behavioral_therapy"
        assert TherapyApproach.HUMANISTIC.value == "humanistic_therapy"
        
        # 测试所有流派
        expected_approaches = [
            "cognitive_behavioral_therapy", "humanistic_therapy", 
            "psychoanalytic", "solution_focused", "mindfulness_therapy"
        ]
        
        actual_approaches = [approach.value for approach in TherapyApproach]
        assert set(actual_approaches) == set(expected_approaches)
    
    def test_counselor_state_enum(self):
        """测试咨询师状态枚举"""
        assert CounselorState.INTRODUCTION.value == "introduction"
        assert CounselorState.EXPLORATION.value == "exploration"
        assert CounselorState.ASSESSMENT.value == "assessment"
        assert CounselorState.SCALE_RECOMMENDATION.value == "scale_recommendation"
    
    def test_emotion_state_enum(self):
        """测试情绪状态枚举"""
        # 测试基本情绪
        assert EmotionState.ANXIOUS.value == "anxious"
        assert EmotionState.CALM.value == "calm"
        assert EmotionState.OTHER.value == "other"  # 测试新增的OTHER状态
        
        # 确保包含所有预期的情绪状态
        expected_emotions = [
            "anxious", "depressed", "confused", "angry", "calm", 
            "hopeful", "resistant", "trusting", "avoidant", "open", "other"
        ]
        
        actual_emotions = [emotion.value for emotion in EmotionState]
        assert set(actual_emotions) == set(expected_emotions)


class TestModelIntegration:
    """测试模型集成"""
    
    def test_generation_result_complete(self):
        """测试完整的生成结果模型"""
        # 创建完整的数据结构
        student_info = StudentBackground(
            name="集成测试学生",
            age=21,
            gender="男",
            grade="大三",
            major="软件工程",
            family_background="集成测试背景",
            personality_traits=["理性", "内向"],
            psychological_profile="集成测试侧写",
            hidden_personal_info="集成测试深层信息",
            current_psychological_issue=PsychologicalIssue.PROCRASTINATION,
            symptom_description="拖延症状描述"
        )
        
        counselor_info = CounselorBackground(
            name="集成测试咨询师",
            therapy_approach=TherapyApproach.SOLUTION_FOCUSED,
            communication_style="解决焦点风格",
            experience_years=10,
            specialization=["拖延症", "时间管理"]
        )
        
        background_info = BackgroundInfo(
            student_info=student_info,
            counselor_info=counselor_info
        )
        
        trajectory = CounselingTrajectory(
            state_transitions=[],
            rounds_per_state={"introduction": 4, "exploration": 8},
            total_rounds=12
        )
        
        assessment = QualityAssessment(
            core_issue="拖延症",
            key_transitions=["建立关系", "识别模式"],
            counseling_trajectory=trajectory,
            counseling_techniques_score=7.5,
            final_result="初步改善",
            issue_consistency=True,
            overall_quality_score=7.8
        )
        
        result = GenerationResult(
            session_id="test_session_123",
            background=background_info,
            conversation={"history": [], "total_rounds": 12},
            assessment=assessment,
            generation_metadata={"test": True}
        )
        
        # 验证完整结构
        assert result.session_id == "test_session_123"
        assert result.background.student_info.name == "集成测试学生"
        assert result.assessment.overall_quality_score == 7.8
        assert result.generation_metadata["test"] is True
    
    def test_model_serialization_deserialization(self):
        """测试模型的序列化和反序列化"""
        # 创建消息
        original_msg = ConversationMessage(
            role="student",
            content="我需要帮助",
            emotion=EmotionState.CONFUSED,
            round_number=3
        )
        
        # 序列化
        msg_dict = original_msg.dict()
        
        # 反序列化
        restored_msg = ConversationMessage(**msg_dict)
        
        # 验证
        assert restored_msg.role == original_msg.role
        assert restored_msg.content == original_msg.content
        assert restored_msg.emotion == original_msg.emotion
        assert restored_msg.round_number == original_msg.round_number


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
