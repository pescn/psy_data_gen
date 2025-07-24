"""
测试包初始化文件
包含测试的共用工具和配置
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# 确保测试目录存在
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# 测试用的常量
MOCK_SESSION_ID = "test_session_12345"
MOCK_API_KEY = "test-api-key-mock"
MOCK_MODEL_NAME = "test-gpt-model"

# 共用的Mock数据
MOCK_STUDENT_DATA = {
    "name": "测试学生",
    "age": 20,
    "gender": "男",
    "grade": "大二",
    "major": "计算机科学与技术",
    "family_background": "来自普通工薪家庭，父母都是教师",
    "personality_traits": ["内向", "认真", "有责任心"],
    "psychological_profile": "性格内向但踏实，学习认真但容易给自己压力",
    "hidden_personal_info": "从小被寄予厚望，内心承受较大期待压力",
    "current_psychological_issue": "academic_anxiety",
    "symptom_description": "临近考试时会出现失眠、心慌、注意力难以集中等症状"
}

MOCK_COUNSELOR_DATA = {
    "name": "李心理老师",
    "therapy_approach": "cognitive_behavioral_therapy",
    "communication_style": "温和耐心，善于倾听，注重实用性指导",
    "experience_years": 8,
    "specialization": ["学业焦虑", "考试焦虑", "学习压力管理"]
}

MOCK_CONVERSATION_HISTORY = [
    {
        "role": "student",
        "content": "老师，我最近总是很紧张，特别是快考试的时候",
        "emotion": "anxious",
        "round_number": 1
    },
    {
        "role": "counselor", 
        "content": "我能理解你的感受，考试前紧张是很正常的。能具体说说你的紧张都体现在哪些方面吗？",
        "state": "introduction",
        "round_number": 1
    },
    {
        "role": "student",
        "content": "就是晚上睡不着觉，白天上课也无法集中注意力，总是想着考试的事情",
        "emotion": "anxious", 
        "round_number": 2
    },
    {
        "role": "counselor",
        "content": "听起来你的焦虑主要影响了睡眠和注意力。这种情况持续多长时间了？",
        "state": "exploration",
        "round_number": 2
    }
]

# 测试工具函数
def create_mock_llm_client():
    """创建模拟的LLM客户端"""
    from unittest.mock import AsyncMock, Mock
    
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "这是模拟的LLM响应"
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

def create_test_background_info():
    """创建测试用的背景信息"""
    from models import BackgroundInfo, StudentBackground, CounselorBackground
    from models import PsychologicalIssue, TherapyApproach
    
    student_info = StudentBackground(**MOCK_STUDENT_DATA)
    student_info.current_psychological_issue = PsychologicalIssue.ACADEMIC_ANXIETY
    
    counselor_info = CounselorBackground(**MOCK_COUNSELOR_DATA)
    counselor_info.therapy_approach = TherapyApproach.CBT
    
    return BackgroundInfo(
        student_info=student_info,
        counselor_info=counselor_info
    )

def create_test_conversation_history():
    """创建测试用的对话历史"""
    from models import ConversationMessage, EmotionState
    
    messages = []
    for msg_data in MOCK_CONVERSATION_HISTORY:
        msg = ConversationMessage(
            role=msg_data["role"],
            content=msg_data["content"],
            round_number=msg_data["round_number"]
        )
        
        if "emotion" in msg_data:
            msg.emotion = EmotionState(msg_data["emotion"])
        if "state" in msg_data:
            msg.state = msg_data["state"]
            
        messages.append(msg)
    
    return messages

# 测试装饰器
def skip_if_no_llm_client(func):
    """如果没有真实的LLM客户端则跳过测试"""
    import pytest
    import os
    
    def wrapper(*args, **kwargs):
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("TEST_WITH_REAL_LLM"):
            pytest.skip("Skipping test that requires real LLM client")
        return func(*args, **kwargs)
    
    return wrapper

def pytest_configure(config):
    """Pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring real LLM client")

def pytest_runtest_setup(item):
    """测试运行前的设置"""
    # 清理测试输出目录
    import shutil
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# 异常处理测试工具
class MockLLMError(Exception):
    """模拟LLM API错误"""
    pass

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_mock_json_response(data_type="background"):
        """生成模拟的JSON响应"""
        if data_type == "background":
            return {
                "student_info": MOCK_STUDENT_DATA,
                "counselor_info": MOCK_COUNSELOR_DATA,
                "initial_question": "老师，我最近学习压力很大，经常失眠",
                "generation_params": {"mode": "random", "complexity_level": "适中"}
            }
        elif data_type == "flow_control":
            return {
                "round_analysis": {
                    "current_round": 3,
                    "student_trust_level": "适中",
                    "student_openness": "较高",
                    "information_saturation": "基本充分",
                    "counselor_effectiveness": "良好"
                },
                "state_transition": {
                    "need_transition": True,
                    "current_state": "introduction",
                    "recommended_state": "exploration",
                    "transition_reason": "学生开始信任，可以深入探索",
                    "confidence_level": "高"
                },
                "risk_assessment": {
                    "overall_risk_level": 1,
                    "suicide_risk": 0,
                    "self_harm_risk": 0,
                    "harm_others_risk": 0,
                    "risk_indicators": [],
                    "emergency_required": False,
                    "risk_description": "无明显风险"
                },
                "improvement_suggestions": [
                    "继续保持共情态度",
                    "可以适当深入探索具体问题"
                ],
                "next_focus": "了解学生的具体学习困难和压力来源"
            }
        elif data_type == "quality_assessment":
            return {
                "core_issue_identification": {
                    "identified_issue": "学业焦虑",
                    "original_issue": "学业焦虑",
                    "accuracy_score": 9,
                    "consistency_check": True,
                    "analysis": "咨询师准确识别了学生的核心问题"
                },
                "counseling_trajectory": {
                    "state_transitions": [
                        {
                            "from_state": "introduction",
                            "to_state": "exploration",
                            "transition_round": 5,
                            "appropriateness": "appropriate",
                            "reason": "时机把握恰当"
                        }
                    ],
                    "phase_effectiveness": {
                        "introduction_phase": {
                            "rounds_used": 4,
                            "effectiveness_score": 8,
                            "key_achievements": ["建立信任关系", "了解基本问题"],
                            "missed_opportunities": []
                        },
                        "exploration_phase": {
                            "rounds_used": 8,
                            "effectiveness_score": 7,
                            "information_depth": "moderate",
                            "key_achievements": ["深入了解症状", "探索问题根源"],
                            "missed_opportunities": ["可以更多探索家庭因素"]
                        }
                    }
                },
                "overall_quality": {
                    "total_score": 8.2,
                    "quality_level": "good",
                    "strengths": ["共情能力强", "问题识别准确", "状态转换恰当"],
                    "weaknesses": ["可以更深入探索", "技术运用可以更丰富"],
                    "critical_incidents": ["第5轮建立信任的关键时刻"],
                    "missed_opportunities": ["错失了探索家庭动力的机会"]
                },
                "consistency_check": {
                    "issue_consistency": True,
                    "consistency_analysis": "最终诊断与初始问题设定完全一致",
                    "consistency_score": 10
                }
            }
        else:
            return {"mock_data": True, "type": data_type}

# 性能测试工具
class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
    
    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

# 测试断言辅助函数
def assert_valid_conversation_message(msg):
    """断言对话消息的有效性"""
    from models import ConversationMessage
    
    assert isinstance(msg, ConversationMessage)
    assert msg.role in ["student", "counselor"]
    assert isinstance(msg.content, str)
    assert len(msg.content) > 0
    assert msg.round_number > 0

def assert_valid_background_info(background):
    """断言背景信息的有效性"""
    from models import BackgroundInfo
    
    assert isinstance(background, BackgroundInfo)
    assert background.student_info is not None
    assert background.counselor_info is not None
    assert len(background.student_info.name) > 0
    assert len(background.counselor_info.name) > 0
    assert 18 <= background.student_info.age <= 25
    assert 3 <= background.counselor_info.experience_years <= 15

def assert_valid_quality_assessment(assessment):
    """断言质量评估的有效性"""
    from models import QualityAssessment
    
    assert isinstance(assessment, QualityAssessment)
    assert 0 <= assessment.counseling_techniques_score <= 10
    assert 0 <= assessment.overall_quality_score <= 10
    assert isinstance(assessment.issue_consistency, bool)
    assert len(assessment.core_issue) > 0

# 模拟数据清理
def cleanup_test_data():
    """清理测试数据"""
    import shutil
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

# 注册清理函数
import atexit
atexit.register(cleanup_test_data)