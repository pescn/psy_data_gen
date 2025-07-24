"""
数据模型定义
使用 Pydantic BaseModel 定义所有数据结构
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class CounselorState(str, Enum):
    """咨询师状态枚举"""
    INTRODUCTION = "introduction"  # 引入与建立关系阶段
    EXPLORATION = "exploration"    # 深入探索阶段
    ASSESSMENT = "assessment"      # 评估诊断阶段
    SCALE_RECOMMENDATION = "scale_recommendation"  # 量表推荐阶段


class TherapyApproach(str, Enum):
    """咨询流派枚举"""
    CBT = "cognitive_behavioral_therapy"  # 认知行为疗法
    HUMANISTIC = "humanistic_therapy"     # 人本主义疗法
    PSYCHOANALYTIC = "psychoanalytic"     # 精神分析取向
    SOLUTION_FOCUSED = "solution_focused" # 解决焦点疗法
    MINDFULNESS = "mindfulness_therapy"   # 正念疗法


class PsychologicalIssue(str, Enum):
    """心理问题类型枚举"""
    ACADEMIC_ANXIETY = "academic_anxiety"           # 学业焦虑
    SOCIAL_PHOBIA = "social_phobia"                # 社交恐惧
    DEPRESSION = "depression"                       # 抑郁情绪
    PROCRASTINATION = "procrastination"            # 拖延症
    OCD_SYMPTOMS = "ocd_symptoms"                  # 强迫症状
    ADAPTATION_ISSUES = "adaptation_issues"        # 适应性问题
    RELATIONSHIP_ISSUES = "relationship_issues"    # 恋爱情感问题
    FAMILY_CONFLICTS = "family_conflicts"          # 家庭关系问题
    IDENTITY_CONFUSION = "identity_confusion"      # 自我认同困惑
    SLEEP_PROBLEMS = "sleep_problems"              # 睡眠问题


class EmotionState(str, Enum):
    """情绪状态枚举"""
    ANXIOUS = "anxious"         # 焦虑
    DEPRESSED = "depressed"     # 抑郁
    CONFUSED = "confused"       # 困惑
    ANGRY = "angry"             # 愤怒
    CALM = "calm"               # 平静
    HOPEFUL = "hopeful"         # 充满希望
    RESISTANT = "resistant"     # 抗拒
    TRUSTING = "trusting"       # 信任
    AVOIDANT = "avoidant"       # 回避
    OPEN = "open"               # 开放
    OTHER = "other"             # 其他情绪状态


class ConversationMessage(BaseModel):
    """对话消息模型"""
    role: str = Field(..., description="角色：student 或 counselor")
    content: str = Field(..., description="消息内容")
    state: Optional[str] = Field(None, description="当前状态（仅咨询师有状态）")
    emotion: Optional[EmotionState] = Field(None, description="当前情绪状态")
    round_number: int = Field(..., description="对话轮数")


class StudentBackground(BaseModel):
    """学生背景信息模型"""
    name: str = Field(..., description="学生姓名")
    age: int = Field(..., description="年龄")
    gender: str = Field(..., description="性别")
    grade: str = Field(..., description="年级")
    major: str = Field(..., description="专业")
    family_background: str = Field(..., description="家庭背景")
    personality_traits: List[str] = Field(..., description="性格特征")
    psychological_profile: str = Field(..., description="心理侧写")
    hidden_personal_info: str = Field(..., description="隐形个人画像")
    current_psychological_issue: PsychologicalIssue = Field(..., description="当前心理问题")
    symptom_description: str = Field(..., description="症状描述")


class CounselorBackground(BaseModel):
    """咨询师背景信息模型"""
    name: str = Field(..., description="咨询师姓名")
    therapy_approach: TherapyApproach = Field(..., description="咨询流派")
    communication_style: str = Field(..., description="沟通习惯和风格")
    experience_years: int = Field(..., description="从业年限")
    specialization: List[str] = Field(..., description="专业领域")


class BackgroundInfo(BaseModel):
    """背景信息汇总模型"""
    student_info: StudentBackground
    counselor_info: CounselorBackground
    generation_params: Dict[str, Any] = Field(default_factory=dict, description="生成参数")


class StateTransition(BaseModel):
    """状态转换记录模型"""
    from_state: CounselorState = Field(..., description="转换前状态")
    to_state: CounselorState = Field(..., description="转换后状态")
    transition_round: int = Field(..., description="转换发生的轮数")
    reason: str = Field(..., description="转换原因")


class CounselingTrajectory(BaseModel):
    """咨询轨迹模型"""
    state_transitions: List[StateTransition] = Field(..., description="状态转换记录")
    rounds_per_state: Dict[str, int] = Field(..., description="各状态持续轮数")
    total_rounds: int = Field(..., description="总对话轮数")


class QualityAssessment(BaseModel):
    """质量评估模型"""
    core_issue: str = Field(..., description="识别的核心问题")
    key_transitions: List[str] = Field(..., description="关键转折点")
    counseling_trajectory: CounselingTrajectory = Field(..., description="咨询轨迹")
    counseling_techniques_score: float = Field(..., ge=0, le=10, description="咨询技巧评分(0-10)")
    final_result: str = Field(..., description="最终结果描述")
    issue_consistency: bool = Field(..., description="最终结果是否与初始设置一致")
    overall_quality_score: float = Field(..., ge=0, le=10, description="总体质量评分(0-10)")
    improvement_suggestions: List[str] = Field(default_factory=list, description="改进建议")


class ConversationData(BaseModel):
    """对话数据模型"""
    session_id: str = Field(..., description="会话ID")
    background: BackgroundInfo = Field(..., description="背景信息")
    conversation_history: List[ConversationMessage] = Field(..., description="对话历史")
    current_counselor_state: CounselorState = Field(default=CounselorState.INTRODUCTION, description="当前咨询师状态")
    is_completed: bool = Field(default=False, description="对话是否完成")
    risk_level: int = Field(default=0, ge=0, le=5, description="风险等级(0-5)")
    risk_alert: bool = Field(default=False, description="是否触发风险警报")


class GenerationResult(BaseModel):
    """最终生成结果模型"""
    session_id: str = Field(..., description="会话ID")
    background: BackgroundInfo = Field(..., description="背景信息")
    conversation: Dict[str, Any] = Field(..., description="对话记录")
    assessment: QualityAssessment = Field(..., description="质量评估结果")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="生成元数据")


class RiskAssessment(BaseModel):
    """风险评估模型"""
    suicide_risk: int = Field(default=0, ge=0, le=5, description="自杀风险等级")
    self_harm_risk: int = Field(default=0, ge=0, le=5, description="自残风险等级")
    harm_others_risk: int = Field(default=0, ge=0, le=5, description="伤害他人风险等级")
    overall_risk: int = Field(default=0, ge=0, le=5, description="总体风险等级")
    risk_indicators: List[str] = Field(default_factory=list, description="风险指标")
    emergency_required: bool = Field(default=False, description="是否需要紧急干预")
