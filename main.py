"""
StreamLit主界面
心理咨询对话数据生成系统的Web界面
"""

import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from models import (
    BackgroundInfo, ConversationMessage, ConversationData, 
    GenerationResult, CounselorState, EmotionState
)
from settings import SystemConfig
from constants import PSYCHOLOGICAL_ISSUES_DATA, THERAPY_APPROACHES_DATA
from agents.background_gen import BackgroundGenerationAgent
from agents.student import StudentBot
from agents.counselor import CounselorBot
from agents.flow_control import FlowControlAgent
from agents.quality_assess import QualityAssessmentAgent


# 页面配置
st.set_page_config(
    page_title=SystemConfig.PAGE_TITLE,
    page_icon=SystemConfig.PAGE_ICON,
    layout=SystemConfig.LAYOUT
)

# 确保输出目录存在
SystemConfig.ensure_output_dirs()


class ConversationSession:
    """对话会话管理器"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.background_info: Optional[BackgroundInfo] = None
        self.initial_question: str = ""
        self.conversation_history: List[ConversationMessage] = []
        self.current_round = 0
        self.is_completed = False
        self.risk_alert = False
        
        # Agent实例（LLM配置已在类中定义）
        self.background_agent = BackgroundGenerationAgent()
        self.student_bot = StudentBot()
        self.counselor_bot = CounselorBot()
        self.flow_control_agent = FlowControlAgent()
        self.quality_agent = QualityAssessmentAgent()
    
    async def generate_background(self, mode: str = "random", **kwargs) -> tuple[BackgroundInfo, str]:
        """生成背景信息和首句问题"""
        background, initial_question = await self.background_agent.generate_background(
            mode=mode, **kwargs
        )
        
        self.background_info = background
        self.initial_question = initial_question
        
        # 配置Bot的背景信息
        self.student_bot = StudentBot(student_background=background.student_info)
        self.counselor_bot = CounselorBot(counselor_background=background.counselor_info)
        self.flow_control_agent = FlowControlAgent(
            student_background=background.student_info,
            counselor_background=background.counselor_info
        )
        
        return background, initial_question
    
    async def start_conversation(self) -> str:
        """开始对话，返回咨询师的首轮回复"""
        if not self.background_info or not self.initial_question:
            raise ValueError("请先生成背景信息")
        
        # 添加学生的首句问题
        student_msg = ConversationMessage(
            role="student",
            content=self.initial_question,
            emotion=self.student_bot.current_emotion,
            round_number=1
        )
        self.conversation_history.append(student_msg)
        self.current_round = 1
        
        # 生成咨询师的首轮回复
        counselor_response = await self.counselor_bot.generate_counselor_response(
            self.conversation_history
        )
        
        counselor_msg = ConversationMessage(
            role="counselor",
            content=counselor_response,
            state=self.counselor_bot.current_state,
            round_number=1
        )
        self.conversation_history.append(counselor_msg)
        
        return counselor_response
    
    async def continue_conversation(self) -> Optional[str]:
        """继续对话，返回下一轮学生回复"""
        if self.is_completed or self.risk_alert:
            return None
        
        # 检查是否达到最大轮数
        if self.current_round >= SystemConfig.MAX_CONVERSATION_ROUNDS:
            self.is_completed = True
            return None
        
        # 生成学生回复
        student_response = await self.student_bot.generate_student_response(
            self.conversation_history
        )
        
        self.current_round += 1
        student_msg = ConversationMessage(
            role="student",
            content=student_response,
            emotion=self.student_bot.current_emotion,
            round_number=self.current_round
        )
        self.conversation_history.append(student_msg)
        
        # 流程控制评估
        evaluation = await self.flow_control_agent.evaluate_round(
            self.conversation_history,
            self.counselor_bot.current_state,
            self.current_round
        )
        
        # 检查风险
        if self.flow_control_agent.should_terminate_session(evaluation['risk_assessment']):
            self.risk_alert = True
            return student_response
        
        # 检查状态转换
        new_state = self.flow_control_agent.get_state_transition_recommendation(evaluation)
        if new_state:
            self.counselor_bot.transition_to_state(
                new_state, 
                evaluation['state_transition']['transition_reason']
            )
        
        # 生成咨询师回复
        counselor_response = await self.counselor_bot.generate_counselor_response(
            self.conversation_history
        )
        
        counselor_msg = ConversationMessage(
            role="counselor",
            content=counselor_response,
            state=self.counselor_bot.current_state,
            round_number=self.current_round
        )
        self.conversation_history.append(counselor_msg)
        
        # 检查是否到达结束状态
        if (self.counselor_bot.current_state == CounselorState.SCALE_RECOMMENDATION and 
            self.current_round >= SystemConfig.MIN_CONVERSATION_ROUNDS):
            # 可以选择结束对话
            pass
        
        return student_response
    
    async def finalize_conversation(self) -> GenerationResult:
        """完成对话并生成最终结果"""
        self.is_completed = True
        
        # 进行质量评估
        quality_assessment = await self.quality_agent.assess_conversation_quality(
            self.background_info,
            self.conversation_history
        )
        
        # 构建最终结果
        conversation_data = {
            "history": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "state": msg.state,
                    "emotion": msg.emotion.value if msg.emotion else None,
                    "round_number": msg.round_number
                }
                for msg in self.conversation_history
            ],
            "total_rounds": self.current_round,
            "final_state": self.counselor_bot.current_state.value,
            "risk_alert": self.risk_alert
        }
        
        result = GenerationResult(
            session_id=self.session_id,
            background=self.background_info,
            conversation=conversation_data,
            assessment=quality_assessment,
            generation_metadata={
                "generated_at": datetime.now().isoformat(),
                "total_rounds": self.current_round,
                "risk_alert": self.risk_alert,
                "completion_reason": "risk_alert" if self.risk_alert else "normal_completion"
            }
        )
        
        return result


def initialize_session_state():
    """初始化会话状态"""
    if 'session' not in st.session_state:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.session = ConversationSession(session_id)
    
    if 'generation_stage' not in st.session_state:
        st.session_state.generation_stage = "background"  # 直接从背景生成开始


def render_header():
    """渲染页面头部"""
    st.title("🧠 心理咨询对话数据生成系统")
    st.markdown("---")
    
    # 系统状态显示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("会话ID", st.session_state.session.session_id[-8:])
    
    with col2:
        stage_names = {
            "background": "背景生成",
            "conversation": "对话进行中", 
            "completed": "已完成"
        }
        st.metric("当前阶段", stage_names.get(st.session_state.generation_stage, "未知"))
    
    with col3:
        if st.session_state.session.conversation_history:
            st.metric("对话轮数", len(st.session_state.session.conversation_history))
        else:
            st.metric("对话轮数", 0)
    
    with col4:
        if st.session_state.session.risk_alert:
            st.metric("风险状态", "⚠️ 风险警报", delta="需要关注")
        else:
            st.metric("风险状态", "✅ 正常", delta="安全")
    
    st.markdown("---")


def render_llm_status():
    """渲染LLM配置状态"""
    st.subheader("🔧 LLM 配置状态")
    
    # 检查各个Agent的配置状态
    session = st.session_state.session
    agents_status = {
        "背景生成Agent": session.background_agent,
        "学生Bot": session.student_bot,
        "咨询师Bot": session.counselor_bot,
        "流程控制Agent": session.flow_control_agent,
        "质量评估Agent": session.quality_agent
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Agent配置状态：**")
        for name, agent in agents_status.items():
            if hasattr(agent, 'api_key') and agent.api_key:
                st.success(f"✅ {name}: 已配置")
            else:
                st.error(f"❌ {name}: 未配置API密钥")
    
    with col2:
        st.write("**配置说明：**")
        st.info("""
        各Agent的LLM配置已通过类属性定义：
        - 请在各Agent类中设置 `api_key`、`base_url`、`model`
        - 可通过环境变量管理API密钥
        - 支持不同Agent使用不同的模型
        """)
    
    # 环境变量配置提示
    st.write("**环境变量配置示例：**")
    st.code("""
# .env 文件
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
CLAUDE_API_KEY=your_claude_api_key

# 或在代码中直接设置
class StudentBot(ChatBot):
    api_key = "your-api-key"
    base_url = "https://api.deepseek.com" 
    model = "deepseek-chat"
    """, language="bash")
    
    # 配置验证
    if st.button("验证LLM配置"):
        with st.spinner("验证配置中..."):
            all_configured = True
            for name, agent in agents_status.items():
                if not hasattr(agent, 'api_key') or not agent.api_key:
                    st.error(f"❌ {name} 缺少API密钥配置")
                    all_configured = False
            
            if all_configured:
                st.success("✅ 所有Agent配置完成，可以开始生成背景信息！")
                if st.button("开始背景生成"):
                    st.session_state.generation_stage = "background"
                    st.rerun()
            else:
                st.warning("⚠️ 请先配置所有Agent的API密钥")


def render_background_generation():
    """渲染背景生成界面"""
    st.subheader("📝 背景信息生成")
    
    # 检查LLM配置状态
    session = st.session_state.session
    config_issues = []
    
    for name, agent in [
        ("背景生成Agent", session.background_agent),
        ("学生Bot", session.student_bot),
        ("咨询师Bot", session.counselor_bot),
        ("流程控制Agent", session.flow_control_agent),
        ("质量评估Agent", session.quality_agent)
    ]:
        if not hasattr(agent, 'api_key') or not agent.api_key:
            config_issues.append(name)
    
    if config_issues:
        st.error(f"❌ 以下Agent缺少API密钥配置：{', '.join(config_issues)}")
        st.info("💡 请在各Agent类中设置类属性：api_key, base_url, model")
        return
    
    # 生成模式选择
    mode = st.radio(
        "生成模式",
        ["random", "guided"],
        format_func=lambda x: "随机生成" if x == "random" else "指定生成",
        horizontal=True
    )
    
    # 参数配置
    kwargs = {}
    if mode == "guided":
        col1, col2 = st.columns(2)
        
        with col1:
            issue_options = {
                issue.value: data['name']
                for issue, data in PSYCHOLOGICAL_ISSUES_DATA.items()
            }
            selected_issue = st.selectbox(
                "心理问题类型",
                options=list(issue_options.keys()),
                format_func=lambda x: issue_options[x]
            )
            kwargs['psychological_issue'] = selected_issue
        
        with col2:
            user_background = st.text_area(
                "额外背景描述",
                placeholder="请描述学生的具体情况、背景或特殊要求...",
                height=100
            )
            kwargs['user_background'] = user_background
    
    # 生成按钮
    if st.button("生成背景信息", type="primary"):
        with st.spinner("正在生成背景信息..."):
            try:
                import asyncio
                background, initial_question = asyncio.run(
                    st.session_state.session.generate_background(mode, **kwargs)
                )
                st.success("✅ 背景信息生成成功！")
                st.session_state.generation_stage = "conversation"
                st.rerun()
            except Exception as e:
                st.error(f"❌ 生成失败: {str(e)}")
                st.info("💡 请检查API密钥配置和网络连接")
    
    # 显示已生成的背景信息
    if st.session_state.session.background_info:
        render_background_info()


def render_background_info():
    """渲染背景信息显示"""
    st.subheader("📊 已生成的背景信息")
    
    background = st.session_state.session.background_info
    
    tab1, tab2, tab3 = st.tabs(["学生信息", "咨询师信息", "首句问题"])
    
    with tab1:
        student = background.student_info
        st.write("**基本信息**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("姓名", student.name)
        with col2:
            st.metric("年龄", f"{student.age}岁")
        with col3:
            st.metric("性别", student.gender)
        with col4:
            st.metric("年级", student.grade)
        
        st.write("**专业**")
        st.write(student.major)
        
        st.write("**性格特征**")
        st.write(", ".join(student.personality_traits))
        
        st.write("**家庭背景**")
        st.write(student.family_background)
        
        st.write("**心理侧写**")
        st.write(student.psychological_profile)
        
        issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(student.current_psychological_issue, {}).get('name', '未知')
        st.write("**当前心理问题**")
        st.write(f"{issue_name}: {student.symptom_description}")
        
        with st.expander("🔐 深层个人信息"):
            st.write(student.hidden_personal_info)
    
    with tab2:
        counselor = background.counselor_info
        st.write("**基本信息**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("姓名", counselor.name)
        with col2:
            st.metric("从业年限", f"{counselor.experience_years}年")
        
        approach_name = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {}).get('name', '未知')
        st.write("**咨询流派**")
        st.write(approach_name)
        
        st.write("**沟通风格**")
        st.write(counselor.communication_style)
        
        st.write("**专业领域**")
        st.write(", ".join(counselor.specialization))
    
    with tab3:
        st.write("**学生首句问题 (Q0)**")
        st.info(st.session_state.session.initial_question)
        
        if st.button("开始对话", type="primary"):
            st.session_state.generation_stage = "conversation"
            st.rerun()


def render_conversation():
    """渲染对话界面"""
    st.subheader("💬 对话进行中")
    
    session = st.session_state.session
    
    # 对话控制
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("继续对话", disabled=session.is_completed or session.risk_alert):
            with st.spinner("生成对话中..."):
                try:
                    import asyncio
                    response = asyncio.run(session.continue_conversation())
                    if response:
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ 对话生成失败: {str(e)}")
                    st.info("💡 请检查API配置和网络连接")
    
    with col2:
        if st.button("完成对话", disabled=len(session.conversation_history) < SystemConfig.MIN_CONVERSATION_ROUNDS):
            import asyncio
            try:
                asyncio.run(session.finalize_conversation())
                st.session_state.generation_stage = "completed"
                st.rerun()
            except Exception as e:
                st.error(f"❌ 完成对话失败: {str(e)}")
    
    with col3:
        if st.button("重新开始"):
            st.session_state.clear()
            st.rerun()
    
    # 显示对话历史
    if session.conversation_history:
        st.subheader("📜 对话记录")
        
        for i, msg in enumerate(session.conversation_history):
            with st.container():
                if msg.role == "student":
                    st.markdown(f"**🎓 学生** (轮次 {msg.round_number})")
                    if msg.emotion:
                        st.caption(f"情绪状态: {msg.emotion.value}")
                    st.info(msg.content)
                else:
                    st.markdown(f"**👨‍⚕️ 咨询师** (轮次 {msg.round_number})")
                    if msg.state:
                        st.caption(f"咨询状态: {msg.state}")
                    st.success(msg.content)
                
                st.markdown("---")
    
    # 显示当前状态
    if session.conversation_history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎓 学生状态")
            student_state = session.student_bot.get_student_state()
            st.json(student_state)
        
        with col2:
            st.subheader("👨‍⚕️ 咨询师状态")
            counselor_state = session.counselor_bot.get_counselor_state()
            st.json(counselor_state)


def render_completion():
    """渲染完成界面"""
    st.subheader("✅ 对话已完成")
    
    session = st.session_state.session
    
    if st.button("生成质量评估报告", type="primary"):
        with st.spinner("正在进行质量评估..."):
            try:
                import asyncio
                result = asyncio.run(session.finalize_conversation())
                st.success("✅ 质量评估完成！")
                
                # 显示评估结果摘要
                st.subheader("📈 评估结果摘要")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("总体质量评分", f"{result.assessment.overall_quality_score:.1f}/10")
                
                with col2:
                    st.metric("咨询技巧评分", f"{result.assessment.counseling_techniques_score:.1f}/10")
                
                with col3:
                    consistency_text = "✅ 一致" if result.assessment.issue_consistency else "❌ 不一致"
                    st.metric("问题一致性", consistency_text)
                
                # 详细评估信息
                with st.expander("📋 详细评估报告", expanded=True):
                    st.write("**核心问题识别：**", result.assessment.core_issue)
                    st.write("**关键转折点：**", "、".join(result.assessment.key_transitions))
                    st.write("**最终结果：**", result.assessment.final_result)
                    
                    if result.assessment.improvement_suggestions:
                        st.write("**改进建议：**")
                        for suggestion in result.assessment.improvement_suggestions:
                            st.write(f"• {suggestion}")
                
            except Exception as e:
                st.error(f"❌ 评估失败: {str(e)}")
                st.info("💡 请检查API配置和数据完整性")
    
    # 显示基本统计
    st.subheader("📊 对话统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总轮数", len(session.conversation_history))
    
    with col2:
        student_msgs = [msg for msg in session.conversation_history if msg.role == "student"]
        st.metric("学生消息", len(student_msgs))
    
    with col3:
        counselor_msgs = [msg for msg in session.conversation_history if msg.role == "counselor"]
        st.metric("咨询师消息", len(counselor_msgs))
    
    with col4:
        if session.risk_alert:
            st.metric("状态", "风险终止", delta="⚠️")
        else:
            st.metric("状态", "正常完成", delta="✅")
    
    # 导出选项
    st.subheader("💾 导出数据")
    
    if st.button("导出JSON"):
        # 生成导出数据
        export_data = {
            "session_id": session.session_id,
            "background": session.background_info.dict() if session.background_info else {},
            "conversation": [msg.dict() for msg in session.conversation_history],
            "metadata": {
                "total_rounds": len(session.conversation_history),
                "risk_alert": session.risk_alert,
                "export_time": datetime.now().isoformat()
            }
        }
        
        st.download_button(
            label="下载对话数据",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"conversation_{session.session_id}.json",
            mime="application/json"
        )


def main():
    """主函数"""
    initialize_session_state()
    render_header()
    
    # 根据当前阶段渲染不同界面
    if st.session_state.generation_stage == "background":
        # 先显示LLM配置状态
        render_llm_status()
        st.markdown("---")
        render_background_generation()
    
    elif st.session_state.generation_stage == "conversation":
        render_conversation()
    
    elif st.session_state.generation_stage == "completed":
        render_completion()
    
    # 侧边栏信息
    with st.sidebar:
        st.header("💡 使用说明")
        st.markdown("""
        1. **配置LLM**: 在Agent类中设置API密钥
        2. **生成背景**: 创建学生和咨询师背景
        3. **进行对话**: 模拟心理咨询过程
        4. **质量评估**: 评估对话质量
        5. **导出数据**: 保存生成的数据
        """)
        
        st.header("🔧 配置说明")
        st.markdown("""
        **API密钥配置方式：**
        - 在各Agent类中设置类属性
        - 支持环境变量配置
        - 不同Agent可使用不同模型
        
        **示例：**
        ```python
        class StudentBot(ChatBot):
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
            model = "deepseek-chat"
        ```
        """)
        
        st.header("📋 系统配置")
        st.json({
            "最大轮数": SystemConfig.MAX_CONVERSATION_ROUNDS,
            "最小轮数": SystemConfig.MIN_CONVERSATION_ROUNDS,
            "风险阈值": SystemConfig.RISK_THRESHOLD
        })
        
        if st.button("🔄 重置系统"):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    main()
