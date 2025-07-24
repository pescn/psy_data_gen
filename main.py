"""
StreamLit主界面
心理咨询对话数据生成系统的Web界面
"""

import streamlit as st
import asyncio
import json
import uuid
import time
import threading
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from models import (
    BackgroundInfo,
    ConversationMessage,
    GenerationResult,
    CounselorState,
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
    layout=SystemConfig.LAYOUT,
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

    async def generate_background(
        self, mode: str = "random", **kwargs
    ) -> tuple[BackgroundInfo, str]:
        """生成背景信息和首句问题"""
        background, initial_question = await self.background_agent.generate_background(
            mode=mode, **kwargs
        )

        self.background_info = background
        self.initial_question = initial_question

        # 配置Bot的背景信息
        self.student_bot = StudentBot(student_background=background.student_info)
        self.counselor_bot = CounselorBot(
            counselor_background=background.counselor_info
        )
        self.flow_control_agent = FlowControlAgent(
            student_background=background.student_info,
            counselor_background=background.counselor_info,
        )

        return background, initial_question

    async def start_conversation(self) -> str:
        """开始对话，返回咨询师的首轮回复"""
        if not self.background_info or not self.initial_question:
            raise ValueError("请先生成背景信息")

        # 清空对话历史，确保干净开始
        self.conversation_history = []
        self.current_round = 0

        # 添加学生的首句问题 (Q1) - 学生消息轮次从1开始
        self.current_round = 1
        student_msg = ConversationMessage(
            role="student",
            content=self.initial_question,
            emotion=self.student_bot.current_emotion,
            round_number=self.current_round,
        )
        self.conversation_history.append(student_msg)

        # 生成咨询师的首轮回复 (A1) - 咨询师回复同一轮次
        counselor_response = await self.counselor_bot.generate_counselor_response(
            self.conversation_history
        )

        counselor_msg = ConversationMessage(
            role="counselor",
            content=counselor_response,
            state=self.counselor_bot.current_state,
            round_number=self.current_round,  # 咨询师回复使用同一轮次
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

        # 进入下一轮对话
        self.current_round += 1

        # 生成学生回复 (Q_n)
        student_response = await self.student_bot.generate_student_response(
            self.conversation_history
        )

        student_msg = ConversationMessage(
            role="student",
            content=student_response,
            emotion=self.student_bot.current_emotion,
            round_number=self.current_round,
        )
        self.conversation_history.append(student_msg)

        # 流程控制评估
        evaluation = await self.flow_control_agent.evaluate_round(
            self.conversation_history,
            self.counselor_bot.current_state,
            self.current_round,
        )

        # 检查风险
        if self.flow_control_agent.should_terminate_session(
            evaluation["risk_assessment"]
        ):
            self.risk_alert = True
            return student_response

        # 检查状态转换
        new_state = self.flow_control_agent.get_state_transition_recommendation(
            evaluation
        )
        if new_state:
            self.counselor_bot.transition_to_state(
                new_state, evaluation["state_transition"]["transition_reason"]
            )

        # 生成咨询师回复 (A_n) - 使用同一轮次
        counselor_response = await self.counselor_bot.generate_counselor_response(
            self.conversation_history
        )

        counselor_msg = ConversationMessage(
            role="counselor",
            content=counselor_response,
            state=self.counselor_bot.current_state,
            round_number=self.current_round,  # 咨询师回复使用同一轮次
        )
        self.conversation_history.append(counselor_msg)

        # 检查是否到达结束状态
        if (
            self.counselor_bot.current_state == CounselorState.SCALE_RECOMMENDATION
            and self.current_round >= SystemConfig.MIN_CONVERSATION_ROUNDS
        ):
            # 可以选择结束对话
            pass

        return student_response

    async def finalize_conversation(self) -> GenerationResult:
        """完成对话并生成最终结果"""
        self.is_completed = True

        # 进行质量评估
        quality_assessment = await self.quality_agent.assess_conversation_quality(
            self.background_info, self.conversation_history
        )

        # 构建最终结果
        conversation_data = {
            "history": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "state": msg.state,
                    "emotion": msg.emotion.value if msg.emotion else None,
                    "round_number": msg.round_number,
                }
                for msg in self.conversation_history
            ],
            "total_rounds": self.current_round,
            "final_state": self.counselor_bot.current_state.value,
            "risk_alert": self.risk_alert,
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
                "completion_reason": "risk_alert"
                if self.risk_alert
                else "normal_completion",
            },
        )

        return result


def safe_async_run(coro):
    """安全的异步执行器，避免事件循环冲突"""
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有运行中的循环，使用线程池
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=60)  # 增加超时时间
        else:
            return asyncio.run(coro)
    except RuntimeError:
        # 没有事件循环，直接创建新的
        return asyncio.run(coro)
    except Exception as e:
        st.error(f"异步执行失败: {str(e)}")
        return None


def initialize_session_state():
    """初始化会话状态 - 只执行一次"""
    # 使用锁防止重复初始化
    if "init_lock" not in st.session_state:
        st.session_state.init_lock = threading.Lock()

    with st.session_state.init_lock:
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.generation_stage = "background"
            st.session_state.error_message = None
            st.session_state.is_processing = False

        # 确保session存在且唯一
        if "session" not in st.session_state:
            st.session_state.session = ConversationSession(st.session_state.session_id)


def render_header():
    """渲染页面头部"""
    st.title("🧠 心理咨询对话数据生成系统")
    st.markdown("---")

    # 系统状态显示
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("会话ID", st.session_state.session_id[-8:])

    with col2:
        stage_names = {
            "background": "背景生成",
            "conversation": "对话进行中",
            "completed": "已完成",
        }
        st.metric(
            "当前阶段", stage_names.get(st.session_state.generation_stage, "未知")
        )

    with col3:
        msg_count = len(st.session_state.session.conversation_history)
        round_count = st.session_state.session.current_round

        if st.session_state.is_processing:
            st.metric("当前状态", "🤖 自动运行中...", delta="正在生成")
        else:
            st.metric("当前轮次", f"{round_count}/消息数:{msg_count}")

    with col4:
        if st.session_state.session.risk_alert:
            st.metric("风险状态", "⚠️ 风险警报", delta="需要关注")
        else:
            st.metric("风险状态", "✅ 正常", delta="安全")

    # 背景信息状态指示
    if st.session_state.session.background_info:
        background = st.session_state.session.background_info
        student_name = background.student_info.name
        counselor_name = background.counselor_info.name
        issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(
            background.student_info.current_psychological_issue, {}
        ).get("name", "未知")

        st.info(
            f"💡 当前对话: **{student_name}** 与咨询师 **{counselor_name}** 讨论 **{issue_name}** 问题"
        )

    st.markdown("---")


def render_background_generation():
    """渲染背景生成界面"""
    st.subheader("📝 背景信息生成")

    # 生成模式选择
    mode = st.radio(
        "生成模式",
        ["random", "guided"],
        format_func=lambda x: "随机生成" if x == "random" else "指定生成",
        horizontal=True,
    )

    # 参数配置
    kwargs = {}
    if mode == "guided":
        col1, col2 = st.columns(2)

        with col1:
            issue_options = {
                issue.value: data["name"]
                for issue, data in PSYCHOLOGICAL_ISSUES_DATA.items()
            }
            selected_issue = st.selectbox(
                "心理问题类型",
                options=list(issue_options.keys()),
                format_func=lambda x: issue_options[x],
            )
            kwargs["psychological_issue"] = selected_issue

        with col2:
            user_background = st.text_area(
                "额外背景描述",
                placeholder="请描述学生的具体情况、背景或特殊要求...",
                height=100,
            )
            kwargs["user_background"] = user_background

    # 生成按钮 - 防止重复处理
    generate_btn = st.button(
        "生成背景信息", type="primary", disabled=st.session_state.is_processing
    )

    if generate_btn:
        st.session_state.is_processing = True

        # 创建进度显示区域
        progress_container = st.empty()

        try:
            with progress_container.container():
                with st.spinner("正在生成背景信息..."):
                    # 安全执行异步操作
                    result = safe_async_run(
                        st.session_state.session.generate_background(mode, **kwargs)
                    )

            if result:
                background, initial_question = result
                progress_container.success("✅ 背景信息生成成功！")
                st.session_state.generation_stage = "conversation"
                print(f"生成的背景信息: {background.model_dump()}")
                print(f"首句问题: {initial_question}")

                # 延迟重新运行，避免状态冲突
                time.sleep(0.5)
                st.rerun()
            else:
                progress_container.error("❌ 生成失败，请重试")

        except Exception as e:
            progress_container.error(f"❌ 生成失败: {str(e)}")
        finally:
            st.session_state.is_processing = False

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

        issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(
            student.current_psychological_issue, {}
        ).get("name", "未知")
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

        approach_name = THERAPY_APPROACHES_DATA.get(counselor.therapy_approach, {}).get(
            "name", "未知"
        )
        st.write("**咨询流派**")
        st.write(approach_name)

        st.write("**沟通风格**")
        st.write(counselor.communication_style)

        st.write("**专业领域**")
        st.write(", ".join(counselor.specialization))

    with tab3:
        st.write("**学生首句问题 (Q1)**")
        st.info(st.session_state.session.initial_question)

        # 提供两种开始方式
        col1, col2 = st.columns(2)

        with col1:
            if st.button("开始对话", type="primary"):
                # 清理可能存在的旧对话记录 - 确保完全重置
                st.session_state.session.conversation_history.clear()
                st.session_state.session.current_round = 0
                st.session_state.session.is_completed = False
                st.session_state.session.risk_alert = False

                # 异步开始对话
                progress_container = st.empty()
                st.session_state.is_processing = True

                try:
                    with progress_container.container():
                        with st.spinner("正在开始对话..."):
                            result = safe_async_run(
                                st.session_state.session.start_conversation()
                            )

                    if result:
                        progress_container.success("✅ 对话已开始！")
                        st.session_state.generation_stage = "conversation"
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        progress_container.error("❌ 开始对话失败")

                except Exception as e:
                    progress_container.error(f"❌ 开始对话失败: {str(e)}")
                finally:
                    st.session_state.is_processing = False

        with col2:
            if st.button("🤖 一键自动生成", type="secondary"):
                # 启动完整的自动流程
                st.session_state.session.conversation_history.clear()
                st.session_state.session.current_round = 0
                st.session_state.session.is_completed = False
                st.session_state.session.risk_alert = False

                # 先开始对话
                result = safe_async_run(st.session_state.session.start_conversation())
                if result:
                    st.session_state.generation_stage = "conversation"
                    # 设置自动运行标志
                    st.session_state.auto_run_on_conversation = True
                    st.rerun()
                else:
                    st.error("❌ 启动自动生成失败")


def render_single_message(msg: ConversationMessage):
    """使用 st.chat_message 渲染单条消息"""
    if msg.role == "student":
        with st.chat_message("user", avatar="🎓"):
            st.write(f"**学生 Q{msg.round_number}**")
            if msg.emotion:
                st.caption(f"情绪状态: {msg.emotion.value}")
            st.write(msg.content)
    else:
        with st.chat_message("assistant", avatar="👨‍⚕️"):
            st.write(f"**咨询师 A{msg.round_number}**")
            if msg.state:
                st.caption(f"咨询状态: {msg.state}")
            st.write(msg.content)


def render_conversation():
    """渲染对话界面 - 使用增量渲染避免重复"""
    st.subheader("💬 对话进行中")

    session = st.session_state.session

    # 检查是否需要自动运行
    if (
        hasattr(st.session_state, "auto_run_on_conversation")
        and st.session_state.auto_run_on_conversation
    ):
        # 清除标志
        del st.session_state.auto_run_on_conversation

        # 立即触发自动运行
        st.session_state.is_processing = True
        progress_container = st.empty()

        try:
            with progress_container.container():
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 自动运行对话直到完成
                max_rounds = SystemConfig.MAX_CONVERSATION_ROUNDS

                while (
                    not session.is_completed
                    and not session.risk_alert
                    and session.current_round < max_rounds
                ):
                    status_text.write(
                        f"正在生成第 {session.current_round + 1} 轮对话..."
                    )
                    progress_bar.progress(min(session.current_round / max_rounds, 0.9))

                    # 生成下一轮对话
                    response = safe_async_run(session.continue_conversation())

                    if not response:
                        break

                    # 短暂延迟，让用户看到进度
                    time.sleep(0.3)

                # 完成对话并生成质量评估
                status_text.write("正在生成质量评估报告...")
                progress_bar.progress(0.95)

                result = safe_async_run(session.finalize_conversation())

                if result:
                    # 自动保存对话数据
                    export_data = {
                        "session_id": session.session_id,
                        "background": session.background_info.dict()
                        if session.background_info
                        else {},
                        "conversation": [
                            msg.dict() for msg in session.conversation_history
                        ],
                        "assessment": result.assessment.dict()
                        if result.assessment
                        else {},
                        "metadata": {
                            "total_rounds": session.current_round,
                            "risk_alert": session.risk_alert,
                            "export_time": datetime.now().isoformat(),
                            "auto_generated": True,
                        },
                    }

                    # 保存到session state供下载
                    st.session_state.export_data = json.dumps(
                        export_data, ensure_ascii=False, indent=2
                    )

                    progress_bar.progress(1.0)
                    st.session_state.generation_stage = "completed"

                    # 显示完成信息
                    completion_reason = (
                        "风险警报终止" if session.risk_alert else "正常完成"
                    )
                    progress_container.success(
                        f"✅ 自动生成完成！\n"
                        f"- 总轮数: {session.current_round}\n"
                        f"- 消息数: {len(session.conversation_history)}\n"
                        f"- 完成原因: {completion_reason}\n"
                        f"- 质量评分: {result.assessment.overall_quality_score:.1f}/10"
                    )

                    time.sleep(2)
                    st.rerun()
                else:
                    progress_container.error("❌ 自动生成失败，无法生成质量评估")

        except Exception as e:
            progress_container.error(f"❌ 自动生成失败: {str(e)}")
        finally:
            st.session_state.is_processing = False

        return  # 结束函数，不继续渲染其他内容

    # 在对话界面中显示背景信息摘要
    if session.background_info:
        with st.expander("📋 背景信息摘要", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**👨‍🎓 学生信息**")
                student = session.background_info.student_info
                st.write(f"• **姓名**: {student.name}")
                st.write(f"• **年龄**: {student.age}岁")
                st.write(f"• **专业**: {student.major}")

                issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(
                    student.current_psychological_issue, {}
                ).get("name", "未知")
                st.write(f"• **心理问题**: {issue_name}")
                st.write(
                    f"• **性格特征**: {', '.join(student.personality_traits[:3])}..."
                )

            with col2:
                st.write("**👨‍⚕️ 咨询师信息**")
                counselor = session.background_info.counselor_info
                st.write(f"• **姓名**: {counselor.name}")
                st.write(f"• **经验**: {counselor.experience_years}年")

                approach_name = THERAPY_APPROACHES_DATA.get(
                    counselor.therapy_approach, {}
                ).get("name", "未知")
                st.write(f"• **流派**: {approach_name}")
                st.write(
                    f"• **专业领域**: {', '.join(counselor.specialization[:2])}..."
                )

        st.markdown("---")

    # 对话控制按钮
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        continue_disabled = (
            session.is_completed or session.risk_alert or st.session_state.is_processing
        )

        if st.button("继续对话", disabled=continue_disabled):
            st.session_state.is_processing = True
            progress_container = st.empty()

            try:
                with progress_container.container():
                    with st.spinner("生成对话中..."):
                        response = safe_async_run(session.continue_conversation())

                if response:
                    progress_container.success("✅ 对话生成成功")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    if session.is_completed:
                        progress_container.info("ℹ️ 对话已完成")
                    elif session.risk_alert:
                        progress_container.warning("⚠️ 检测到风险，对话已终止")
                    else:
                        progress_container.error("❌ 对话生成失败")

            except Exception as e:
                progress_container.error(f"❌ 对话生成失败: {str(e)}")
            finally:
                st.session_state.is_processing = False

    with col2:
        auto_run_disabled = (
            session.is_completed or session.risk_alert or st.session_state.is_processing
        )

        if st.button("🤖 自动运行", disabled=auto_run_disabled, type="secondary"):
            st.session_state.is_processing = True
            progress_container = st.empty()

            try:
                with progress_container.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 自动运行对话直到完成
                    round_count = 0
                    max_rounds = SystemConfig.MAX_CONVERSATION_ROUNDS

                    while (
                        not session.is_completed
                        and not session.risk_alert
                        and session.current_round < max_rounds
                    ):
                        round_count += 1
                        status_text.write(
                            f"正在生成第 {session.current_round + 1} 轮对话..."
                        )
                        progress_bar.progress(
                            min(session.current_round / max_rounds, 0.9)
                        )

                        # 生成下一轮对话
                        response = safe_async_run(session.continue_conversation())

                        if not response:
                            break

                        # 短暂延迟，让用户看到进度
                        time.sleep(0.3)

                    # 完成对话并生成质量评估
                    status_text.write("正在生成质量评估报告...")
                    progress_bar.progress(0.95)

                    result = safe_async_run(session.finalize_conversation())

                    if result:
                        # 自动保存对话数据
                        export_data = {
                            "session_id": session.session_id,
                            "background": session.background_info.dict()
                            if session.background_info
                            else {},
                            "conversation": [
                                msg.dict() for msg in session.conversation_history
                            ],
                            "assessment": result.assessment.dict()
                            if result.assessment
                            else {},
                            "metadata": {
                                "total_rounds": session.current_round,
                                "risk_alert": session.risk_alert,
                                "export_time": datetime.now().isoformat(),
                                "auto_generated": True,
                            },
                        }

                        # 保存到session state供下载
                        st.session_state.export_data = json.dumps(
                            export_data, ensure_ascii=False, indent=2
                        )

                        progress_bar.progress(1.0)
                        st.session_state.generation_stage = "completed"

                        # 显示完成信息
                        completion_reason = (
                            "风险警报终止" if session.risk_alert else "正常完成"
                        )
                        progress_container.success(
                            f"✅ 自动运行完成！\n"
                            f"- 总轮数: {session.current_round}\n"
                            f"- 消息数: {len(session.conversation_history)}\n"
                            f"- 完成原因: {completion_reason}\n"
                            f"- 质量评分: {result.assessment.overall_quality_score:.1f}/10"
                        )

                        time.sleep(2)
                        st.rerun()
                    else:
                        progress_container.error("❌ 自动运行失败，无法生成质量评估")

            except Exception as e:
                progress_container.error(f"❌ 自动运行失败: {str(e)}")
            finally:
                st.session_state.is_processing = False

    with col3:
        complete_disabled = (
            len(session.conversation_history) < SystemConfig.MIN_CONVERSATION_ROUNDS
            or st.session_state.is_processing
        )

        if st.button("完成对话", disabled=complete_disabled):
            st.session_state.is_processing = True

            try:
                safe_async_run(session.finalize_conversation())
                st.session_state.generation_stage = "completed"
                st.rerun()
            except Exception as e:
                st.error(f"❌ 完成对话失败: {str(e)}")
            finally:
                st.session_state.is_processing = False

    with col4:
        if st.button("重新开始"):
            # 清理状态并重新初始化
            keys_to_keep = {"init_lock"}  # 保留锁
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()

    # 渲染对话历史 - 确保正确顺序和唯一性
    if session.conversation_history:
        st.subheader("📜 对话记录")

        # 创建聊天容器
        chat_container = st.container()

        with chat_container:
            # 按轮次分组消息，确保学生-咨询师对话顺序
            rounds = {}
            for msg in session.conversation_history:
                round_num = msg.round_number
                if round_num not in rounds:
                    rounds[round_num] = {"student": None, "counselor": None}
                rounds[round_num][msg.role] = msg

            # 按轮次顺序渲染
            for round_num in sorted(rounds.keys()):
                round_msgs = rounds[round_num]

                # 先渲染学生消息
                if round_msgs["student"]:
                    render_single_message(round_msgs["student"])

                # 再渲染咨询师消息
                if round_msgs["counselor"]:
                    render_single_message(round_msgs["counselor"])

    # 显示当前状态
    if session.conversation_history:
        with st.expander("📊 当前状态信息", expanded=False):
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

    # 显示背景信息摘要
    if session.background_info:
        with st.expander("📋 本次对话背景信息", expanded=False):
            background = session.background_info

            col1, col2 = st.columns(2)
            with col1:
                st.write("**👨‍🎓 学生**")
                student = background.student_info
                st.write(f"• {student.name} ({student.age}岁, {student.gender})")
                st.write(f"• {student.grade} - {student.major}")

                issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(
                    student.current_psychological_issue, {}
                ).get("name", "未知")
                st.write(f"• 主要问题: {issue_name}")

            with col2:
                st.write("**👨‍⚕️ 咨询师**")
                counselor = background.counselor_info
                st.write(f"• {counselor.name} ({counselor.experience_years}年经验)")

                approach_name = THERAPY_APPROACHES_DATA.get(
                    counselor.therapy_approach, {}
                ).get("name", "未知")
                st.write(f"• 咨询流派: {approach_name}")
                st.write(f"• 专业领域: {', '.join(counselor.specialization)}")

    # 质量评估按钮
    if st.button(
        "生成质量评估报告", type="primary", disabled=st.session_state.is_processing
    ):
        st.session_state.is_processing = True
        progress_container = st.empty()

        try:
            with progress_container.container():
                with st.spinner("正在进行质量评估..."):
                    result = safe_async_run(session.finalize_conversation())

            if result:
                progress_container.success("✅ 质量评估完成！")

                # 显示评估结果摘要
                st.subheader("📈 评估结果摘要")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "总体质量评分",
                        f"{result.assessment.overall_quality_score:.1f}/10",
                    )

                with col2:
                    st.metric(
                        "咨询技巧评分",
                        f"{result.assessment.counseling_techniques_score:.1f}/10",
                    )

                with col3:
                    consistency_text = (
                        "✅ 一致"
                        if result.assessment.issue_consistency
                        else "❌ 不一致"
                    )
                    st.metric("问题一致性", consistency_text)

                # 详细评估信息
                with st.expander("📋 详细评估报告", expanded=True):
                    st.write("**核心问题识别：**", result.assessment.core_issue)
                    st.write(
                        "**关键转折点：**", "、".join(result.assessment.key_transitions)
                    )
                    st.write("**最终结果：**", result.assessment.final_result)

                    if result.assessment.improvement_suggestions:
                        st.write("**改进建议：**")
                        for suggestion in result.assessment.improvement_suggestions:
                            st.write(f"• {suggestion}")
            else:
                progress_container.error("❌ 评估失败")

        except Exception as e:
            progress_container.error(f"❌ 评估失败: {str(e)}")
        finally:
            st.session_state.is_processing = False

    # 显示基本统计
    st.subheader("📊 对话统计")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总轮数", len(session.conversation_history))

    with col2:
        student_msgs = [
            msg for msg in session.conversation_history if msg.role == "student"
        ]
        st.metric("学生消息", len(student_msgs))

    with col3:
        counselor_msgs = [
            msg for msg in session.conversation_history if msg.role == "counselor"
        ]
        st.metric("咨询师消息", len(counselor_msgs))

    with col4:
        if session.risk_alert:
            st.metric("状态", "风险终止", delta="⚠️")
        else:
            st.metric("状态", "正常完成", delta="✅")

    # 导出选项
    st.subheader("💾 导出数据")

    # 检查是否有自动生成的数据
    if hasattr(st.session_state, "export_data"):
        st.success("✅ 对话数据已准备就绪（自动生成）")

        # 显示数据概览
        try:
            data = json.loads(st.session_state.export_data)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("会话ID", data.get("session_id", "N/A")[-8:])
            with col2:
                st.metric("消息总数", len(data.get("conversation", [])))
            with col3:
                is_auto = data.get("metadata", {}).get("auto_generated", False)
                st.metric("生成方式", "🤖 自动" if is_auto else "👤 手动")

        except json.JSONDecodeError:
            st.warning("数据格式异常")

        # 下载按钮
        st.download_button(
            label="📥 下载对话数据 (JSON)",
            data=st.session_state.export_data,
            file_name=f"conversation_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            type="primary",
        )

    else:
        if st.button("准备导出数据"):
            # 生成导出数据
            export_data = {
                "session_id": session.session_id,
                "background": session.background_info.dict()
                if session.background_info
                else {},
                "conversation": [msg.dict() for msg in session.conversation_history],
                "metadata": {
                    "total_rounds": len(session.conversation_history),
                    "risk_alert": session.risk_alert,
                    "export_time": datetime.now().isoformat(),
                    "auto_generated": False,
                },
            }

            # 缓存导出数据到session state
            st.session_state.export_data = json.dumps(
                export_data, ensure_ascii=False, indent=2
            )
            st.success("✅ 导出数据已准备就绪！")
            st.rerun()

        # 下载按钮（如果数据已准备）
        if hasattr(st.session_state, "export_data"):
            st.download_button(
                label="📥 下载对话数据",
                data=st.session_state.export_data,
                file_name=f"conversation_{session.session_id}.json",
                mime="application/json",
            )


def main():
    """主函数"""
    # 初始化会话状态
    initialize_session_state()

    # 渲染页面头部
    render_header()

    # 显示错误信息（如果有）
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("清除错误"):
            st.session_state.error_message = None
            st.rerun()

    # 根据当前阶段渲染不同界面
    if st.session_state.generation_stage == "background":
        render_background_generation()

    elif st.session_state.generation_stage == "conversation":
        render_conversation()

    elif st.session_state.generation_stage == "completed":
        render_completion()

    # 侧边栏信息
    with st.sidebar:
        st.header("💡 使用说明")
        st.markdown("""
        1. **生成背景**: 创建学生和咨询师背景
        2. **进行对话**: 模拟心理咨询过程
           - 🔄 **继续对话**: 逐步生成对话
           - 🤖 **自动运行**: 一键生成完整对话并自动保存
        3. **质量评估**: 评估对话质量
        4. **导出数据**: 保存生成的数据
        """)

        st.header("📋 系统配置")
        st.json(
            {
                "最大轮数": SystemConfig.MAX_CONVERSATION_ROUNDS,
                "最小轮数": SystemConfig.MIN_CONVERSATION_ROUNDS,
                "风险阈值": SystemConfig.RISK_THRESHOLD,
            }
        )

        # 背景信息快速查看
        if st.session_state.session.background_info:
            with st.expander("📊 当前背景信息"):
                background = st.session_state.session.background_info

                st.write("**学生**")
                student = background.student_info
                st.write(f"👤 {student.name} ({student.age}岁)")
                st.write(f"🎓 {student.major}")

                issue_name = PSYCHOLOGICAL_ISSUES_DATA.get(
                    student.current_psychological_issue, {}
                ).get("name", "未知")
                st.write(f"💭 {issue_name}")

                st.write("**咨询师**")
                counselor = background.counselor_info
                st.write(f"👨‍⚕️ {counselor.name}")
                st.write(f"📈 {counselor.experience_years}年经验")

                approach_name = THERAPY_APPROACHES_DATA.get(
                    counselor.therapy_approach, {}
                ).get("name", "未知")
                st.write(f"🎯 {approach_name}")

        # 调试信息
        with st.expander("🔧 调试信息"):
            st.write("**会话状态:**")
            st.write(f"- 会话ID: {st.session_state.session_id}")
            st.write(f"- 当前阶段: {st.session_state.generation_stage}")
            st.write(f"- 是否处理中: {st.session_state.is_processing}")
            st.write(
                f"- 对话消息总数: {len(st.session_state.session.conversation_history)}"
            )
            st.write(f"- 当前轮次: {st.session_state.session.current_round}")
            st.write(f"- 是否有风险警报: {st.session_state.session.risk_alert}")
            st.write(
                f"- 当前咨询师状态: {getattr(st.session_state.session.counselor_bot, 'current_state', 'N/A')}"
            )

            # 显示对话历史的详细信息
            if st.session_state.session.conversation_history:
                st.write("**对话历史摘要:**")
                for i, msg in enumerate(st.session_state.session.conversation_history):
                    role_prefix = "Q" if msg.role == "student" else "A"
                    st.write(
                        f"  {i + 1}. {role_prefix}{msg.round_number} ({msg.role}): {msg.content[:30]}..."
                    )

        if st.button("🔄 重置系统"):
            # 清理所有状态，保留必要的锁
            keys_to_keep = {"init_lock"}
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
