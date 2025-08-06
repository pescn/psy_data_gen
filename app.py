from decimal import Decimal
import os
import streamlit as st
import asyncio
import json
from datetime import datetime
import traceback
import time

from interactive_session import SessionManager, ConversationMessage, CounselorState
from models import BackgroundContext


def _get_current_state_round_fixed(self) -> int:
    if (
        not self.counselor_bot
        or not self.counselor_bot.current_state
        or not self.counselor_state_history
    ):
        return 1
    current_state_value = self.counselor_bot.current_state.value
    count = 0
    for state_record in reversed(self.counselor_state_history):
        if state_record.get("state") == current_state_value:
            count += 1
        else:
            break
    return count if count > 0 else 1


async def execute_flow_control_and_update(self):
    from llm_agent.flow_control import FlowControlContext

    current_state_round = self._get_current_state_round()
    flow_context = FlowControlContext(
        conversation_history=self.conversation_history,
        current_state=self.counselor_bot.current_state,
        current_state_round=current_state_round,
        round_number=self.current_round,
        background_info=self.background,
        current_student_trust_level=self.student_bot.trust_level,
        current_student_openness_level=self.student_bot.openness_level,
        current_student_information_revealed=self.student_bot.information_revealed,
        current_student_emotion=self.student_bot.current_emotion,
    )
    flow_result = await self.flow_control_agent.execute(flow_context)
    self.usages.append(self.flow_control_agent.usage)
    self.flow_control_results.append(
        {"round": self.current_round, "flow_result": flow_result.model_dump()}
    )
    await self.flow_control_agent.update_student_bot_state(
        self.student_bot, flow_result
    )
    return flow_result


def handle_state_transition_fixed(self, flow_result):
    state_transition = flow_result.state_transition
    should_end = False
    if state_transition.need_transition:
        history_entry = {
            "round": self.current_round,
            "from_state": self.counselor_bot.current_state.value,
            "reason": state_transition.transition_reason,
        }
        if state_transition.recommended_state is None:
            history_entry["to_state"] = None
            should_end = True
        else:
            new_state = CounselorState(state_transition.recommended_state)
            history_entry["to_state"] = new_state.value
            self.counselor_bot.update_state(
                new_state, state_transition.transition_reason
            )
        self.state_transition_history.append(history_entry)
    return should_end


async def export_session_data_fixed(self, save_to_file=True):
    session_data = {
        "session_info": {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_rounds": self.current_round,
        },
        "background_info": self.background.model_dump() if self.background else None,
        "conversation_history": [msg.model_dump() for msg in self.conversation_history],
        "flow_control_results": self.flow_control_results,
        "state_transition_history": self.state_transition_history,
        "student_state_history": self.student_state_history,
        "counselor_state_history": self.counselor_state_history,
        "final_states": {
            "student_final_state": self.student_bot.get_student_state()
            if self.student_bot
            else None,
            "counselor_final_state": self.counselor_bot.current_state.value
            if self.counselor_bot
            else None,
        },
        "quality_assessment": await self.quality_assess(),
        "usages": self.usage_summary,
    }
    if save_to_file:
        import os

        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"{export_dir}/session_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"会话数据已导出到: {filename}")
    return session_data


if not hasattr(SessionManager, "_monkey_patched"):
    setattr(SessionManager, "_get_current_state_round", _get_current_state_round_fixed)
    setattr(
        SessionManager,
        "execute_flow_control_and_update",
        execute_flow_control_and_update,
    )
    setattr(SessionManager, "handle_state_transition", handle_state_transition_fixed)
    setattr(SessionManager, "export_session_data", export_session_data_fixed)
    setattr(SessionManager, "_monkey_patched", True)


def run_async(awaitable):
    return asyncio.run(awaitable)


def get_role_and_avatar(role: str):
    return ("🎓", "student") if role == "student" else ("👨‍⚕️", "assistant")


def render_sidebar():
    with st.sidebar:
        st.header("会话状态监控")
        if st.button("🔄 开始新会话"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if "manager" not in st.session_state:
            st.info("请先选择模式并点击“开始会话”来初始化。")
            return

        manager = st.session_state.manager

        # 【新增】显示背景信息
        st.subheader("📝 背景信息")
        if manager.background:
            # 使用 model_dump 将 Pydantic 对象转为字典以供 st.json 使用
            st.json(manager.background.model_dump(exclude_none=True))
        else:
            st.info("背景信息尚未生成。")

        st.subheader("👨‍⚕️ 咨询师状态")
        if manager.counselor_bot:
            st.info(f"当前阶段: **{manager.counselor_bot.current_state.value}**")
            st.write(f"该状态已持续 **{manager._get_current_state_round()}** 轮")
        else:
            st.info("等待会话开始...")

        st.subheader("🎓 学生状态")
        if manager.student_bot:
            st.json(manager.student_bot.get_student_state())
        else:
            st.info("等待会话开始...")

        st.subheader("📊 最新流程控制评估")
        if "latest_flow_control" in st.session_state:
            st.json(st.session_state.latest_flow_control)
        else:
            st.info("暂无评估结果")


def render_conversation_history():
    if "manager" in st.session_state:
        for msg in st.session_state.manager.conversation_history:
            avatar, role_name = get_role_and_avatar(msg.role)
            with st.chat_message(name=role_name, avatar=avatar):
                extra_info = ""
                if msg.role == "student" and msg.emotion:
                    extra_info = f"<small>情绪: {msg.emotion}</small>"
                elif msg.role == "counselor" and msg.state:
                    extra_info = f"<small>状态: {msg.state}</small>"
                st.markdown(
                    f"**第 {msg.round_number} 轮** - {extra_info}",
                    unsafe_allow_html=True,
                )
                st.markdown(msg.content)


def usage_summary(usages) -> str:
    """生成使用情况摘要"""
    if not usages:
        return "无使用情况数据"
    usage_details = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": Decimal("0.0"),
    }
    for item in usages:
        prompt_tokens = Decimal(item.get("prompt_tokens", 0))
        completion_tokens = Decimal(item.get("completion_tokens", 0))
        if prompt_tokens <= 32 * 1024:
            if completion_tokens <= 200:
                prompt_price = Decimal("0.0000008")
                completion_price = Decimal("0.000002")
            else:
                prompt_price = Decimal("0.0000008")
                completion_price = Decimal("0.000008")
        elif prompt_tokens <= 128 * 1024:
            prompt_price = Decimal("0.0000012")
            completion_price = Decimal("0.000016")
        else:
            prompt_price = Decimal("0.0000024")
            completion_price = Decimal("0.000024")
        total_cost = prompt_tokens * prompt_price + completion_tokens * completion_price
        usage_details["prompt_tokens"] += prompt_tokens
        usage_details["completion_tokens"] += completion_tokens
        usage_details["total_cost"] += total_cost
    return (
        f"总提示Token数: **{usage_details['prompt_tokens']}**, "
        f"总完成Token数: **{usage_details['completion_tokens']}**, "
        f"总成本: **{usage_details['total_cost'].quantize(Decimal('0.0001'))} 元**"
    )


def render_end_of_session_summary():
    st.success("对话已结束！")
    st.header("会话总结与质量评估")
    if "session_data" in st.session_state:
        session_data = st.session_state.session_data
        st.subheader("✨ 质量评估")
        st.json(session_data.get("quality_assessment", "评估失败或未执行。"))
        st.subheader("成本与 Token 使用情况")
        st.markdown(usage_summary(session_data.get("usages", [])))
        st.json(session_data.get("usages", []), expanded=False)

        # 直接保存到服务器 exports 目录
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{export_dir}/session_streamlit_{session_data['session_info']['session_id']}_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"✅ 会话数据已导出到: {filename}")
        except Exception as e:
            print(f"❌ 导出失败: {str(e)}")

        st.download_button(
            label="📥 下载完整会话数据 (JSON)",
            data=json.dumps(session_data, indent=2, ensure_ascii=False),
            file_name=f"session_{session_data['session_info']['session_id']}.json",
            mime="application/json",
        )


def run_one_round():
    """封装一轮对话的核心逻辑，用于手动和自动模式的复用"""
    manager = st.session_state.manager
    try:
        # --- 咨询师回复 ---
        counselor_response = run_async(
            manager.counselor_bot.chat(manager.conversation_history)
        )
        counselor_msg = ConversationMessage(
            role="counselor",
            content=counselor_response,
            state=manager.counselor_bot.current_state,
            round_number=manager.current_round,
        )
        manager.conversation_history.append(counselor_msg)
        manager.counselor_state_history.append(
            {
                "round": manager.current_round,
                "state": manager.counselor_bot.current_state.value,
                "message": counselor_response,
            }
        )

        # --- 流程控制与状态更新 ---
        flow_result = run_async(manager.execute_flow_control_and_update())
        st.session_state.latest_flow_control = flow_result.model_dump()

        # --- 状态转换检查 ---
        should_end = manager.handle_state_transition(flow_result)

        # --- 检查结束条件 ---
        if should_end or manager.current_round >= manager.max_rounds:
            st.session_state.dialogue_finished = True
            with st.spinner("对话即将结束，正在进行最终评估..."):
                session_data = run_async(
                    manager.export_session_data(save_to_file=False)
                )
                st.session_state.session_data = session_data
        else:
            # --- 学生为下一轮做准备 ---
            student_response = run_async(
                manager.student_bot.chat(manager.conversation_history)
            )
            student_msg = ConversationMessage(
                role="student",
                content=student_response,
                emotion=manager.student_bot.current_emotion,
                round_number=manager.current_round + 1,
            )
            manager.conversation_history.append(student_msg)
            manager.current_round += 1

    except Exception as e:
        st.error(f"在第 {manager.current_round} 轮对话中发生错误: {e}")
        st.error(traceback.format_exc())
        st.session_state.dialogue_finished = True


def main():
    st.set_page_config(page_title="心理咨询对话生成器", layout="wide")
    st.title("🤖 心理咨询对话生成器")

    render_sidebar()

    # --- 阶段 1: 初始化 ---
    if "manager" not in st.session_state:
        st.subheader("1. 配置背景生成方式")
        generation_mode_selection = st.radio(
            "选择背景生成方式",
            ("随机生成", "自定义配置"),
            horizontal=True,
            captions=(
                "完全自动生成学生背景和心理问题。",
                "允许您输入关键信息来影响生成结果。",
            ),
        )
        user_background_input = ""
        psychological_issue_input = ""
        if generation_mode_selection == "自定义配置":
            user_background_input = st.text_area(
                "请输入额外的背景描述和要求（可选）",
                placeholder="例如：学生来自单亲家庭，性格内向，最近感到焦虑。",
                height=100,
            )
            psychological_issue_input = st.text_input(
                "请输入心理问题类型（可选）",
                placeholder="例如：焦虑症、抑郁症等。",
            )
        mode = "random" if generation_mode_selection == "随机生成" else "guided"
        st.markdown("---")
        st.subheader("2. 选择对话模式")
        mode = st.radio(
            "选择运行模式",
            ("手动模式", "自动模式"),
            horizontal=True,
            captions=(
                "每轮手动点击按钮，方便分析。",
                "一次性自动运行所有对话，无需干预。",
            ),
        )

        if st.button("🚀 开始会话", type="primary"):
            with st.spinner("正在初始化会话，生成背景信息..."):
                try:
                    manager = SessionManager(auto_mode=True)
                    background = BackgroundContext(mode="random")
                    if mode == "guided":
                        background = BackgroundContext(
                            mode=mode,
                            user_background=user_background_input,
                            psychological_issue=psychological_issue_input,
                        )
                    run_async(manager.initialize_session(background))
                    student_msg = ConversationMessage(
                        role="student",
                        content=manager.initial_question,
                        emotion=manager.student_bot.current_emotion,
                        round_number=1,
                    )
                    manager.conversation_history.append(student_msg)
                    manager.current_round = 1

                    st.session_state.manager = manager
                    st.session_state.dialogue_finished = False
                    st.session_state.mode = mode  # 保存选择的模式
                    st.success("背景生成完毕，第一轮对话已开始！")
                    st.rerun()
                except Exception as e:
                    st.error(f"初始化失败: {e}")
                    st.error(traceback.format_exc())

    # --- 阶段 4: 对话结束 ---
    elif st.session_state.get("dialogue_finished", False):
        render_conversation_history()
        st.markdown("---")
        render_end_of_session_summary()

    # --- 阶段 2: 手动模式 ---
    elif st.session_state.mode == "手动模式":
        render_conversation_history()
        st.markdown("---")
        button_text = (
            f"➡️ 进行第 {st.session_state.manager.current_round} 轮 (咨询师回应)"
        )
        if st.button(button_text, type="primary"):
            with st.spinner(f"第 {st.session_state.manager.current_round} 轮进行中..."):
                run_one_round()
            st.rerun()

    # --- 阶段 3: 自动模式 ---
    elif st.session_state.mode == "自动模式":
        render_conversation_history()
        st.markdown("---")
        st.info(
            f"🤖 **自动模式运行中**... 当前正在处理第 {st.session_state.manager.current_round} 轮。"
        )
        # 自动运行一轮
        run_one_round()
        # 等待2秒，方便用户观看
        time.sleep(2)
        # 强制页面刷新以进入下一轮循环
        st.rerun()


if __name__ == "__main__":
    main()
