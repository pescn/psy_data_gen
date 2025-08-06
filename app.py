import streamlit as st
import asyncio
import json
from datetime import datetime
import traceback

# 从您的原始代码文件中导入核心类
# 确保 interactive_session.py 在 Python 的可搜索路径中
from interactive_session import SessionManager, ConversationMessage, CounselorState

# ==============================================================================
# 猴子补丁 (Monkey Patching) 区域
# 为了让 SessionManager 更好地与 Streamlit 配合，我们在这里动态修改或添加一些方法
# 长期来看，最好的做法是直接将这些修改整合回 interactive_session.py 文件中
# ==============================================================================

def _get_current_state_round_fixed(self) -> int:
    """
    【已修复】计算当前状态连续进行的轮数。
    从后向前遍历历史，直到遇到不同状态为止。
    """
    if not self.counselor_bot or not self.counselor_bot.current_state or not self.counselor_state_history:
        return 1
        
    current_state_value = self.counselor_bot.current_state.value
    count = 0
    # 从后向前遍历咨询师状态历史
    for state_record in reversed(self.counselor_state_history):
        if state_record.get("state") == current_state_value:
            count += 1
        else:
            # 一旦遇到不同的状态，就停止计数
            break
            
    # 如果循环结束时 count 是 0，意味着这是一个全新的状态，这是它的第一轮
    return count if count > 0 else 1

async def execute_flow_control_and_update(self):
    """
    新方法：将流程控制的执行和学生状态的更新封装在一起。
    """
    from llm_agent.flow_control import FlowControlContext
    
    # 使用修复后的方法来获取当前状态的轮数
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

    # 记录流程控制结果
    self.flow_control_results.append(
        {
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "flow_result": flow_result.model_dump(),
        }
    )
    
    # 更新学生状态
    await self.flow_control_agent.update_student_bot_state(self.student_bot, flow_result)
    
    return flow_result

def handle_state_transition_fixed(self, flow_result):
    """
    【已修复】处理状态转换逻辑，并返回是否应该结束对话。
    """
    state_transition = flow_result.state_transition
    should_end = False

    if state_transition.need_transition:
        history_entry = {
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "from_state": self.counselor_bot.current_state.value,
            "reason": state_transition.transition_reason,
        }
        if state_transition.recommended_state is None:
            # 状态转换为None，表示对话应该结束
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
    """
    修改版：使其能返回数据字典，并可选择是否保存文件。
    """
    session_data = {
        "session_info": {"session_id": self.session_id, "start_time": self.session_start_time.isoformat(), "end_time": datetime.now().isoformat(), "total_rounds": self.current_round},
        "background_info": self.background.model_dump() if self.background else None,
        "initial_question": self.initial_question,
        "conversation_history": [msg.model_dump() for msg in self.conversation_history],
        "flow_control_results": self.flow_control_results,
        "state_transition_history": self.state_transition_history,
        "student_state_history": self.student_state_history,
        "counselor_state_history": self.counselor_state_history,
        "final_states": {"student_final_state": self.student_bot.get_student_state() if self.student_bot else None, "counselor_final_state": self.counselor_bot.current_state.value if self.counselor_bot else None},
        "quality_assessment": await self.quality_assess(),
        "usages": self.usage_summary,
    }
    if save_to_file:
        import os
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{export_dir}/session_{self.session_id}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"会话数据已导出到: {filename}")
    return session_data

# 使用 setattr 动态地将修复后的方法应用到 SessionManager 类上
# 这样可以避免修改原始的 aiser_interactive.py 文件
if not hasattr(SessionManager, '_monkey_patched'):
    setattr(SessionManager, '_get_current_state_round', _get_current_state_round_fixed)
    setattr(SessionManager, 'execute_flow_control_and_update', execute_flow_control_and_update)
    setattr(SessionManager, 'handle_state_transition', handle_state_transition_fixed)
    setattr(SessionManager, 'export_session_data', export_session_data_fixed)
    setattr(SessionManager, '_monkey_patched', True)

# ==============================================================================
# Streamlit UI 代码部分
# ==============================================================================

def run_async(awaitable):
    """在同步的Streamlit环境中运行异步代码的辅助函数"""
    return asyncio.run(awaitable)

def get_role_and_avatar(role: str):
    """根据角色返回头像"""
    if role == "student":
        return "🎓", "student"
    elif role == "counselor":
        return "👨‍⚕️", "assistant"
    return "🤖", "assistant"

def render_sidebar():
    with st.sidebar:
        st.header("会话状态监控")
        if st.button("🔄 开始新会话"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if "manager" in st.session_state:
            manager = st.session_state.manager
            st.subheader("👨‍⚕️ 咨询师状态")
            if manager.counselor_bot:
                st.info(f"当前阶段: **{manager.counselor_bot.current_state.value}**")
                # 使用修复后的方法来显示正确的轮数
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
        else:
            st.info("请先点击“开始会话”来初始化。")

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
                
                st.markdown(f"**第 {msg.round_number} 轮** - {extra_info}", unsafe_allow_html=True)
                st.markdown(msg.content)

def render_end_of_session_summary():
    st.success("对话已结束！")
    st.header("会话总结与质量评估")

    if "session_data" in st.session_state:
        session_data = st.session_state.session_data
        st.subheader("✨ 质量评估")
        st.json(session_data.get("quality_assessment", "评估失败或未执行。"))
        st.subheader("Token 使用情况")
        st.json(session_data.get("usages", []))
        st.download_button(
            label="📥 下载完整会话数据 (JSON)",
            data=json.dumps(session_data, indent=2, ensure_ascii=False),
            file_name=f"session_{session_data['session_info']['session_id']}.json",
            mime="application/json",
        )

def main():
    st.set_page_config(page_title="心理咨询对话生成器", layout="wide")
    st.title("🤖 心理咨询对话生成器")
    st.caption("这是一个模拟心理咨询师与学生对话的交互式工具。")

    render_sidebar()

    if "manager" not in st.session_state:
        st.info("点击下方按钮，将首先生成背景信息，然后开始第一轮对话。")
        if st.button("🚀 开始会话", type="primary"):
            with st.spinner("正在初始化会话，生成背景信息..."):
                try:
                    manager = SessionManager(auto_mode=True)
                    run_async(manager.initialize_session())
                    
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
                    st.success("背景生成完毕，第一轮对话已开始！")
                    st.rerun()
                except Exception as e:
                    st.error(f"初始化失败: {e}")
                    st.error(traceback.format_exc())

    elif not st.session_state.get("dialogue_finished", False):
        manager = st.session_state.manager
        render_conversation_history()
        st.markdown("---")

        button_text = f"➡️ 进行第 {manager.current_round} 轮 (咨询师回应)"
        if st.button(button_text, type="primary"):
            with st.spinner(f"第 {manager.current_round} 轮进行中... (咨询师思考 -> 流程评估 -> 学生回应)"):
                try:
                    # --- 咨询师回复 ---
                    counselor_response = run_async(manager.counselor_bot.chat(manager.conversation_history))
                    manager.usages.append(manager.counselor_bot.usage)
                    counselor_msg = ConversationMessage(
                        role="counselor",
                        content=counselor_response,
                        state=manager.counselor_bot.current_state,
                        round_number=manager.current_round,
                    )
                    manager.conversation_history.append(counselor_msg)
                    # 记录咨询师状态历史
                    manager.counselor_state_history.append({"round": manager.current_round, "timestamp": datetime.now().isoformat(), "state": manager.counselor_bot.current_state.value, "message": counselor_response})

                    # --- 流程控制与状态更新 ---
                    flow_result = run_async(manager.execute_flow_control_and_update())
                    st.session_state.latest_flow_control = flow_result.model_dump()
                    
                    # --- 状态转换检查 ---
                    should_end = manager.handle_state_transition(flow_result)

                    # --- 检查结束条件 ---
                    if should_end or manager.current_round >= manager.max_rounds:
                        st.session_state.dialogue_finished = True
                        with st.spinner("对话即将结束，正在进行最终评估和数据整理..."):
                            session_data = run_async(manager.export_session_data(save_to_file=False))
                            st.session_state.session_data = session_data
                    else:
                        # --- 学生为下一轮做准备 ---
                        student_response = run_async(manager.student_bot.chat(manager.conversation_history))
                        manager.usages.append(manager.student_bot.usage)
                        student_msg = ConversationMessage(
                            role="student",
                            content=student_response,
                            emotion=manager.student_bot.current_emotion,
                            # 学生的轮数是下一轮
                            round_number=manager.current_round + 1,
                        )
                        manager.conversation_history.append(student_msg)
                        
                        # 本轮结束，轮数+1
                        manager.current_round += 1

                except Exception as e:
                    st.error(f"在第 {manager.current_round} 轮对话中发生错误: {e}")
                    st.error(traceback.format_exc())
                    st.session_state.dialogue_finished = True

            st.rerun()

    else:
        render_conversation_history()
        st.markdown("---")
        render_end_of_session_summary()

if __name__ == "__main__":
    main()
