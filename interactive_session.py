"""
交互式会话模块
提供一个SessionManager类来管理整个对话流程，支持手动控制对话的进行。
"""

import asyncio
from collections import Counter
import json
import sys
from typing import List, Dict, Any
import uuid
from datetime import datetime
import os

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

from llm_agent.quality_assess import QualityAssessmentAgent, QualityAssessmentContext
from models import (
    BackgroundContext,
    BackgroundInfo,
    ConversationMessage,
    CounselorState,
)
from llm_agent.background_gen import BackgroundGenerationAgent
from llm_agent.student import StudentBot
from llm_agent.counselor import CounselorBot
from llm_agent.flow_control import FlowControlAgent, FlowControlContext
from settings import settings

Traceloop.init(api_key=settings.TRACELOOP_API_KEY, disable_batch=True)


class Colors:
    """终端颜色定义"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(text: str, color: str = Colors.ENDC) -> None:
    """打印带颜色的文本"""
    print(f"{color}{text}{Colors.ENDC}")


def print_separator(char: str = "=", length: int = 80) -> None:
    """打印分隔线"""
    print_colored(char * length, Colors.OKCYAN)


def print_header(title: str) -> None:
    """打印标题头"""
    print_separator()
    print_colored(f"  {title}  ", Colors.HEADER + Colors.BOLD)
    print_separator()


def print_message(msg: ConversationMessage, is_new: bool = False) -> None:
    """打印消息"""
    if msg.role == "student":
        prefix = "🎓"
        color = Colors.OKCYAN
        role_text = f"学生 Q{msg.round_number}"
        extra_info = f"情绪: {msg.emotion}" if msg.emotion else ""
    else:
        prefix = "👨‍⚕️"
        color = Colors.OKGREEN
        role_text = f"咨询师 A{msg.round_number}"
        extra_info = f"状态: {msg.state}" if msg.state else ""

    if is_new:
        print_colored(f"\n{prefix} {role_text} (新消息)", color + Colors.BOLD)
    else:
        print_colored(f"\n{prefix} {role_text}", color + Colors.BOLD)

    if extra_info:
        print_colored(f"  [{extra_info}]", Colors.WARNING)

    # 打印消息内容，自动换行
    content_lines = msg.content.split("\n")
    for line in content_lines:
        if len(line) > 80:
            # 长行自动折行
            words = line.split(" ")
            current_line = ""
            for word in words:
                if len(current_line + word) > 76:
                    if current_line:
                        print(f"  {current_line}")
                        current_line = word
                    else:
                        print(f"  {word}")
                else:
                    current_line = current_line + " " + word if current_line else word
            if current_line:
                print(f"  {current_line}")
        else:
            print(f"  {line}")


def get_user_confirmation(prompt: str = "是否继续下一轮对话？") -> bool:
    """获取用户确认"""
    while True:
        print_colored(f"\n{prompt} [y/n/q(退出)]: ", Colors.WARNING + Colors.BOLD)
        try:
            response = input().lower().strip()
            if response in ["y", "yes", "是", ""]:
                return True
            elif response in ["n", "no", "否"]:
                return False
            elif response in ["q", "quit", "退出"]:
                print_colored("用户选择退出", Colors.FAIL)
                sys.exit(0)
            else:
                print_colored("请输入 y/n/q", Colors.FAIL)
        except KeyboardInterrupt:
            print_colored("\n\n用户中断程序", Colors.FAIL)
            sys.exit(0)


class SessionManager:
    """
    管理整个聊天流程的会话管理器
    """

    def __init__(self, auto_mode: bool = False):
        """
        初始化会话管理器，创建背景并保留

        Args:
            auto_mode: 是否开启自动模式，自动模式下会连续运行直到达到终止条件
        """
        mode_prefix = "auto" if auto_mode else "interactive"
        self.session_id = f"{mode_prefix}_{uuid.uuid4().hex[:8]}"
        self.background: BackgroundInfo = None
        self.initial_question: str = ""
        self.conversation_history: List[ConversationMessage] = []
        self.current_round = 0
        self.auto_mode = auto_mode
        self.max_rounds = 20  # 自动模式下的最大轮次

        # 新增：记录详细状态信息
        self.flow_control_results: List[Dict[str, Any]] = []
        self.state_transition_history: List[Dict[str, Any]] = []
        self.student_state_history: List[Dict[str, Any]] = []
        self.counselor_state_history: List[Dict[str, Any]] = []
        self.session_start_time = datetime.now()

        self.background_agent = BackgroundGenerationAgent()
        self.student_bot: StudentBot = None
        self.counselor_bot: CounselorBot = None
        self.flow_control_agent: FlowControlAgent = None

    @task(name="initialize_session", version=1)
    async def initialize_session(self):
        """
        异步初始化会话，生成背景信息
        """
        print_header("背景信息生成")
        print_colored("正在生成背景信息...", Colors.OKCYAN)
        try:
            background_result = await self.background_agent.execute(
                BackgroundContext(mode="random")
            )
            self.background = background_result
            self.initial_question = background_result.initial_question
            print_colored("✅ 背景信息生成成功！", Colors.OKGREEN)
            print_colored(
                json.dumps(self.background.model_dump(), indent=2, ensure_ascii=False)
            )
            print(f"  {self.initial_question}")
            print()
        except Exception as e:
            print_colored(f"❌ 背景生成失败: {str(e)}", Colors.FAIL)
            sys.exit(1)

        # 初始化 Agents
        self.student_bot = StudentBot()
        self.student_bot.update_background(self.background.student_info)
        self.counselor_bot = CounselorBot()
        self.counselor_bot.update_background(
            self.background.counselor_info, self.background.student_info
        )
        self.flow_control_agent = FlowControlAgent()

    def get_auto_mode_confirmation(
        self, prompt: str = "背景信息已生成，是否开始对话？"
    ) -> str:
        """
        获取用户确认，支持自动模式

        Returns:
            'yes': 继续手动模式
            'auto': 进入自动模式
            'no': 退出
        """
        while True:
            print_colored(
                f"\n{prompt} [y/a(自动模式)/n/q(退出)]: ", Colors.WARNING + Colors.BOLD
            )
            try:
                response = input().lower().strip()
                if response in ["y", "yes", "是", ""]:
                    return "yes"
                elif response in ["a", "auto", "自动"]:
                    return "auto"
                elif response in ["n", "no", "否"]:
                    return "no"
                elif response in ["q", "quit", "退出"]:
                    print_colored("用户选择退出", Colors.FAIL)
                    sys.exit(0)
                else:
                    print_colored(
                        "请输入 y(手动模式)/a(自动模式)/n(否)/q(退出)", Colors.FAIL
                    )
            except (KeyboardInterrupt, EOFError):
                print_colored("\n用户中断程序", Colors.FAIL)
                sys.exit(0)

    def _get_current_state_round(self) -> int:
        """
        获取当前状态持续的轮数
        """
        if not self.conversation_history:
            return 1
        current_state = self.counselor_bot.current_state
        return sum(
            1
            for state_record in self.counselor_state_history
            if state_record.get("state") == current_state
        )

    @task(name="conversation_loop", version=1)
    async def run(self):
        """
        循环调度咨询师Bot、流程控制Agent和学生Bot
        """
        await self.initialize_session()

        # 如果已经设置了自动模式，直接开始
        if self.auto_mode:
            print_colored("自动模式已启动，将连续运行直到达到终止条件", Colors.OKGREEN)
            start_conversation = True
        else:
            # 获取用户确认，支持选择自动模式
            confirmation = self.get_auto_mode_confirmation(
                "背景信息已生成，是否开始对话？"
            )
            if confirmation == "no":
                print_colored("用户取消对话。", Colors.WARNING)
                return
            elif confirmation == "auto":
                print_colored(
                    "进入自动模式，将连续运行直到达到终止条件", Colors.OKGREEN
                )
                self.auto_mode = True
                start_conversation = True
            else:
                start_conversation = True

        if not start_conversation:
            return

        print_header("开始心理咨询对话")
        if self.auto_mode:
            print_colored(
                f"自动模式：最大轮次 {self.max_rounds}，或状态跳转至None时结束",
                Colors.OKCYAN,
            )

        self.current_round = 1

        # 学生首轮
        student_msg = ConversationMessage(
            role="student",
            content=self.initial_question,
            emotion=self.student_bot.current_emotion,
            round_number=self.current_round,
        )
        self.conversation_history.append(student_msg)
        print_message(student_msg, is_new=True)

        # 记录初始学生状态
        self.student_state_history.append(
            {
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                "state": self.student_bot.get_student_state(),
                "phase": "initial",
            }
        )

        while True:
            # 咨询师回复
            print_colored(f"--- 第 {self.current_round} 轮对话 ---", Colors.HEADER)
            print_colored("正在生成咨询师回复...", Colors.OKCYAN)
            counselor_response = await self.counselor_bot.chat(
                self.conversation_history
            )
            counselor_msg = ConversationMessage(
                role="counselor",
                content=counselor_response,
                state=self.counselor_bot.current_state,
                round_number=self.current_round,
            )
            self.conversation_history.append(counselor_msg)
            print_message(counselor_msg, is_new=True)

            # 记录咨询师状态
            self.counselor_state_history.append(
                {
                    "round": self.current_round,
                    "timestamp": datetime.now().isoformat(),
                    "state": self.counselor_bot.current_state.value,
                    "message": counselor_response,
                }
            )

            # 流程控制
            print_colored("正在进行流程控制评估...", Colors.OKCYAN)
            flow_context = FlowControlContext(
                conversation_history=self.conversation_history,
                current_state=self.counselor_bot.current_state,
                current_state_round=self._get_current_state_round(),
                round_number=self.current_round,
                background_info=self.background,
                current_student_trust_level=self.student_bot.trust_level,
                current_student_openness_level=self.student_bot.openness_level,
                current_student_information_revealed=self.student_bot.information_revealed,
                current_student_emotion=self.student_bot.current_emotion,
            )
            flow_result = await self.flow_control_agent.execute(flow_context)

            # 记录流程控制结果
            self.flow_control_results.append(
                {
                    "round": self.current_round,
                    "timestamp": datetime.now().isoformat(),
                    "flow_result": flow_result.model_dump(),
                    "pre_student_state": self.student_bot.get_student_state(),
                }
            )

            print_colored("-" * 80, Colors.OKCYAN)
            print_colored("流程控制评估完成，状态评估结果: ", Colors.OKGREEN)
            print_colored(
                json.dumps(flow_result.model_dump(), indent=2, ensure_ascii=False)
            )
            print_colored(
                f"当前咨询师状态: {self.counselor_bot.current_state}", Colors.OKGREEN
            )
            print_colored("当前学生状态: ", Colors.OKGREEN)
            print(
                json.dumps(
                    self.student_bot.get_student_state(), indent=2, ensure_ascii=False
                )
            )
            print_colored("-" * 80, Colors.OKCYAN)

            # 更新学生状态
            await self.flow_control_agent.update_student_bot_state(
                self.student_bot, flow_result
            )

            # 检查是否需要状态转换
            state_transition = flow_result.state_transition
            should_end = False

            if state_transition.need_transition:
                if state_transition.recommended_state is None:
                    # 状态转换为None，表示对话应该结束
                    print_colored("流程控制建议结束对话", Colors.WARNING)
                    self.state_transition_history.append(
                        {
                            "round": self.current_round,
                            "timestamp": datetime.now().isoformat(),
                            "from_state": self.counselor_bot.current_state.value,
                            "to_state": None,
                            "reason": state_transition.transition_reason,
                            "auto_end": True,
                        }
                    )
                    should_end = True
                else:
                    new_state = CounselorState(state_transition.recommended_state)
                    self.state_transition_history.append(
                        {
                            "round": self.current_round,
                            "timestamp": datetime.now().isoformat(),
                            "from_state": self.counselor_bot.current_state.value,
                            "to_state": new_state.value,
                            "reason": state_transition.transition_reason,
                            "auto_end": False,
                        }
                    )
                    self.counselor_bot.update_state(
                        new_state, state_transition.transition_reason
                    )
                    print_colored(
                        f"咨询师状态转换为: {new_state.value}", Colors.WARNING
                    )

            # 记录更新后的学生状态
            self.student_state_history.append(
                {
                    "round": self.current_round,
                    "timestamp": datetime.now().isoformat(),
                    "state": self.student_bot.get_student_state(),
                    "phase": "after_flow_control",
                }
            )

            # 检查终止条件
            should_continue = True

            if should_end:
                print_colored("根据流程控制建议结束对话", Colors.WARNING)
                should_continue = False
            elif self.auto_mode:
                # 自动模式下的终止条件
                if self.current_round >= self.max_rounds:
                    print_colored(
                        f"自动模式：已达到最大轮次 {self.max_rounds}，结束对话",
                        Colors.WARNING,
                    )
                    should_continue = False
                else:
                    print_colored(
                        f"自动模式：继续第 {self.current_round + 1} 轮对话 (最大轮次: {self.max_rounds})",
                        Colors.OKCYAN,
                    )
                    should_continue = True
            else:
                # 手动模式下询问用户
                should_continue = get_user_confirmation("是否继续下一轮对话？")

            if not should_continue:
                break

            self.current_round += 1

            # 学生回复
            print_colored("正在生成学生回复...", Colors.OKCYAN)
            student_response = await self.student_bot.chat(self.conversation_history)
            student_msg = ConversationMessage(
                role="student",
                content=student_response,
                emotion=self.student_bot.current_emotion,
                round_number=self.current_round,
            )
            self.conversation_history.append(student_msg)
            print_message(student_msg, is_new=True)

            # 记录学生回复后的状态
            self.student_state_history.append(
                {
                    "round": self.current_round,
                    "timestamp": datetime.now().isoformat(),
                    "state": self.student_bot.get_student_state(),
                    "phase": "after_student_response",
                }
            )

        print_header("对话结束")
        await self.export_session_data()

    async def export_session_data(self):
        """
        导出会话数据为JSON文件
        """
        print_colored("正在导出会话数据...", Colors.OKCYAN)

        session_data = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": self.session_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_rounds": self.current_round,
            },
            "background_info": self.background.model_dump()
            if self.background
            else None,
            "initial_question": self.initial_question,
            "conversation_history": [
                msg.model_dump() for msg in self.conversation_history
            ],
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
        }

        # 创建导出目录
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{export_dir}/session_{self.session_id}_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print_colored(f"✅ 会话数据已导出到: {filename}", Colors.OKGREEN)
        except Exception as e:
            print_colored(f"❌ 导出失败: {str(e)}", Colors.FAIL)

    @task(name="quality_assess", version=1)
    async def quality_assess(self) -> Dict[str, Any]:
        rounds_per_state = dict(
            Counter(
                [
                    round_data["flow_result"]["state_transition"]["current_state"]
                    for round_data in self.flow_control_results
                ]
            )
        )
        counseling_trajectory = {
            "state_transitions": self.state_transition_history,
            "total_rounds": self.current_round,
            "rounds_per_state": rounds_per_state,
        }
        quality_context = QualityAssessmentContext(
            background_info=self.background,
            conversation_history=self.conversation_history,
            counseling_trajectory=counseling_trajectory,
        )
        quality_assessment_agent = QualityAssessmentAgent()
        assessment_result = await quality_assessment_agent.execute(quality_context)
        print_colored("质量评估结果:", Colors.OKGREEN)
        print_colored(
            json.dumps(assessment_result.model_dump(), indent=2, ensure_ascii=False)
        )
        return assessment_result.model_dump()


@workflow(name="心理咨询对话生成器", version=1)
async def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description="心理咨询对话生成器")
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help="启用自动模式，连续运行直到达到最大轮次或状态跳转至None",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=20, help="自动模式下的最大轮次数（默认：20）"
    )

    args = parser.parse_args()

    manager = SessionManager(auto_mode=args.auto)
    if args.auto:
        manager.max_rounds = args.max_rounds
        print_colored(f"自动模式启动，最大轮次: {args.max_rounds}", Colors.OKGREEN)

    await manager.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_colored("\n\n用户中断程序", Colors.FAIL)
        sys.exit(0)
