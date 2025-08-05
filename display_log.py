import json
import os
import argparse
from decimal import Decimal

import inquirer
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from typing import List

console = Console()


def get_log_files() -> List[str]:
    """获取exports目录下的所有JSON日志文件"""
    base_path = os.path.abspath("./exports")
    if not os.path.exists(base_path):
        console.print(f"[red]错误：导出目录 {base_path} 不存在[/red]")
        return []
    try:
        files = os.listdir(base_path)
        json_files = [f for f in files if f.endswith(".json")]
        return sorted(json_files, reverse=True)  # 按时间倒序排列
    except Exception as e:
        console.print(f"[red]读取目录时出错：{e}[/red]")
        return []


def display_background(data: dict):
    background_info = data["background_info"]
    student_info = background_info.get("student_info", {})
    counselor_info = background_info.get("counselor_info", {})
    student_base_str = (
        f"年龄: {student_info.get('age', '未知')}, "
        f"性别: {student_info.get('gender', '未知')}, "
        f"年级: {student_info.get('grade', '未知')}, "
        f"专业: {student_info.get('major', '未知')}"
    )
    student_background_str = (
        f"家庭背景: {student_info.get('family_background', '未知')}\n"
        f"性格特征: {', '.join(student_info.get('personality_traits', []))}\n"
        f"心理侧写: {student_info.get('psychological_profile', '未知')}\n"
        f"隐形个人信息: {student_info.get('hidden_personal_info', '未知')}\n"
        f"症状描述: {student_info.get('symptom_description', '未知')}"
    )
    counselor_str = (
        f"咨询流派: {counselor_info.get('therapy_approach', '未知')}\n"
        f"沟通风格: {counselor_info.get('communication_style', '未知')}\n"
    )
    content_group = Group(
        f"[yellow][bold]学生基本信息[/bold][/yellow]: \n{student_base_str}\n",
        f"[yellow][bold]学生背景信息[/bold][/yellow]: \n{student_background_str}\n",
        Rule(style="white"),
        f"[cyan][bold]咨询师背景信息[/bold][/cyan]: \n{counselor_str}",
    )
    return Panel(
        content_group,
        title="背景信息",
        border_style="green",
    )


def display_conversation(data: dict):
    """显示对话内容"""
    conversation = data.get("conversation_history", [])
    if not conversation:
        return "[red]没有找到对话内容[/red]"

    messages = []
    for message in conversation:
        role = {
            "student": "[magenta][bold]学生[/bold][/magenta]",
            "counselor": "[blue][bold]咨询师[/bold][/blue]",
        }.get(message.get("role"))
        content = message.get("content", "")
        messages.append(f"[{role}]: \n{content}")
        if message.get("role") == "counselor":
            messages.append(
                Rule(
                    title=f"第 {message.get('round_number', 0)} 轮结束，咨询状态: {message.get('state', '无')}",
                    style="white",
                )
            )
    group = Group(*messages)
    if not group:
        console.print("[red]对话内容为空[/red]")
        return
    return Panel(
        group,
        title="对话内容",
        border_style="#b35220",
    )


def display_state_transitions(data: dict):
    """显示状态转换信息"""
    state_history = data.get("state_transition_history", [])
    if not state_history:
        return "[red]没有找到状态转换信息[/red]"

    transitions = []
    for item in state_history:
        transitions.append(
            f"[[bold]第{item['round']}轮[/bold] 结束状态切换: {item['from_state']} -> {item['to_state']}] \n"
            f"{item.get('reason', '无')}\n"
        )
    return Panel(
        Group(*transitions),
        title="状态转换记录",
        border_style="#4720b3",
    )


def display_quality_assessment(data: dict):
    """显示质量评估结果"""
    assessment = data.get("quality_assessment", {})
    if not assessment:
        return "[red]没有找到质量评估结果[/red]"
    overall_quality = assessment.get("overall_quality", {})
    score = overall_quality.get("total_score", "未知")
    good_comment = "，".join(overall_quality.get("strengths", []))
    bad_comment = "，".join(overall_quality.get("weaknesses", []))

    consistency_check = assessment.get("consistency_check", {})
    issue_consistency = consistency_check.get("issue_consistency", "未知")
    issue_consistency_comment = consistency_check.get("consistency_analysis", "无")

    return f"""
[bold]质量评估结果[/bold]:
- [bold]总体评分:[/bold] {score}
- [bold]一致性检查:[/bold] {"与预设病情一致" if issue_consistency else "与预设病情不一致"}
- [bold]优点:[/bold]\n    {good_comment if good_comment else "无"}
- [bold]缺点:[/bold]\n    {bad_comment if bad_comment else "无"}
- [bold]一致性分析:[/bold]\n    {issue_consistency_comment if issue_consistency_comment else "无"}
"""


def display_usage(data: dict):
    """显示API调用的使用情况"""
    usage = data.get("usages", [])
    if not usage:
        return "[red]没有找到API调用使用情况[/red]"

    usage_details = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": Decimal("0.0"),
    }
    for item in usage:
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
    return Group(
        "[bold]成本情况[/bold]: ",
        f"总提示Token数: [bold]{usage_details['prompt_tokens']}[/bold]",
        f"总完成Token数: [bold]{usage_details['completion_tokens']}[/bold]",
        f"总成本: [bold]{usage_details['total_cost'].quantize(Decimal('0.0001'))} 元[/bold]",
    )


def display_summary(data: dict):
    return Panel(
        Group(
            display_quality_assessment(data),
            display_usage(data),
        ),
        title="总结分析",
        border_style="cyan",
    )


def display_log(file_path: str):
    if not os.path.exists(file_path):
        console.print(f"[red]错误：文件 {file_path} 不存在[/red]")
        return

    try:
        console.print("[blue]正在显示日志文件：[/blue] " + file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        console.print(display_background(data))
        console.print(display_conversation(data))
        console.print(display_state_transitions(data))
        console.print(display_summary(data))
    except Exception as e:
        console.print(f"[red]读取文件时出错：{e}[/red]")


def select_and_display_log() -> None:
    """显示日志文件选择菜单并展示选中的日志"""
    log_files = get_log_files()
    if not log_files:
        console.print("[yellow]没有找到任何日志文件[/yellow]")
        return

    # 创建选择菜单
    questions = [
        inquirer.List(
            "log_file",
            message="请选择要查看的日志文件（使用上下键选择，回车确认）",
            choices=log_files,
            carousel=True,  # 允许循环选择
        ),
    ]
    try:
        answers = inquirer.prompt(questions)
        if answers and answers["log_file"]:
            selected_file = answers["log_file"]
            file_path = os.path.join(os.path.abspath("./exports"), selected_file)
            console.print(f"\n[green]已选择：{selected_file}[/green]\n")
            display_log(file_path)
        else:
            console.print("[yellow]未选择任何文件[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[yellow]操作已取消[/yellow]")
    except Exception as e:
        console.print(f"[red]选择过程中出错：{e}[/red]")


def main():
    """主函数"""
    console.print(
        Panel("日志文件查看器", subtitle="使用上下键选择日志文件", border_style="green")
    )
    parser = argparse.ArgumentParser(description="心理咨询日志查看器")
    parser.add_argument(
        "--file",
        type=str,
        help="指定要查看的日志文件路径，如果不指定则显示选择菜单",
    )
    args = parser.parse_args()
    if args.file:
        display_log(args.file)
    else:
        # 如果没有指定文件，则显示选择菜单
        console.print("[blue]请选择要查看的日志文件：[/blue]")
        select_and_display_log()


if __name__ == "__main__":
    main()
