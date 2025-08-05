"""
全局系统配置
包含系统运行参数、控制逻辑等核心配置
"""

import os
from typing import Dict, List

from pydantic_settings import BaseSettings


class SystemConfig(BaseSettings):
    """系统配置类"""

    # ==================== 对话控制参数 ====================
    MAX_CONVERSATION_ROUNDS: int = 50  # 最大对话轮数
    MIN_CONVERSATION_ROUNDS: int = 15  # 最小对话轮数

    # ==================== 状态转换控制 ====================
    MIN_ROUNDS_PER_STATE: int = 3  # 每个状态最小轮数
    MAX_ROUNDS_PER_STATE: int = 15  # 每个状态最大轮数

    # ==================== 风险控制 ====================
    RISK_THRESHOLD: int = 3  # 风险阈值，超过此值将终止对话

    # 风险关键词库
    SUICIDE_KEYWORDS: List[str] = [
        "自杀",
        "结束生命",
        "不想活",
        "死了算了",
        "想死",
        "自我了断",
        "活着没意思",
        "消失",
        "离开这个世界",
        "解脱",
    ]

    SELF_HARM_KEYWORDS: List[str] = [
        "自残",
        "割腕",
        "伤害自己",
        "自我伤害",
        "切割",
        "撞墙",
        "拿刀",
        "疼痛",
        "流血",
        "划伤",
    ]

    HARM_OTHERS_KEYWORDS: List[str] = [
        "伤害别人",
        "报复",
        "杀死",
        "打击",
        "攻击",
        "暴力",
        "恨死",
        "弄死",
        "教训",
    ]

    # ==================== LLM 配置 ====================
    DEFAULT_TEMPERATURE: float = 0.8  # 默认创造性温度
    DEFAULT_MAX_TOKENS: int = 16384  # 默认最大token数

    # ==================== 输出配置 ====================
    OUTPUT_DIR: str = "output"
    CONVERSATIONS_DIR: str = os.path.join(OUTPUT_DIR, "conversations")
    BACKGROUNDS_DIR: str = os.path.join(OUTPUT_DIR, "backgrounds")
    ASSESSMENTS_DIR: str = os.path.join(OUTPUT_DIR, "assessments")

    # ==================== StreamLit 界面配置 ====================
    PAGE_TITLE: str = "心理咨询对话数据生成系统"
    PAGE_ICON: str = "🧠"
    LAYOUT: str = "wide"

    # ==================== 数据生成配置 ====================
    DEFAULT_SESSION_PREFIX: str = "session"  # 会话ID前缀
    ENABLE_DEBUG_MODE: bool = False  # 调试模式开关
    SAVE_INTERMEDIATE_RESULTS: bool = True  # 是否保存中间结果

    # ==================== 大语言模型相关 ====================
    LLM_API_KEY: str = ""
    LLM_API_BASE_URL: str = "http://localhost:8000/v1"  # LLM API 基础URL
    LLM_MODEL: str = "deepseek-v3"  # 默认模型名称

    # ==================== 监控和日志配置 ====================
    TRACELOOP_API_KEY: str = ""  # Traceloop API Key

    def ensure_output_dirs(self):
        """确保输出目录存在"""
        dirs = [
            self.OUTPUT_DIR,
            self.CONVERSATIONS_DIR,
            self.BACKGROUNDS_DIR,
            self.ASSESSMENTS_DIR,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def get_risk_keywords(self) -> Dict[str, List[str]]:
        """获取所有风险关键词"""
        return {
            "suicide": self.SUICIDE_KEYWORDS,
            "self_harm": self.SELF_HARM_KEYWORDS,
            "harm_others": self.HARM_OTHERS_KEYWORDS,
        }


settings = SystemConfig(_env_file=".env", _env_file_encoding="utf-8")

print(f"LLM_API_KEY: {settings.LLM_API_KEY}")
