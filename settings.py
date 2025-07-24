"""
全局系统配置
包含系统运行参数、控制逻辑等核心配置
"""

import os


class SystemConfig:
    """系统配置类"""

    # ==================== 对话控制参数 ====================
    MAX_CONVERSATION_ROUNDS = 50  # 最大对话轮数
    MIN_CONVERSATION_ROUNDS = 15  # 最小对话轮数

    # ==================== 状态转换控制 ====================
    MIN_ROUNDS_PER_STATE = 3  # 每个状态最小轮数
    MAX_ROUNDS_PER_STATE = 15  # 每个状态最大轮数

    # ==================== 风险控制 ====================
    RISK_THRESHOLD = 3  # 风险阈值，超过此值将终止对话

    # 风险关键词库
    SUICIDE_KEYWORDS = [
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

    SELF_HARM_KEYWORDS = [
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

    HARM_OTHERS_KEYWORDS = [
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
    DEFAULT_TEMPERATURE = 0.8  # 默认创造性温度
    DEFAULT_MAX_TOKENS = 1000  # 默认最大token数

    # ==================== 输出配置 ====================
    OUTPUT_DIR = "output"
    CONVERSATIONS_DIR = os.path.join(OUTPUT_DIR, "conversations")
    BACKGROUNDS_DIR = os.path.join(OUTPUT_DIR, "backgrounds")
    ASSESSMENTS_DIR = os.path.join(OUTPUT_DIR, "assessments")

    # ==================== StreamLit 界面配置 ====================
    PAGE_TITLE = "心理咨询对话数据生成系统"
    PAGE_ICON = "🧠"
    LAYOUT = "wide"

    # ==================== 数据生成配置 ====================
    DEFAULT_SESSION_PREFIX = "session"  # 会话ID前缀
    ENABLE_DEBUG_MODE = False  # 调试模式开关
    SAVE_INTERMEDIATE_RESULTS = True  # 是否保存中间结果

    @classmethod
    def ensure_output_dirs(cls):
        """确保输出目录存在"""
        dirs = [
            cls.OUTPUT_DIR,
            cls.CONVERSATIONS_DIR,
            cls.BACKGROUNDS_DIR,
            cls.ASSESSMENTS_DIR,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def get_risk_keywords(cls):
        """获取所有风险关键词"""
        return {
            "suicide": cls.SUICIDE_KEYWORDS,
            "self_harm": cls.SELF_HARM_KEYWORDS,
            "harm_others": cls.HARM_OTHERS_KEYWORDS,
        }
