"""
背景生成Agent
负责生成学生和咨询师的背景信息，包括个人信息、心理问题等
"""

import random
from typing import Any

from .base import Agent
from constants import (
    PSYCHOLOGICAL_ISSUES_DATA,
    THERAPY_APPROACHES_DATA,
)
from models import BackgroundInfo, BackgroundContext

background_format_prompt = """# 输出格式
请返回一个完整的 Background 的 JSON对象，下面是其 Interface 结构：

```typescript
/** 学生信息 */
interface StudentInfo {
  age: number; // 年龄 (18-25)
  gender: '男' | '女';
  grade: string; // 年级
  major: string; // 专业

  family_background: string; // 详细的家庭背景描述 (100-200字)
  personality_traits: string[]; // 3 个主要性格特征
  psychological_profile: string; // 心理侧写描述 (150-250字)
  hidden_personal_info: string; // 深层个人信息和经历 (200-300字)
  symptom_description: string; // 症状的详细描述，以学生真实体验的方式表达 (200-300字)
}

/** 咨询师信息 */
interface CounselorInfo {
  therapy_approach: string; // 咨询师的咨询流派
  communication_style: string; // 沟通风格和习惯描述 (100-150字)
  specialization: string[]; // 专业领域 (2-3个)
}

/** 完整的咨询背景信息结构 */
interface Background {
  student_info: StudentInfo;
  counselor_info: CounselorInfo;
  /** 学生与咨询师交流的开放白。应体现学生的初始谨慎状态，只描述最表面、最明显的问题。(30-80字) */
  initial_question: string;
}
```

请确保生成的 JSON 对象符合上述结构，并且所有字段都包含有效内容。
"""


class BackgroundGenerationAgent(Agent[BackgroundContext, BackgroundInfo]):
    """
    背景生成Agent
    生成学生Bot和咨询师Bot的完整背景信息
    """

    context_class = BackgroundContext
    result_class = BackgroundInfo

    psychological_issue = None

    def prompt(self, context: BackgroundContext) -> str:
        """
        构建背景生成的提示词
        """
        base_prompt = """你是一个专业的心理咨询数据生成专家。请根据给定的基本信息，生成完整的学生和咨询师背景信息。

## 生成要求：
1. 学生背景要真实可信，符合大学生特点
2. 心理问题的描述要以学生的主观体验为主，不要过于专业化
3. 症状描述要循序渐进，最开始学生不会一次性说出所有问题,只会表达表面问题
4. 首句问题要单独生成在initial_question字段中，长度控制在30-80字，模拟学生真实的咨询开场，要体现学生的谨慎和试探性
5. 所有信息要相互一致，创造一个有内在逻辑的完整故事，形成完整的背景故事
"""
        # 获取用户指定的信息
        gen_mode = context.mode
        if gen_mode not in ["random", "guided"]:
            gen_mode = "random"  # 默认使用随机模式

        # 输出格式提示词
        if gen_mode == "guided" and context.psychological_issue:
            self.psychological_issue = context.psychological_issue
            guidance_prompt = f"""
## 用户指定信息：
- 心理问题类型：{context.psychological_issue}
- 额外背景描述及要求：{context.user_background}
"""
        else:
            guidance_prompt = f"""
## 基础配置
{self._random_issues_reference()}
{self._random_therapy_reference()}
"""
        # 构建完整的提示词
        prompt = base_prompt + guidance_prompt + background_format_prompt
        return prompt

    def _random_issues_reference(self) -> str:
        """随机选择一个心理学问题，构建心理问题参考信息"""
        if not self.psychological_issue:
            self.psychological_issue = random.choice(
                list(PSYCHOLOGICAL_ISSUES_DATA.keys())
            )
        data = PSYCHOLOGICAL_ISSUES_DATA[self.psychological_issue]
        reference = f"\n学生存在的心理学问题类型为 **{data['name']}**:\n\n"
        reference += f"- 描述：{data['description']}\n"
        reference += f"- 常见症状：{', '.join(data['common_symptoms'])}\n"
        reference += f"- 学生表达方式示例：{', '.join(data['student_expressions'])}\n"
        return reference

    def _random_therapy_reference(self) -> str:
        """随机选择一个咨询流派，构建咨询流派参考信息"""
        approach = random.choice(list(THERAPY_APPROACHES_DATA.keys()))
        data = THERAPY_APPROACHES_DATA[approach]
        reference = f"\n咨询师主要熟悉的心理咨询流派为 **{data['name']}**:\n\n"
        reference += f"- 描述：{data['description']}\n"
        reference += f"- 沟通风格：{', '.join(data['communication_style'])}\n"

        return reference

    def clean_response_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        清理和格式化响应数据
        确保所有字段符合预期格式
        """
        data["student_info"]["current_psychological_issue"] = (
            self.psychological_issue.value
        )
        return data
