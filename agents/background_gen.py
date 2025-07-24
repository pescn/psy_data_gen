"""
背景生成Agent
负责生成学生和咨询师的背景信息，包括个人信息、心理问题等
"""

import random
import re
from typing import Dict, Any, Optional

from agents.base import Agent
from models import (
    BackgroundInfo, StudentBackground, CounselorBackground,
    PsychologicalIssue, TherapyApproach
)
from constants import (
    PSYCHOLOGICAL_ISSUES_DATA, THERAPY_APPROACHES_DATA,
    COMMON_MAJORS, COMMON_GRADES, COMMON_FAMILY_BACKGROUNDS,
    COMMON_PERSONALITY_TRAITS
)


class BackgroundGenerationAgent(Agent):
    """
    背景生成Agent
    生成学生Bot和咨询师Bot的完整背景信息
    """
    
    def _init_config(self, **kwargs):
        """初始化配置"""
        # LLM配置通过类属性定义
        pass
    
    def _build_prompt(self, **context) -> str:
        """
        构建背景生成的提示词
        
        Args:
            **context: 包含用户输入的背景信息或随机生成标志
        """
        # 获取用户指定的信息
        user_specified_issue = context.get('psychological_issue')
        user_specified_background = context.get('user_background', '')
        generation_mode = context.get('mode', 'random')  # 'random' 或 'guided'
        
        # 构建心理问题参考信息
        issues_reference = self._build_issues_reference()
        therapy_reference = self._build_therapy_reference()
        
        if generation_mode == 'guided' and user_specified_issue:
            # 用户指定模式
            prompt = f"""你是一个专业的心理咨询数据生成专家。请根据用户指定的信息，生成完整的学生和咨询师背景信息。

用户指定信息：
- 心理问题类型：{user_specified_issue}
- 额外背景描述：{user_specified_background}

{issues_reference}

{therapy_reference}

请生成一个完整的背景信息JSON对象，包含以下结构：
{{
    "student_info": {{
        "name": "学生姓名",
        "age": 年龄(18-25),
        "gender": "男/女",
        "grade": "年级",
        "major": "专业",
        "family_background": "详细的家庭背景描述(100-200字)",
        "personality_traits": ["性格特征1", "性格特征2", "性格特征3"],
        "psychological_profile": "心理侧写描述(150-250字)",
        "hidden_personal_info": "深层个人信息和经历(200-300字)",
        "current_psychological_issue": "{user_specified_issue}",
        "symptom_description": "症状的详细描述，以学生真实体验的方式表达(200-300字)"
    }},
    "counselor_info": {{
        "name": "咨询师姓名",
        "therapy_approach": "咨询流派",
        "communication_style": "沟通风格和习惯描述(100-150字)",
        "experience_years": 从业年限(3-15),
        "specialization": ["专业领域1", "专业领域2"]
    }},
    "initial_question": "学生首次咨询的问题Q0，应该体现学生的初始谨慎状态，只描述最表面、最明显的问题，不会一次性说出所有困扰(30-80字)",
    "generation_params": {{
        "mode": "guided",
        "user_input": "{user_specified_background}",
        "complexity_level": "适中/复杂"
    }}
}}

生成要求：
1. 学生背景要真实可信，符合大学生特点
2. 心理问题的描述要以学生的主观体验为主，不要过于专业化
3. 症状描述要循序渐进，最开始学生不会一次性说出所有问题
4. **首句问题Q0要单独生成在initial_question字段中，长度控制在30-80字，体现学生的谨慎和试探性**
5. **症状描述字段不要包含Q0内容，只描述症状本身**
6. 咨询师的流派选择要与学生问题匹配
7. 所有信息要相互一致，形成完整的背景故事

请只返回JSON格式，不要包含其他解释。"""

        else:
            # 随机生成模式
            prompt = f"""你是一个专业的心理咨询数据生成专家。请随机生成一个完整的学生和咨询师背景信息。

{issues_reference}

{therapy_reference}

请生成一个完整的背景信息JSON对象，包含以下结构：
{{
    "student_info": {{
        "name": "随机学生姓名",
        "age": 年龄(18-25),
        "gender": "男/女",
        "grade": "年级",
        "major": "专业",
        "family_background": "详细的家庭背景描述(100-200字)",
        "personality_traits": ["性格特征1", "性格特征2", "性格特征3"],
        "psychological_profile": "心理侧写描述(150-250字)",
        "hidden_personal_info": "深层个人信息和经历(200-300字)",
        "current_psychological_issue": "从心理问题列表中随机选择",
        "symptom_description": "症状的详细描述，以学生真实体验的方式表达(200-300字)"
    }},
    "counselor_info": {{
        "name": "随机咨询师姓名",
        "therapy_approach": "从咨询流派中随机选择",
        "communication_style": "沟通风格和习惯描述(100-150字)",
        "experience_years": 从业年限(3-15),
        "specialization": ["专业领域1", "专业领域2"]
    }},
    "initial_question": "学生首次咨询的问题Q0，应该体现学生的初始谨慎状态，只描述最表面、最明显的问题，不会一次性说出所有困扰(30-80字)",
    "generation_params": {{
        "mode": "random",
        "complexity_level": "适中/复杂",
        "random_seed": "用于复现的随机种子"
    }}
}}

生成要求：
1. 从提供的心理问题列表中随机选择一个作为核心问题
2. 学生背景要真实可信，符合大学生特点
3. 心理问题的描述要以学生的主观体验为主，避免专业术语
4. 症状要有层次性，初期只会表达表面问题
5. **首句问题Q0要单独生成在initial_question字段中，长度控制在30-80字，模拟学生真实的咨询开场**
6. **症状描述字段不要包含Q0内容，只描述症状和感受**
7. 咨询师的流派要与学生问题类型匹配
8. 创造一个有内在逻辑的完整故事
9. 增加适当的复杂性和深度

请只返回JSON格式，不要包含其他解释。"""

        return prompt
    
    def _build_issues_reference(self) -> str:
        """构建心理问题参考信息"""
        reference = "心理问题参考信息：\n"
        
        for issue, data in PSYCHOLOGICAL_ISSUES_DATA.items():
            reference += f"\n{issue.value} - {data['name']}:\n"
            reference += f"描述：{data['description']}\n"
            reference += f"常见症状：{', '.join(data['common_symptoms'][:5])}\n"
            reference += f"学生表达方式示例：{', '.join(data['student_expressions'][:3])}\n"
        
        return reference
    
    def _build_therapy_reference(self) -> str:
        """构建咨询流派参考信息"""
        reference = "\n咨询流派参考信息：\n"
        
        for approach, data in THERAPY_APPROACHES_DATA.items():
            reference += f"\n{approach.value} - {data['name']}:\n"
            reference += f"描述：{data['description']}\n"
            reference += f"沟通风格：{', '.join(data['communication_style'][:3])}\n"
        
        return reference
    
    def _extract_json_from_response(self, response: str) -> str:
        """从响应中提取JSON内容"""
        # 移除可能的markdown代码块标记
        response = response.strip()
        
        # 使用正则表达式查找JSON内容
        # 匹配 ```json...``` 或 ```...``` 包装的内容
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1)
        
        # 如果没有找到代码块，查找纯JSON对象
        # 查找从第一个{到最后一个}的内容
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return response[start_idx:end_idx + 1]
        
        # 如果都没找到，返回原始响应让后续处理
        return response
    
    def _extract_initial_question_from_symptom(self, symptom_desc: str) -> str:
        """从症状描述中提取初始问题Q0"""
        # 查找Q0:后面的内容
        q0_pattern = r'Q0[：:]\s*(.+?)(?=\n|$|。(?:\s|$))'
        match = re.search(q0_pattern, symptom_desc)
        
        if match:
            question = match.group(1).strip()
            # 清理可能的多余标点
            question = re.sub(r'[。！？]*$', '', question)
            return question
        
        # 如果没有找到Q0标记，查找是否整个描述就是问题形式
        if symptom_desc.startswith(('老师', '我', '您好')) and len(symptom_desc) < 200:
            return symptom_desc.strip()
        
        return ""
    
    def _clean_symptom_description(self, symptom_desc: str) -> str:
        """清理症状描述中的Q0内容"""
        # 移除Q0:开头的部分
        cleaned = re.sub(r'Q0[：:]\s*[^。！？]*[。！？]*', '', symptom_desc)
        
        # 清理多余的空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 如果清理后内容太短，保留原内容但移除Q0标记
        if len(cleaned) < 50:
            cleaned = re.sub(r'Q0[：:]\s*', '', symptom_desc)
        
        return cleaned
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 清理响应内容，提取JSON
            cleaned_response = self._extract_json_from_response(response)
            parsed_data = self._safe_json_parse(cleaned_response)
            
            # 验证必需的字段
            required_fields = ['student_info', 'counselor_info']
            for field in required_fields:
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # 处理initial_question字段
            if 'initial_question' not in parsed_data:
                # 如果没有单独的initial_question，尝试从symptom_description中提取
                student_info = parsed_data.get('student_info', {})
                symptom_desc = student_info.get('symptom_description', '')
                
                # 查找Q0内容
                initial_question = self._extract_initial_question_from_symptom(symptom_desc)
                if initial_question:
                    parsed_data['initial_question'] = initial_question
                    # 清理symptom_description中的Q0内容
                    cleaned_symptom = self._clean_symptom_description(symptom_desc)
                    parsed_data['student_info']['symptom_description'] = cleaned_symptom
                else:
                    # 如果完全没有找到，生成一个默认的
                    issue_key = student_info.get('current_psychological_issue', 'academic_anxiety')
                    # 确保issue_key是有效的
                    if issue_key not in [issue.value for issue in PsychologicalIssue]:
                        issue_key = 'academic_anxiety'
                    
                    # 从constants中获取对应的问题数据
                    for issue, data in PSYCHOLOGICAL_ISSUES_DATA.items():
                        if issue.value == issue_key:
                            default_expressions = data.get('student_expressions', [])
                            parsed_data['initial_question'] = default_expressions[0] if default_expressions else "老师，我需要您的帮助。"
                            break
                    else:
                        parsed_data['initial_question'] = "老师，我需要您的帮助。"
            
            # 验证学生信息字段
            student_required = [
                'name', 'age', 'gender', 'grade', 'major', 
                'family_background', 'personality_traits', 'psychological_profile',
                'hidden_personal_info', 'current_psychological_issue', 'symptom_description'
            ]
            for field in student_required:
                if field not in parsed_data['student_info']:
                    raise ValueError(f"Missing student field: {field}")
            
            # 验证咨询师信息字段
            counselor_required = [
                'name', 'therapy_approach', 'communication_style', 
                'experience_years', 'specialization'
            ]
            for field in counselor_required:
                if field not in parsed_data['counselor_info']:
                    raise ValueError(f"Missing counselor field: {field}")
            
            return parsed_data
            
        except Exception as e:
            raise ValueError(f"Failed to parse background generation response: {str(e)}")
    
    async def generate_background(
        self, 
        psychological_issue: Optional[str] = None,
        user_background: str = "",
        mode: str = "random"
    ) -> tuple[BackgroundInfo, str]:
        """
        生成背景信息和首句问题
        
        Args:
            psychological_issue: 指定的心理问题类型
            user_background: 用户提供的背景描述
            mode: 生成模式 ('random' 或 'guided')
            
        Returns:
            tuple[BackgroundInfo, str]: (背景信息, 首句问题Q0)
        """
        context = {
            'psychological_issue': psychological_issue,
            'user_background': user_background,
            'mode': mode
        }
        
        # 执行生成
        result = await self.execute(**context)
        
        # 转换为Pydantic模型
        try:
            # 转换心理问题枚举
            issue_str = result['student_info']['current_psychological_issue']
            psychological_issue_enum = None
            for issue in PsychologicalIssue:
                if issue.value == issue_str or PSYCHOLOGICAL_ISSUES_DATA[issue]['name'] == issue_str:
                    psychological_issue_enum = issue
                    break
            
            if not psychological_issue_enum:
                # 如果没有匹配，默认使用第一个
                psychological_issue_enum = list(PsychologicalIssue)[0]
            
            # 转换咨询流派枚举
            approach_str = result['counselor_info']['therapy_approach']
            therapy_approach_enum = None
            for approach in TherapyApproach:
                if approach.value == approach_str or THERAPY_APPROACHES_DATA[approach]['name'] == approach_str:
                    therapy_approach_enum = approach
                    break
            
            if not therapy_approach_enum:
                # 如果没有匹配，默认使用第一个
                therapy_approach_enum = list(TherapyApproach)[0]
            
            # 创建学生背景
            student_info = StudentBackground(
                name=result['student_info']['name'],
                age=result['student_info']['age'],
                gender=result['student_info']['gender'],
                grade=result['student_info']['grade'],
                major=result['student_info']['major'],
                family_background=result['student_info']['family_background'],
                personality_traits=result['student_info']['personality_traits'],
                psychological_profile=result['student_info']['psychological_profile'],
                hidden_personal_info=result['student_info']['hidden_personal_info'],
                current_psychological_issue=psychological_issue_enum,
                symptom_description=result['student_info']['symptom_description']
            )
            
            # 创建咨询师背景
            counselor_info = CounselorBackground(
                name=result['counselor_info']['name'],
                therapy_approach=therapy_approach_enum,
                communication_style=result['counselor_info']['communication_style'],
                experience_years=result['counselor_info']['experience_years'],
                specialization=result['counselor_info']['specialization']
            )
            
            # 创建背景信息
            background = BackgroundInfo(
                student_info=student_info,
                counselor_info=counselor_info,
                generation_params=result.get('generation_params', {})
            )
            
            # 获取首句问题
            initial_question = result.get('initial_question', '')
            
            return background, initial_question
            
        except Exception as e:
            raise ValueError(f"Failed to create background models: {str(e)}")
    
    def get_available_issues(self) -> Dict[str, str]:
        """获取可用的心理问题列表"""
        return {
            issue.value: data['name'] 
            for issue, data in PSYCHOLOGICAL_ISSUES_DATA.items()
        }
    
    def get_available_approaches(self) -> Dict[str, str]:
        """获取可用的咨询流派列表"""
        return {
            approach.value: data['name']
            for approach, data in THERAPY_APPROACHES_DATA.items()
        }
    
    def get_issue_details(self, issue: PsychologicalIssue) -> Dict[str, Any]:
        """获取特定心理问题的详细信息"""
        return PSYCHOLOGICAL_ISSUES_DATA.get(issue, {})
    
    def get_approach_details(self, approach: TherapyApproach) -> Dict[str, Any]:
        """获取特定咨询流派的详细信息"""
        return THERAPY_APPROACHES_DATA.get(approach, {})
    
    def validate_background(self, background: BackgroundInfo) -> Dict[str, Any]:
        """
        验证生成的背景信息质量
        
        Args:
            background: 背景信息
            
        Returns:
            Dict: 验证结果
        """
        issues = []
        warnings = []
        
        # 检查学生信息完整性
        student = background.student_info
        if len(student.name) < 2:
            issues.append("学生姓名过短")
        if not (18 <= student.age <= 25):
            warnings.append("学生年龄不在典型范围内")
        if len(student.family_background) < 50:
            warnings.append("家庭背景描述过短")
        if len(student.symptom_description) < 100:
            warnings.append("症状描述过短")
        
        # 检查咨询师信息完整性
        counselor = background.counselor_info
        if len(counselor.name) < 2:
            issues.append("咨询师姓名过短")
        if not (3 <= counselor.experience_years <= 15):
            warnings.append("咨询师经验年限不在典型范围内")
        if len(counselor.communication_style) < 50:
            warnings.append("沟通风格描述过短")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "completeness_score": self._calculate_completeness_score(background),
            "consistency_score": self._calculate_consistency_score(background)
        }
    
    def _calculate_completeness_score(self, background: BackgroundInfo) -> float:
        """计算完整性评分"""
        score = 0.0
        total_checks = 10
        
        student = background.student_info
        counselor = background.counselor_info
        
        # 学生信息检查
        if len(student.family_background) >= 100: score += 1
        if len(student.psychological_profile) >= 150: score += 1
        if len(student.hidden_personal_info) >= 200: score += 1
        if len(student.symptom_description) >= 200: score += 1
        if len(student.personality_traits) >= 3: score += 1
        
        # 咨询师信息检查
        if len(counselor.communication_style) >= 100: score += 1
        if len(counselor.specialization) >= 2: score += 1
        if 3 <= counselor.experience_years <= 15: score += 1
        
        # 整体一致性检查
        if student.major in COMMON_MAJORS: score += 1
        if student.grade in COMMON_GRADES: score += 1
        
        return score / total_checks
    
    def _calculate_consistency_score(self, background: BackgroundInfo) -> float:
        """计算一致性评分"""
        # 这里可以实现更复杂的一致性检查逻辑
        # 例如检查心理问题与症状描述的匹配度
        # 检查咨询师流派与问题类型的适配性等
        return 0.8  # 暂时返回固定值，后续可以优化
