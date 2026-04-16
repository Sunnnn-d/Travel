import json
import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 配置日志记录器
logger = logging.getLogger(__name__)


class AttractionAgent:
    """
    景点推荐代理类

    使用阿里云通义千问大语言模型根据目的地、天气信息和用户偏好推荐旅游景点。
    通过较高的温度参数（0.7）增加推荐的多样性和创造性，为用户提供更丰富的选择。

    属性:
        model: ChatOpenAI 模型实例，配置为较高温度模式以增加推荐多样性

    工作流程:
        1. 接收用户的旅行信息（目的地、天气、偏好等）
        2. 构建包含详细要求的 system prompt，指定 JSON 输出格式
        3. 异步调用 LLM 生成景点推荐列表
        4. 解析并验证返回的 JSON 格式数据
        5. 如果解析失败，返回安全的默认结果
    """

    def __init__(self):
        """
        初始化 AttractionAgent 实例

        创建并配置 ChatOpenAI 模型实例（阿里云兼容），设置关键参数：
        - temperature: 0.7（较高值，增加景点推荐的多样性和创造性）
        - model: 使用阿里云 qwen-turbo 模型
        - base_url: 阿里云 DashScope API 端点

        温度参数说明:
            temperature 控制模型输出的随机性：
            - 0.0: 完全确定性输出，每次回答相同
            - 0.3: 较低随机性，适合需要准确性的任务（如天气分析）
            - 0.7: 中等偏高随机性，适合创意推荐和多样化选择
            - 1.0: 高随机性，适合创意写作

            这里使用 0.7 是因为景点推荐需要：
            1. 多样化的推荐结果（避免每次都推荐相同的景点）
            2. 创造性的建议（发现小众但有趣的景点）
            3. 个性化的匹配（根据用户偏好灵活调整）
            4. 平衡准确性和创新性
        """
        logger.info("正在初始化 AttractionAgent...")

        # 从环境变量中安全地获取阿里云 API 密钥
        # 优先从环境变量读取，提高安全性
        api_key = os.environ.get("DASHSCOPE_API_KEY")

        # 如果环境变量未设置，使用备用方案（仅用于开发测试）
        if not api_key:
            # 注意：生产环境不建议硬编码 API key，应使用环境变量
            api_key = "sk-140a61693bda414eb1c4e4e2e3996083"
            logger.warning("未找到 DASHSCOPE_API_KEY 环境变量，使用默认密钥（仅用于开发测试）")

        # 创建 ChatOpenAI 模型实例，配置阿里云 DashScope 端点
        # 阿里云 DashScope API 兼容 OpenAI 接口格式
        # 设置温度为 0.7 以增加景点推荐的多样性
        self.model = ChatOpenAI(
            api_key=api_key,
            temperature=0.7,
            model="qwen-turbo",  # 阿里云通义千问模型
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 API 端点
        )

        logger.info("AttractionAgent 初始化完成，使用模型: qwen-turbo, 温度: 0.7")

    async def recommend_attractions(
            self,
            destination: str,
            weather_info: Optional[Dict[str, Any]] = None,
            preferences: Optional[Dict[str, Any]] = None,
            duration: int = 1
    ) -> Dict[str, Any]:
        """
        异步推荐目的地的旅游景点

        根据用户提供的目的地、天气信息和偏好设置，使用大语言模型生成个性化的景点推荐列表。
        包含完整的错误处理机制，确保即使 API 调用或 JSON 解析失败也能返回有效结果。

        参数:
            destination (str): 旅游目的地城市或地区名称
            weather_info (Optional[Dict[str, Any]]): 天气信息字典，包含天气预报和分析结果
            preferences (Optional[Dict[str, Any]]): 用户偏好设置，如兴趣类型、预算等
            duration (int): 旅行持续天数，用于确定推荐景点的数量

        返回:
            Dict[str, Any]: 包含景点推荐信息的字典，结构如下：
                {
                    "success": bool,              # 操作是否成功
                    "attractions": List[Dict],     # 景点推荐列表
                    "destination": str,            # 目的地名称
                    "total_count": int,            # 推荐景点总数
                    "recommendation_summary": str, # 推荐理由摘要
                    "message": str,                # 状态消息
                    "error_code": str,             # 错误代码（可选，仅在失败时存在）
                    "timestamp": str               # 时间戳
                }

        异常处理:
            - 网络请求失败：捕获异常并返回空推荐列表
            - JSON 解析失败：尝试修复或使用默认值
            - 模型调用超时：返回超时提示信息
        """
        # 初始化可选参数，避免 None 值导致的错误
        if weather_info is None:
            weather_info = {}
        if preferences is None:
            preferences = {}

        # 记录请求开始时间
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info("=" * 60)
        logger.info(f"[景点推荐] 开始推荐 - 目的地: {destination}, 时长: {duration}天")
        if weather_info:
            weather_summary = weather_info.get("overall_weather", {}).get("summary", "未知")
            logger.info(f"  天气概况: {weather_summary}")
        if preferences:
            interest = preferences.get("interest", "未指定")
            logger.info(f"  用户兴趣: {interest}")
        logger.info("=" * 60)

        try:
            # 第一步：验证输入参数
            if not destination or not destination.strip():
                logger.error("目的地不能为空")
                return self._create_error_response(
                    destination=destination,
                    duration=duration,
                    error_message="目的地不能为空",
                    error_code="INVALID_DESTINATION",
                    timestamp=timestamp
                )

            if duration <= 0:
                logger.error(f"旅行天数无效: {duration}")
                return self._create_error_response(
                    destination=destination,
                    duration=duration,
                    error_message=f"旅行天数必须大于0，当前值: {duration}",
                    error_code="INVALID_DURATION",
                    timestamp=timestamp
                )

            # 第二步：构建系统提示词，指定景点推荐的格式和要求
            logger.debug("正在构建系统提示词...")
            system_prompt = self._build_system_prompt()

            # 第三步：构建用户消息，包含具体的旅行信息和上下文
            logger.debug("正在构建用户消息...")
            user_message = self._build_user_message(
                destination,
                weather_info,
                preferences,
                duration
            )

            # 第四步：异步调用大语言模型进行景点推荐
            logger.info("正在调用大语言模型进行景点推荐...")
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])

            # 第五步：提取模型返回的内容
            response_content = response.content
            logger.info(f"模型响应接收成功，响应长度: {len(response_content)} 字符")

            # 第六步：尝试解析 JSON 格式的响应
            logger.debug("正在解析模型响应...")
            attractions_data = self._parse_response(response_content)

            # 计算耗时
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            # 提取景点数量
            attractions_list = attractions_data.get("attractions", [])
            total_count = len(attractions_list)

            logger.info(f"✓ 景点推荐成功完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"  - 推荐景点数量: {total_count}个")

            # 打印每个景点的简要信息
            if attractions_list:
                logger.info("  推荐景点列表:")
                for idx, attraction in enumerate(attractions_list, 1):
                    name = attraction.get("name", "未知景点")
                    attr_type = attraction.get("type", "未知类型")
                    rating = attraction.get("rating", 0)
                    logger.info(f"    {idx}. {name} ({attr_type}) - 推荐指数: {rating}/5")

            # 第七步：返回成功的结果
            return {
                "success": True,
                "attractions": attractions_list,
                "destination": destination,
                "total_count": total_count,
                "recommendation_summary": attractions_data.get("summary", ""),
                "message": f"成功为 {destination} 推荐了 {total_count} 个景点",
                "timestamp": timestamp,
                "processing_time": f"{elapsed_time:.2f}s"
            }

        except TimeoutError as e:
            # 处理超时错误
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"景点推荐请求超时（已等待 {elapsed_time:.2f}秒）"
            logger.error(f"✗ {error_msg}: {str(e)}")

            return self._create_error_response(
                destination=destination,
                duration=duration,
                error_message=error_msg,
                error_code="TIMEOUT_ERROR",
                timestamp=timestamp
            )

        except ConnectionError as e:
            # 处理网络连接错误
            error_msg = f"网络连接失败，无法访问景点推荐服务"
            logger.error(f"✗ {error_msg}: {str(e)}")

            return self._create_error_response(
                destination=destination,
                duration=duration,
                error_message=error_msg,
                error_code="CONNECTION_ERROR",
                timestamp=timestamp
            )

        except Exception as e:
            # 错误处理：记录错误信息并返回安全默认值
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"景点推荐过程中发生未知错误（已运行 {elapsed_time:.2f}秒）"
            logger.error(f"✗ {error_msg}")
            logger.error(f"  错误类型: {type(e).__name__}")
            logger.error(f"  错误详情: {str(e)}")
            logger.exception("完整堆栈跟踪:")

            return self._create_error_response(
                destination=destination,
                duration=duration,
                error_message=error_msg,
                error_code="UNKNOWN_ERROR",
                timestamp=timestamp
            )

    def _create_error_response(
            self,
            destination: str,
            duration: int,
            error_message: str,
            error_code: str,
            timestamp: str
    ) -> Dict[str, Any]:
        """
        创建统一的错误响应格式

        参数:
            destination (str): 目的地名称
            duration (int): 旅行天数
            error_message (str): 错误描述信息
            error_code (str): 错误代码
            timestamp (str): 时间戳

        返回:
            Dict[str, Any]: 标准化的错误响应字典
        """
        return {
            "success": False,
            "attractions": [],
            "destination": destination,
            "total_count": 0,
            "recommendation_summary": "",
            "message": error_message,
            "error_code": error_code,
            "timestamp": timestamp,
            "suggestion": "请稍后重试或检查网络连接，如问题持续存在请联系技术支持"
        }

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        设计详细的指令来指导大语言模型如何根据目的地、天气和用户偏好推荐景点。
        指定输出格式为 JSON，确保返回的数据结构化和易于解析。

        返回:
            str: 格式化的系统提示词字符串

        提示词设计要点:
            1. 明确角色定位：专业的旅游规划师
            2. 指定输出格式：严格的 JSON 结构
            3. 定义字段要求：每个景点必须包含的关键信息
            4. 提供推荐标准：考虑天气、季节、用户偏好等因素
            5. 强调多样性：推荐不同类型的景点组合
        """
        prompt = """你是一个专业的旅游规划师助手，擅长根据目的地、天气条件和用户偏好推荐最适合的旅游景点。

你的任务是分析用户提供的旅行信息，并给出个性化的景点推荐建议。

【输出格式要求】
你必须以 JSON 格式返回结果，严格遵循以下结构：
{
    "summary": "整体推荐理由总结（100字以内，说明为什么推荐这些景点）",
    "attractions": [
        {
            "name": "景点名称",
            "type": "景点类型（如：自然景观、历史文化、现代建筑、美食街区、购物区、主题公园等）",
            "description": "景点简介和特色（50-100字）",
            "recommended_duration": "建议游览时长（如：2-3小时、半天、全天）",
            "best_visit_time": "最佳游览时间段（如：上午、下午、傍晚、夜晚）",
            "ticket_price_range": "门票价格范围（如：免费、50-100元、100-200元等）",
            "crowd_level": "预计人流密度（低/中/高）",
            "weather_suitability": ["适合的天气条件列表（如：晴天、多云、小雨等）"],
            "highlights": ["景点亮点列表（3-5个）"],
            "tips": "实用小贴士（如：最佳拍照点、注意事项、交通建议等）",
            "rating": 推荐指数（1-5分，基于用户偏好和天气条件）
        }
    ],
    "suggested_route": "建议的游览路线顺序（简要说明）",
    "alternative_options": ["备选景点列表（如果主要景点人多或关闭时的替代选择）"]
}

【推荐原则】
1. 根据目的地特色推荐最具代表性的景点
2. 结合天气信息，优先推荐适合当前天气的景点
3. 考虑用户的个人偏好（如喜欢自然、历史、美食等）
4. 平衡景点类型，提供多样化的体验组合
5. 考虑游览时间的合理性，不要安排过于紧凑的行程
6. 标注每个景点的特色和必看内容
7. 提供实用的游览建议和注意事项
8. 如果有季节性景点，特别说明

【多样性要求】
- 推荐的景点应该涵盖不同类型（自然、人文、娱乐、美食等）
- 考虑不同时间段的活动（白天、夜晚）
- 包含热门景点和小众特色景点的组合
- 根据旅行天数合理控制推荐数量（每天2-3个主要景点）

【注意事项】
- 如果无法获取实时信息，基于已知知识进行合理推荐
- 保持客观公正，不要过度夸大景点特色
- 使用中文回复
- 只返回 JSON 格式内容，不要添加其他解释文字
- 确保推荐的景点真实存在且可访问"""

        return prompt

    def _build_user_message(
            self,
            destination: str,
            weather_info: Dict[str, Any],
            preferences: Dict[str, Any],
            duration: int
    ) -> str:
        """
        构建用户消息

        将用户的旅行信息、天气信息和偏好格式化为清晰的消息文本，供大语言模型分析。

        参数:
            destination (str): 目的地名称
            weather_info (Dict[str, Any]): 天气信息字典
            preferences (Dict[str, Any]): 用户偏好
            duration (int): 旅行天数

        返回:
            str: 格式化后的用户消息
        """
        message = f"""请为我推荐以下旅行的景点：

【目的地】{destination}
【旅行时长】{duration}天

【天气信息】"""

        # 添加天气信息到上下文中
        if weather_info:
            overall_weather = weather_info.get("overall_weather", {})
            if overall_weather:
                summary = overall_weather.get("summary", "暂无天气概况")
                comfort_level = overall_weather.get("comfort_level", "未知")
                conditions = overall_weather.get("weather_conditions", [])

                message += f"\n- 天气概况: {summary}"
                message += f"\n- 舒适度: {comfort_level}/5分"
                if conditions:
                    message += f"\n- 主要天气: {', '.join(conditions)}"

            # 添加穿衣建议和必备物品
            recommendations = weather_info.get("travel_recommendations", {})
            if recommendations:
                clothing = recommendations.get("clothing_suggestions", "")
                essential_items = recommendations.get("essential_items", [])
                if clothing:
                    message += f"\n- 穿衣建议: {clothing}"
                if essential_items:
                    message += f"\n- 建议携带: {', '.join(essential_items)}"
        else:
            message += "\n暂无详细天气信息，请基于一般情况推荐"

        message += "\n\n【我的偏好】"
        if preferences:
            for key, value in preferences.items():
                message += f"\n- {key}: {value}"
        else:
            message += "\n无特殊偏好，请推荐经典的必游景点"

        message += f"\n\n【期望】请根据以上信息，为我推荐{duration}天行程的合适景点，并以 JSON 格式返回结果。"

        return message

    def _parse_response(self, response_content: str) -> Dict[str, Any]:
        """
        解析大语言模型的响应内容

        采用多种策略尝试解析 JSON 格式的响应内容，确保即使格式不完美也能提取有效数据。

        参数:
            response_content (str): 模型返回的原始文本内容

        返回:
            Dict[str, Any]: 解析后的字典，如果所有解析方法都失败则返回默认结构

        解析策略:
            1. 直接解析：尝试直接解析为标准 JSON
            2. 代码块提取：查找 markdown 代码块中的 JSON
            3. 花括号提取：查找第一个 { 和最后一个 } 之间的内容
            4. 默认返回：所有方法失败时返回安全的默认结构
        """
        logger.debug("开始解析模型响应...")

        # 定义默认的返回结构，确保即使解析失败也有有效的数据结构
        default_result = {
            "summary": "暂无景点推荐（服务暂时不可用）",
            "attractions": [],
            "suggested_route": "建议咨询当地旅游局或查阅旅游攻略网站",
            "alternative_options": []
        }

        try:
            # 策略1：尝试直接解析（最常见的情况）
            cleaned_content = response_content.strip()
            parsed_data = json.loads(cleaned_content)
            logger.debug("✓ JSON 解析成功（策略1：直接解析）")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.warning(f"直接JSON解析失败: {str(e)}，尝试其他方法...")

        try:
            # 策略2：查找 markdown 代码块中的 JSON
            import re
            json_pattern = r''
            match = re.search(json_pattern, response_content, re.DOTALL)

            if match:
                json_str = match.group(1)
                parsed_data = json.loads(json_str)
                logger.debug("✓ JSON 解析成功（策略2：代码块提取）")
                return parsed_data
            else:
                logger.debug("未找到 markdown 代码块")

        except Exception as e:
            logger.warning(f"代码块提取失败: {str(e)}")

        try:
            # 策略3：查找花括号之间的内容（处理包含额外文本的情况）
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = response_content[start_idx:end_idx + 1]
                parsed_data = json.loads(json_str)
                logger.debug("✓ JSON 解析成功（策略3：花括号提取）")
                return parsed_data
            else:
                logger.warning("未找到有效的 JSON 结构（花括号）")

        except Exception as e:
            logger.warning(f"花括号提取失败: {str(e)}")

        # 所有解析方法均失败，返回默认结构
        logger.error("✗ 所有JSON解析方法均失败，返回默认结构")
        return default_result

