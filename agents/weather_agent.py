import json
import os
import logging
import requests
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 配置日志记录器
logger = logging.getLogger(__name__)


class WeatherAgent:
    """
    天气分析代理类

    使用阿里云大语言模型分析目的地天气情况并提供出行建议。
    通过精心设计的提示词和结构化输出，确保天气分析的准确性和可用性。

    属性:
        model: ChatOpenAI 模型实例，配置为低温度模式以保证输出稳定性

    工作流程:
        1. 接收用户的旅行信息（目的地、日期等）
        2. 构建包含详细要求的 system prompt
        3. 调用 LLM 生成天气分析报告
        4. 解析并验证返回的 JSON 格式数据
        5. 如果解析失败，返回安全的默认结果
    """

    def __init__(self):
        """
        初始化 WeatherAgent 实例

        创建并配置 ChatOpenAI 模型实例（阿里云兼容），设置关键参数：
        - temperature: 0.3（较低值，保证输出的确定性和一致性）
        - model: 使用阿里云 qwen-turbo 模型
        - base_url: 阿里云 DashScope API 端点

        温度参数说明:
            temperature 控制模型输出的随机性：
            - 0.0: 完全确定性输出，每次回答相同
            - 0.3: 较低随机性，适合需要准确性的任务（如天气分析）
            - 0.7: 中等随机性，平衡创造性和准确性
            - 1.0: 高随机性，适合创意写作

            这里使用 0.3 是因为天气分析需要：
            1. 稳定的输出格式（便于后续解析）
            2. 准确的天气信息（减少幻觉）
            3. 一致的分析质量
        """
        logger.info("正在初始化 WeatherAgent...")

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
        self.model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            model="qwen-turbo",  # 阿里云通义千问模型
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 API 端点
        )

        # 天气 API 配置
        self.weather_api_key = "969f5924c2542b20279e713d4547aaf6"  # OpenWeatherMap API Key
        self.weather_api_base_url = "https://api.openweathermap.org/data/2.5"

        logger.info("WeatherAgent 初始化完成，使用模型: qwen-turbo, 温度: 0.3")

    async def analyze_weather(
            self,
            destination: str,
            start_date: str,
            duration: int,
            preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        异步分析目的地天气情况

        根据用户提供的目的地和日期，使用真实天气 API 获取数据，然后结合 LLM 生成天气分析报告和出行建议。
        包含完整的错误处理机制，确保即使 API 调用或 JSON 解析失败也能返回有效结果。

        参数:
            destination (str): 旅游目的地城市或地区名称
            start_date (str): 旅行开始日期，格式为 YYYY-MM-DD
            duration (int): 旅行持续天数
            preferences (Optional[Dict[str, Any]]): 用户偏好设置，如活动类型等

        返回:
            Dict[str, Any]: 包含天气分析信息的字典，结构如下：
                {
                    "success": bool,              # 操作是否成功
                    "weather_analysis": Dict,      # 天气分析详情
                    "destination": str,            # 目的地名称
                    "duration": int,               # 旅行天数
                    "message": str,                # 状态消息
                    "error_code": str,             # 错误代码（可选，仅在失败时存在）
                    "timestamp": str               # 时间戳
                }

        异常处理:
            - 网络请求失败：捕获异常并返回空分析结果
            - JSON 解析失败：尝试修复或使用默认值
            - 模型调用超时：返回超时提示信息
        """
        if preferences is None:
            preferences = {}

        # 记录请求开始时间
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info("=" * 60)
        logger.info(f"[天气分析] 开始分析 - 目的地: {destination}, 日期: {start_date}, 时长: {duration}天")
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

            if not start_date or not start_date.strip():
                logger.error("开始日期不能为空")
                return self._create_error_response(
                    destination=destination,
                    duration=duration,
                    error_message="开始日期不能为空",
                    error_code="INVALID_DATE",
                    timestamp=timestamp
                )

            # 第二步：获取真实天气数据
            logger.info("正在获取真实天气数据...")
            weather_api_data = None
            coordinates = self._get_city_coordinates(destination)
            
            if coordinates and coordinates.get("lat") and coordinates.get("lon"):
                logger.info(f"获取到 {destination} 的坐标: {coordinates}")
                weather_api_data = self._get_weather_forecast(
                    coordinates["lat"], 
                    coordinates["lon"], 
                    min(duration, 5)  # OpenWeatherMap 免费版最多提供 5 天预报
                )
                if weather_api_data:
                    logger.info("成功获取天气 forecast 数据")
                else:
                    logger.warning("未能获取天气 forecast 数据，将使用 LLM 推断")
            else:
                logger.warning("未能获取城市坐标，将使用 LLM 推断天气")

            # 第三步：构建系统提示词，指定天气分析的格式和要求
            logger.debug("正在构建系统提示词...")
            system_prompt = self._build_system_prompt()

            # 第四步：构建用户消息，包含具体的旅行信息和真实天气数据
            logger.debug("正在构建用户消息...")
            user_message = self._build_user_message(
                destination,
                start_date,
                duration,
                preferences,
                weather_api_data
            )

            # 第五步：异步调用大语言模型进行天气分析
            logger.info("正在调用大语言模型进行天气分析...")
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])

            # 第六步：提取模型返回的内容
            response_content = response.content
            logger.info(f"模型响应接收成功，响应长度: {len(response_content)} 字符")

            # 第七步：尝试解析 JSON 格式的响应
            logger.debug("正在解析模型响应...")
            weather_data = self._parse_response(response_content)

            # 计算耗时
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            logger.info(f"✓ 天气分析成功完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"  - 舒适度评级: {weather_data.get('overall_weather', {}).get('comfort_level', 'N/A')}/5")
            logger.info(f"  - 最佳出行日: {len(weather_data.get('best_travel_days', []))}天")
            logger.info(f"  - 预警信息: {len(weather_data.get('warnings', []))}条")

            # 第八步：返回成功的结果
            return {
                "success": True,
                "weather_analysis": weather_data,
                "destination": destination,
                "duration": duration,
                "message": f"成功获取 {destination} 的天气分析",
                "timestamp": timestamp,
                "processing_time": f"{elapsed_time:.2f}s"
            }

        except TimeoutError as e:
            # 处理超时错误
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"天气分析请求超时（已等待 {elapsed_time:.2f}秒）"
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
            error_msg = f"网络连接失败，无法访问天气分析服务"
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
            error_msg = f"天气分析过程中发生未知错误（已运行 {elapsed_time:.2f}秒）"
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
            "weather_analysis": self._get_default_weather_analysis(),
            "destination": destination,
            "duration": duration,
            "message": error_message,
            "error_code": error_code,
            "timestamp": timestamp,
            "suggestion": "请稍后重试或检查网络连接，如问题持续存在请联系技术支持"
        }

    def _get_default_weather_analysis(self) -> Dict[str, Any]:
        """
        获取默认的天气预报数据结构

        返回:
            Dict[str, Any]: 默认的天气分析数据结构
        """
        return {
            "overall_weather": {
                "summary": "暂无天气分析（服务暂时不可用）",
                "average_temperature": "未知",
                "weather_conditions": [],
                "comfort_level": 3
            },
            "daily_forecast": [],
            "travel_recommendations": {
                "clothing_suggestions": "建议出发前查询当地实时天气预报",
                "best_activities": ["室内博物馆参观", "美术馆游览", "购物中心逛街"],
                "activities_to_avoid": ["长时间户外活动", "登山徒步"],
                "essential_items": ["雨具", "常用药品", "充电宝", "身份证"],
                "health_tips": "注意关注天气变化，适时增减衣物"
            },
            "warnings": ["未能获取实时天气数据，请以当地气象部门发布为准"],
            "best_travel_days": [],
            "overall_rating": 3
        }

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        设计详细的指令来指导大语言模型如何分析天气情况并提供出行建议。
        指定输出格式为 JSON，确保返回的数据结构化和易于解析。

        返回:
            str: 格式化的系统提示词字符串

        提示词设计要点:
            1. 明确角色定位：专业的天气分析师
            2. 指定输出格式：严格的 JSON 结构
            3. 定义字段要求：必须包含的关键天气信息
            4. 提供分析标准：帮助用户理解天气影响
            5. 考虑实际因素：季节、气候特点、活动适宜性等
        """
        prompt = """你是一个专业的天气分析师助手，擅长根据目的地和日期提供详细的天气分析和出行建议。

你的任务是分析用户提供的旅行信息，并给出天气相关的专业建议。

【输出格式要求】
你必须以 JSON 格式返回结果，严格遵循以下结构：
{
    "overall_weather": {
        "summary": "整体天气概况总结（50字以内）",
        "average_temperature": "平均温度范围（如：15-25°C）",
        "weather_conditions": ["主要天气状况列表（如：晴天、多云、小雨）"],
        "comfort_level": "舒适度评级（1-5分，5分为最舒适）"
    },
    "daily_forecast": [
        {
            "date": "日期（YYYY-MM-DD）",
            "temperature_high": 最高温度（数字）,
            "temperature_low": 最低温度（数字）,
            "condition": "天气状况（如：晴、多云、小雨、大雨等）",
            "humidity": "湿度百分比（如：60%）",
            "wind": "风力描述（如：微风、3-4级北风）",
            "uv_index": "紫外线指数（1-10）",
            "precipitation_chance": "降水概率（0-100%）"
        }
    ],
    "travel_recommendations": {
        "clothing_suggestions": "穿衣建议（如：建议穿长袖衬衫，携带薄外套）",
        "best_activities": ["推荐的户外活动列表"],
        "activities_to_avoid": ["不建议的活动列表"],
        "essential_items": ["必备物品清单（如：雨伞、防晒霜、墨镜等）"],
        "health_tips": "健康提示（如：注意防晒、多喝水等）"
    },
    "warnings": [
        "天气预警信息列表（如：暴雨预警、高温预警等，如无则留空数组）"
    ],
    "best_travel_days": ["最适合出行的日期列表"],
    "overall_rating": 整体出行推荐评分（1-5分）
}

【分析原则】
1. 基于目的地的地理位置和历史气候数据进行分析
2. 考虑季节因素对天气的影响
3. 提供实用的出行建议和注意事项
4. 标注可能的极端天气情况
5. 结合旅行时长给出分日天气预报
6. 考虑不同天气条件下的活动适宜性

【注意事项】
- 如果无法获取实时天气数据，基于历史气候数据进行合理推断
- 标注数据来源和可信度
- 保持客观公正，不要过度夸大天气影响
- 使用中文回复
- 只返回 JSON 格式内容，不要添加其他解释文字"""

        return prompt

    def _build_user_message(
            self,
            destination: str,
            start_date: str,
            duration: int,
            preferences: Dict[str, Any],
            weather_api_data: Optional[Dict] = None
    ) -> str:
        """
        构建用户消息

        将用户的旅行信息和真实天气数据格式化为清晰的消息文本，供大语言模型分析。

        参数:
            destination (str): 目的地名称
            start_date (str): 开始日期
            duration (int): 旅行天数
            preferences (Dict[str, Any]): 用户偏好
            weather_api_data (Optional[Dict]): 真实天气 API 数据

        返回:
            str: 格式化后的用户消息
        """
        message = f"""请为我分析以下旅行的天气情况：

        【目的地】{destination}
        【开始日期】{start_date}
        【旅行时长】{duration}天

        【我的偏好】"""

        if preferences:
            for key, value in preferences.items():
                message += f"\n- {key}: {value}"
        else:
            message += "\n无特殊偏好，请提供全面的天气分析"

        # 添加真实天气 API 数据
        if weather_api_data:
            message += "\n\n【真实天气数据】\n"
            message += "以下是从 OpenWeatherMap API 获取的真实天气数据，请基于这些数据进行分析：\n"
            
            # 提取并格式化天气数据
            if "list" in weather_api_data:
                daily_data = {}
                for item in weather_api_data["list"]:
                    date = item["dt_txt"].split(" ")[0]
                    if date not in daily_data:
                        daily_data[date] = {
                            "temp_min": item["main"]["temp_min"],
                            "temp_max": item["main"]["temp_max"],
                            "condition": item["weather"][0]["description"],
                            "humidity": item["main"]["humidity"],
                            "wind_speed": item["wind"]["speed"],
                            "precipitation_chance": item.get("pop", 0) * 100
                        }
                
                for date, data in daily_data.items():
                    message += f"\n日期: {date}\n"
                    message += f"  温度范围: {data['temp_min']:.1f}°C 至 {data['temp_max']:.1f}°C\n"
                    message += f"  天气状况: {data['condition']}\n"
                    message += f"  湿度: {data['humidity']}%\n"
                    message += f"  风速: {data['wind_speed']} m/s\n"
                    message += f"  降水概率: {data['precipitation_chance']:.1f}%\n"
            else:
                message += "\n未获取到详细天气数据\n"
        else:
            message += "\n\n【天气数据】\n"
            message += "未获取到真实天气数据，请基于历史气候数据进行合理推断\n"

        message += "\n\n请根据以上信息，为我提供详细的天气分析和出行建议，并以 JSON 格式返回结果。"

        return message

    def _get_city_coordinates(self, city_name: str) -> Optional[Dict[str, float]]:
        """
        获取城市的经纬度坐标

        参数:
            city_name (str): 城市名称

        返回:
            Optional[Dict[str, float]]: 包含经度和纬度的字典，失败时返回 None
        """
        try:
            url = f"{self.weather_api_base_url}/weather"
            params = {
                "q": city_name,
                "appid": self.weather_api_key,
                "units": "metric",
                "lang": "zh_cn"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                "lat": data.get("coord", {}).get("lat"),
                "lon": data.get("coord", {}).get("lon")
            }
        except Exception as e:
            logger.error(f"获取城市坐标失败: {str(e)}")
            return None

    def _get_weather_forecast(self, lat: float, lon: float, days: int) -> Optional[Dict]:
        """
        获取天气 forecast 数据

        参数:
            lat (float): 纬度
            lon (float): 经度
            days (int): 预测天数

        返回:
            Optional[Dict]: 天气 forecast 数据，失败时返回 None
        """
        try:
            url = f"{self.weather_api_base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.weather_api_key,
                "units": "metric",
                "lang": "zh_cn",
                "cnt": days * 8  # 每3小时一个数据点，一天8个
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取天气 forecast 失败: {str(e)}")
            return None

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
            json_pattern = r'```json\n(.*?)\n```'
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
        return self._get_default_weather_analysis()
