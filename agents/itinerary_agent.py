import json
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 配置日志记录器
logger = logging.getLogger(__name__)


class ItineraryAgent:
    """
    行程规划代理类

    使用阿里云通义千问大语言模型根据目的地、天气信息、景点推荐和用户偏好生成详细的旅行行程安排。
    通过适中的温度参数（0.5）平衡创造性和准确性，既保证行程的合理性，又提供个性化的建议。

    属性:
        model: ChatOpenAI 模型实例，配置为中等温度模式以平衡创造性和准确性

    工作流程:
        1. 接收完整的旅行信息（目的地、天气、景点、偏好等）
        2. 构建包含详细要求的 system prompt，指定行程规划的 JSON 输出格式
        3. 异步调用 LLM 生成详细的每日行程安排
        4. 解析并验证返回的 JSON 格式数据
        5. 如果解析失败，返回安全的默认结果
    """

    def __init__(self):
        """
        初始化 ItineraryAgent 实例

        创建并配置 ChatOpenAI 模型实例（阿里云兼容），设置关键参数：
        - temperature: 0.5（适中值，平衡创造性和准确性）
        - model: 使用阿里云 qwen-turbo 模型
        - base_url: 阿里云 DashScope API 端点

        温度参数说明:
            temperature 控制模型输出的随机性：
            - 0.0: 完全确定性输出，每次回答相同
            - 0.3: 较低随机性，适合需要准确性的任务（如天气分析）
            - 0.5: 中等随机性，平衡创造性和准确性，适合行程规划
            - 0.7: 较高随机性，适合创意推荐（如景点推荐）
            - 1.0: 高随机性，适合创意写作

            这里使用 0.5 是因为行程规划需要：
            1. 合理的时间安排（需要一定的准确性）
            2. 个性化的活动建议（需要一定的创造性）
            3. 灵活应对不同天气和偏好（需要平衡判断）
            4. 在结构化框架内提供多样化选择
        """
        logger.info("正在初始化 ItineraryAgent...")

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
        # 设置温度为 0.5 以平衡行程规划的创造性和准确性
        self.model = ChatOpenAI(
            api_key=api_key,
            temperature=0.5,
            model="qwen-turbo",  # 阿里云通义千问模型
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 API 端点
        )

        logger.info("ItineraryAgent 初始化完成，使用模型: qwen-turbo, 温度: 0.5")

    async def plan_itinerary(self, travel_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步规划详细的旅行行程

        根据用户提供的完整旅行信息（包括目的地、日期、天气、景点推荐、个人偏好等），
        使用大语言模型生成详细的每日行程安排，包含时间分配、活动建议、交通提示等。
        包含完整的错误处理机制，确保即使 API 调用或 JSON 解析失败也能返回有效结果。

        参数:
            travel_info (Dict[str, Any]): 完整的旅行信息字典，应包含以下字段：
                - destination (str): 旅游目的地
                - start_date (str): 开始日期（YYYY-MM-DD 格式）
                - duration (int): 旅行天数
                - preferences (Dict): 用户偏好设置
                - weather_info (Dict): 天气分析信息（来自 WeatherAgent）
                - attractions (List[Dict]): 景点推荐列表（来自 AttractionAgent）

        返回:
            Dict[str, Any]: 包含行程规划信息的字典，结构如下：
                {
                    "success": bool,              # 操作是否成功
                    "itinerary": Dict,             # 详细行程信息
                    "destination": str,            # 目的地名称
                    "duration": int,               # 旅行天数
                    "message": str,                # 状态消息
                    "error_code": str,             # 错误代码（可选，仅在失败时存在）
                    "timestamp": str               # 时间戳
                }

        异常处理:
            - 网络请求失败：捕获异常并返回空行程结果
            - JSON 解析失败：尝试修复或使用默认值
            - 模型调用超时：返回超时提示信息
            - 输入数据不完整：提供合理的默认值继续执行

        使用示例:
            >>> agent = ItineraryAgent()
            >>> travel_info = {
            ...     "destination": "杭州",
            ...     "start_date": "2025-04-15",
            ...     "duration": 3,
            ...     "preferences": {"interest": "自然风景"},
            ...     "weather_info": {...},
            ...     "attractions": [...]
            ... }
            >>> result = await agent.plan_itinerary(travel_info)
        """
        # 记录请求开始时间
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

        # 提取基本信息用于日志
        destination = travel_info.get("destination", "未知目的地") if isinstance(travel_info, dict) else "未知目的地"
        duration = travel_info.get("duration", 0) if isinstance(travel_info, dict) else 0

        logger.info("=" * 60)
        logger.info(f"[行程规划] 开始规划 - 目的地: {destination}, 时长: {duration}天")

        if isinstance(travel_info, dict):
            attractions_count = len(travel_info.get("attractions", []))
            has_weather = bool(travel_info.get("weather_info"))
            logger.info(f"  可用景点: {attractions_count}个")
            logger.info(f"  天气信息: {'有' if has_weather else '无'}")
            preferences = travel_info.get("preferences", {})
            if preferences:
                logger.info(f"  用户偏好: {preferences}")

        logger.info("=" * 60)

        try:
            # 第一步：验证输入参数
            if not isinstance(travel_info, dict):
                logger.error("旅行信息必须是字典类型")
                return self._create_error_response(
                    destination=destination,
                    duration=duration,
                    error_message="旅行信息格式错误，必须为字典类型",
                    error_code="INVALID_INPUT_TYPE",
                    timestamp=timestamp
                )

            if not destination or destination == "未知目的地":
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

            # 第二步：从 travel_info 中提取关键信息，设置默认值防止 KeyError
            start_date = travel_info.get("start_date", "")
            preferences = travel_info.get("preferences", {})
            weather_info = travel_info.get("weather_info", {})
            attractions = travel_info.get("attractions", [])

            logger.debug(f"提取的旅行信息 - 开始日期: {start_date}, 偏好项数: {len(preferences)}, 景点数: {len(attractions)}")

            # 第三步：构建系统提示词，指定行程规划的格式和要求
            logger.debug("正在构建系统提示词...")
            system_prompt = self._build_system_prompt()

            # 第四步：构建用户消息，整合所有旅行相关信息
            logger.debug("正在构建用户消息...")
            user_message = self._build_user_message(
                destination=destination,
                start_date=start_date,
                duration=duration,
                preferences=preferences,
                weather_info=weather_info,
                attractions=attractions
            )

            # 第五步：异步调用大语言模型进行行程规划
            logger.info("正在调用大语言模型进行行程规划...")
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])

            # 第六步：提取模型返回的内容
            response_content = response.content
            logger.info(f"模型响应接收成功，响应长度: {len(response_content)} 字符")

            # 第七步：尝试解析 JSON 格式的响应
            logger.debug("正在解析模型响应...")
            itinerary_data = self._parse_response(response_content)

            # 计算耗时
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            # 提取行程概览信息
            trip_overview = itinerary_data.get("trip_overview", {})
            title = trip_overview.get("title", "未命名行程")
            theme = trip_overview.get("theme", "未知主题")
            daily_plans = itinerary_data.get("daily_plans", [])

            logger.info(f"✓ 行程规划成功完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"  - 行程标题: {title}")
            logger.info(f"  - 行程主题: {theme}")
            logger.info(f"  - 规划天数: {len(daily_plans)}天")

            # 打印每日行程摘要
            if daily_plans:
                logger.info("  每日行程概览:")
                for day_plan in daily_plans:
                    day_num = day_plan.get("day", 0)
                    day_theme = day_plan.get("theme", "无主题")
                    schedule_count = len(day_plan.get("schedule", []))
                    logger.info(f"    第{day_num}天: {day_theme} ({schedule_count}个活动)")

            # 第八步：返回成功的结果
            return {
                "success": True,
                "itinerary": itinerary_data,
                "destination": destination,
                "duration": duration,
                "message": f"成功为 {destination} 生成了 {duration} 天的详细行程",
                "timestamp": timestamp,
                "processing_time": f"{elapsed_time:.2f}s"
            }

        except TimeoutError as e:
            # 处理超时错误
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"行程规划请求超时（已等待 {elapsed_time:.2f}秒）"
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
            error_msg = f"网络连接失败，无法访问行程规划服务"
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
            error_msg = f"行程规划过程中发生未知错误（已运行 {elapsed_time:.2f}秒）"
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
            "itinerary": self._get_default_itinerary(),
            "destination": destination,
            "duration": duration,
            "message": error_message,
            "error_code": error_code,
            "timestamp": timestamp,
            "suggestion": "请稍后重试或检查网络连接，如问题持续存在请联系技术支持"
        }

    def _get_default_itinerary(self) -> Dict[str, Any]:
        """
        获取默认的行程数据结构

        返回:
            Dict[str, Any]: 默认的行程数据结构
        """
        return {
            "trip_overview": {
                "title": "暂无行程规划（服务暂时不可用）",
                "summary": "由于技术原因无法生成详细行程，建议参考其他旅游规划资源或咨询旅行社",
                "theme": "待定",
                "difficulty_level": "适中",
                "estimated_budget": "请咨询当地旅行社获取准确报价"
            },
            "daily_plans": [],
            "practical_tips": {
                "transportation": {
                    "arrival_guide": "建议提前查询航班或火车时刻表，预留充足的前往机场/车站时间",
                    "local_transport": "建议使用公共交通或打车软件，提前了解当地交通卡办理方式",
                    "transport_card": "视具体目的地而定，多数城市支持支付宝/微信乘车码"
                },
                "accommodation": {
                    "recommended_areas": ["市中心交通便利区域", "景区附近", "地铁站周边"],
                    "hotel_types": ["经济型连锁酒店", "舒适型商务酒店", "特色民宿"],
                    "booking_tips": "建议提前3-7天预订，对比多个平台价格，注意查看评价"
                },
                "packing_list": ["身份证/护照", "充电器和充电宝", "常用药品", "雨具", "舒适鞋子"],
                "budget_breakdown": {
                    "accommodation": "200-500元/晚（根据城市等级浮动）",
                    "meals": "100-200元/天",
                    "tickets": "视景点而定，建议预留200-500元",
                    "transport": "50-100元/天",
                    "shopping": "根据个人需求，建议预留500-1000元"
                },
                "safety_notes": [
                    "注意人身和财产安全，保管好重要证件",
                    "遵守当地法律法规和风俗习惯",
                    "购买旅游意外保险",
                    "保存紧急联系方式"
                ],
                "emergency_contacts": ["报警电话: 110", "急救电话: 120", "旅游投诉热线: 12301"]
            },
            "flexible_options": {
                "rainy_day_plan": ["参观博物馆或美术馆", "逛购物中心", "体验当地特色美食", "SPA放松"],
                "extra_time_activities": ["自由活动探索小巷", "购买纪念品", "咖啡馆休息", "拍摄城市风光"],
                "skip_if_tired": ["次要景点", "购物环节", "夜间活动", "爬山项目"]
            },
            "overall_rating": 3
        }

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        设计详细的指令来指导大语言模型如何根据目的地、天气、景点和用户偏好
        生成合理的旅行行程安排。指定输出格式为 JSON，确保返回的数据结构化和易于解析。

        返回:
            str: 格式化的系统提示词字符串

        提示词设计要点:
            1. 明确角色定位：专业的旅行规划师
            2. 指定输出格式：严格的 JSON 结构，包含每日详细安排
            3. 定义字段要求：每天必须包含时间段、活动、地点、时长等
            4. 提供规划原则：考虑天气、体力、交通、用餐等因素
            5. 强调实用性：提供具体的时间建议和实用贴士
            6. 保持灵活性：根据天气和用户偏好调整行程
        """
        prompt = """你是一个专业的旅行规划师助手，擅长根据目的地、天气条件、景点信息和用户偏好生成详细的旅行行程安排。

你的任务是综合分析所有可用信息，为用户制定合理、有趣且实用的旅行计划。

【输出格式要求】
你必须以 JSON 格式返回结果，严格遵循以下结构：
{
    "trip_overview": {
        "title": "行程标题（如：杭州3日深度游）",
        "summary": "整体行程概述（100-150字，说明行程特色和亮点）",
        "theme": "行程主题（如：自然风光之旅、文化探索之旅、美食体验之旅等）",
        "difficulty_level": "行程强度等级（轻松/适中/紧凑）",
        "estimated_budget": "预估总预算范围（如：1500-2500元/人，不含大交通）"
    },
    "daily_plans": [
        {
            "day": 第几天（数字，从1开始）,
            "date": "具体日期（YYYY-MM-DD格式）",
            "theme": "当日主题（如：西湖经典游览、灵隐寺文化之旅等）",
            "weather_note": "天气提示（根据天气信息给出当日注意事项）",
            "schedule": [
                {
                    "time_slot": "时间段（如：08:00-10:00）",
                    "activity": "活动内容详细描述",
                    "location": "具体地点/景点名称",
                    "duration": "预计时长（如：2小时）",
                    "transportation": "交通方式及耗时（如：步行15分钟、打车10分钟等）",
                    "tips": "实用小贴士（如：最佳拍照点、注意事项、门票信息等）",
                    "cost_estimate": "预估费用（如：门票50元、餐饮30元等）"
                }
            ],
            "meals": {
                "breakfast": "早餐建议（餐厅名称或类型）",
                "lunch": "午餐建议",
                "dinner": "晚餐建议",
                "snacks": ["推荐的小吃或特色食品列表"]
            },
            "daily_summary": "当日行程总结（50字以内，回顾当天亮点）"
        }
    ],
    "practical_tips": {
        "transportation": {
            "arrival_guide": "到达目的地的交通建议",
            "local_transport": "当地交通方式和推荐（如：地铁、公交、出租车等）",
            "transport_card": "是否需要办理交通卡及办理方式"
        },
        "accommodation": {
            "recommended_areas": ["推荐的住宿区域列表"],
            "hotel_types": ["适合的酒店类型（如：经济型、舒适型、豪华型）"],
            "booking_tips": "预订建议"
        },
        "packing_list": ["必备物品清单（结合天气和活动类型）"],
        "budget_breakdown": {
            "accommodation": "住宿预算范围",
            "meals": "餐饮预算范围",
            "tickets": "门票预算范围",
            "transport": "当地交通预算范围",
            "shopping": "购物预算建议"
        },
        "safety_notes": ["安全注意事项列表"],
        "emergency_contacts": ["紧急联系方式（如：旅游投诉电话、急救电话等）"]
    },
    "flexible_options": {
        "rainy_day_plan": ["雨天备选活动列表"],
        "extra_time_activities": ["如果时间充裕可增加的活动列表"],
        "skip_if_tired": ["如果体力不足可以跳过的项目列表"]
    },
    "overall_rating": 整体行程推荐评分（1-5分）
}

【规划原则】
1. 合理安排时间：
   - 每天主要景点不超过3个，避免过于疲劳
   - 预留充足的交通时间和休息时间
   - 考虑景点开放时间和最佳游览时段

2. 结合天气情况：
   - 晴天优先安排户外景点
   - 雨天准备室内备选方案
   - 极端天气及时调整行程

3. 考虑用户偏好：
   - 根据兴趣类型调整景点顺序和停留时间
   - 尊重用户的特殊需求（如饮食禁忌、行动不便等）
   - 平衡热门景点和小众体验

4. 优化路线设计：
   - 相邻景点安排在同一天，减少往返路程
   - 考虑交通便利性和地理位置
   - 避免重复路线和时间浪费

5. 注重体验质量：
   - 不只是走马观花，留出深度体验时间
   - 融入当地特色美食和文化活动
   - 提供拍照打卡点和独特体验建议

6. 保持灵活性：
   - 提供备选方案应对突发情况
   - 标注可调整的项目和时间段
   - 给出自由活动的建议时间

【注意事项】
- 如果某些信息缺失（如天气、景点），基于常识进行合理推断
- 标注信息来源和不确定性
- 使用中文回复，语气友好专业
- 只返回 JSON 格式内容，不要添加其他解释文字
- 确保推荐的餐厅、活动等真实可行
- 考虑季节因素对行程的影响"""

        return prompt

    def _build_user_message(
            self,
            destination: str,
            start_date: str,
            duration: int,
            preferences: Dict[str, Any],
            weather_info: Dict[str, Any],
            attractions: List[Dict[str, Any]]
    ) -> str:
        """
        构建用户消息

        将用户的旅行信息、天气信息、景点推荐和偏好格式化为清晰的消息文本，
        供大语言模型综合分析并生成行程规划。

        参数:
            destination (str): 目的地名称
            start_date (str): 开始日期（YYYY-MM-DD 格式）
            duration (int): 旅行天数
            preferences (Dict[str, Any]): 用户偏好
            weather_info (Dict[str, Any]): 天气分析信息
            attractions (List[Dict[str, Any]]): 景点推荐列表

        返回:
            str: 格式化后的用户消息
        """
        message = f"""请为我制定以下旅行的详细行程：

【基本信息】
- 目的地：{destination}
- 开始日期：{start_date}
- 旅行时长：{duration}天

【天气信息】"""

        # 添加天气信息到上下文中
        if weather_info:
            overall_weather = weather_info.get("overall_weather", {})
            if overall_weather:
                summary = overall_weather.get("summary", "暂无天气概况")
                comfort_level = overall_weather.get("comfort_level", "未知")
                avg_temp = overall_weather.get("average_temperature", "未知")
                conditions = overall_weather.get("weather_conditions", [])

                message += f"\n- 整体概况: {summary}"
                message += f"\n- 平均温度: {avg_temp}"
                message += f"\n- 舒适度: {comfort_level}/5分"
                if conditions:
                    message += f"\n- 主要天气: {', '.join(conditions)}"

            # 添加穿衣建议和必备物品
            recommendations = weather_info.get("travel_recommendations", {})
            if recommendations:
                clothing = recommendations.get("clothing_suggestions", "")
                essential_items = recommendations.get("essential_items", [])
                health_tips = recommendations.get("health_tips", "")

                if clothing:
                    message += f"\n- 穿衣建议: {clothing}"
                if essential_items:
                    message += f"\n- 建议携带: {', '.join(essential_items)}"
                if health_tips:
                    message += f"\n- 健康提示: {health_tips}"

            # 添加最佳出行日期
            best_days = weather_info.get("best_travel_days", [])
            if best_days:
                message += f"\n- 最佳出行日: {', '.join(best_days)}"
        else:
            message += "\n暂无详细天气信息，请基于一般情况规划"

        message += "\n\n【推荐景点】"
        if attractions:
            message += f"\n共推荐了 {len(attractions)} 个景点："
            for idx, attraction in enumerate(attractions, 1):
                name = attraction.get("name", "未知景点")
                attr_type = attraction.get("type", "未知类型")
                recommended_duration = attraction.get("recommended_duration", "未知时长")
                rating = attraction.get("rating", 0)
                highlights = attraction.get("highlights", [])

                message += f"\n{idx}. {name}（{attr_type}）"
                message += f"\n   - 建议游览时长: {recommended_duration}"
                message += f"\n   - 推荐指数: {rating}/5分"
                if highlights:
                    message += f"\n   - 亮点: {'; '.join(highlights[:3])}"
        else:
            message += "\n暂无景点推荐，请根据目的地特色自行推荐并安排行程"

        message += "\n\n【我的偏好】"
        if preferences:
            for key, value in preferences.items():
                message += f"\n- {key}: {value}"
        else:
            message += "\n无特殊偏好，请安排经典的必游行程"

        message += f"\n\n【期望】请根据以上信息，为我制定{duration}天的详细行程安排，包括每天的具体活动、时间安排、用餐建议、交通提示等，并以 JSON 格式返回结果。"

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
        return self._get_default_itinerary()

