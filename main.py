import asyncio
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from states import TravelInfo, PlanningState
from agents.weather_agent import WeatherAgent
from agents.attraction_agent import AttractionAgent
from agents.itinerary_agent import ItineraryAgent


class TravelPlannerWorkflow:
    """
    旅行规划工作流类

    使用 LangGraph 管理整个旅行规划流程，协调三个 Agent（天气、景点、行程）
    按照预定顺序执行，并在节点间传递和更新状态。

    工作流程:
        1. 检查天气信息 (check_weather)
        2. 推荐景点 (recommend_attractions)
        3. 规划行程 (plan_itinerary)
    属性:
        weather_agent: 天气分析代理实例
        attraction_agent: 景点推荐代理实例
        itinerary_agent: 行程规划代理实例
        workflow: LangGraph StateGraph 实例
    """

    def __init__(self):
        """
        初始化旅行规划工作流

        步骤说明:
            1. 创建三个 Agent 实例（天气、景点、行程）
            2. 创建 StateGraph 实例，指定状态类型为 TravelInfo
            3. 添加各个处理节点到工作流图中
            4. 设置节点之间的连接关系（边和条件）
            5. 编译工作流图使其可执行

        工作流程改进:
            - 根据天气情况动态选择景点推荐策略
            - 雨天：优先推荐室内景点（博物馆、美术馆、购物中心等）
            - 晴天：优先推荐户外景点（公园、自然风景、户外古迹等）
        """
        # ========== 第一步：初始化所有 Agent ==========
        # 每个 Agent 负责不同的专业领域，通过大语言模型提供智能服务
        print("正在初始化 Agent...")
        self.weather_agent = WeatherAgent()
        self.attraction_agent = AttractionAgent()
        self.itinerary_agent = ItineraryAgent()

        # ========== 第二步：创建 StateGraph 实例 ==========
        # StateGraph 是 LangGraph 的核心组件，用于管理工作流状态
        # 参数 state_schema 指定状态的类型结构（TravelInfo）
        self.workflow = StateGraph(TravelInfo)

        # ========== 第三步：定义各节点处理函数并添加到图中 ==========
        # add_node 方法将处理函数注册为工作流中的一个节点
        # 节点名称用于后续设置连接关系
        self.workflow.add_node("check_weather", self.check_weather)
        self.workflow.add_node("recommend_indoor_attractions", self.recommend_indoor_attractions)
        self.workflow.add_node("recommend_outdoor_attractions", self.recommend_outdoor_attractions)
        self.workflow.add_node("plan_itinerary", self.plan_itinerary)

        # ========== 第四步：设置工作流的入口点 ==========
        # set_entry_point 指定工作流从哪个节点开始执行
        # 这里从 "check_weather" 节点开始
        self.workflow.set_entry_point("check_weather")

        # ========== 第五步：设置条件边（根据天气动态路由）==========
        # 添加条件边函数，根据天气情况决定跳转到哪个节点
        # 如果下雨或天气不佳 -> recommend_indoor_attractions（室内景点）
        # 如果晴天或天气良好 -> recommend_outdoor_attractions（户外景点）
        self.workflow.add_conditional_edges(
            "check_weather",  # 源节点
            self.route_by_weather,  # 条件判断函数
            {
                "indoor": "recommend_indoor_attractions",  # 雨天路由
                "outdoor": "recommend_outdoor_attractions"  # 晴天路由
            }
        )

        # 室内景点和户外景点都汇聚到行程规划节点
        self.workflow.add_edge("recommend_indoor_attractions", "plan_itinerary")
        self.workflow.add_edge("recommend_outdoor_attractions", "plan_itinerary")

        # 行程规划完成后结束工作流
        self.workflow.add_edge("plan_itinerary", END)

        # ========== 第六步：编译工作流 ==========
        # compile 方法将工作流图转换为可执行的应用程序
        # 返回的 app 对象可以调用 ainvoke 方法异步执行工作流
        self.app = self.workflow.compile()

        print("旅行规划工作流初始化完成！")
        print("工作流特点：支持根据天气条件智能选择景点推荐策略")

    def route_by_weather(self, state: TravelInfo) -> str:
        """
        条件路由函数：根据天气情况决定下一步流程

        分析天气信息，判断是否适合户外活动，从而决定推荐室内还是户外景点。

        参数:
            state (TravelInfo): 当前工作流状态，包含天气分析结果

        返回:
            str: 路由键值
                - "indoor": 推荐室内景点（雨天/恶劣天气）
                - "outdoor": 推荐户外景点（晴天/良好天气）

        判断逻辑:
            1. 检查是否有降雨预警或高降雨概率
            2. 检查舒适度评分是否过低
            3. 检查是否有恶劣天气警告（台风、暴雨等）
            4. 综合判断后返回相应的路由键
        """
        print("\n" + "=" * 60)
        print("【天气路由决策】正在分析天气条件...")
        print("=" * 60)

        try:
            # 获取天气信息
            weather_info = state.weather_info

            # 如果没有天气信息，默认推荐户外景点
            if not weather_info:
                print("⚠️  未获取到天气信息，默认推荐户外景点")
                return "outdoor"

            # 提取关键天气指标
            overall_weather = weather_info.get("overall_weather", {})
            comfort_level = overall_weather.get("comfort_level", 3)  # 默认中等舒适度

            # 获取每日天气详情（如果有）
            daily_forecast = weather_info.get("daily_forecast", [])

            # ========== 判断逻辑 ==========

            # 判断标准1：舒适度评分（满分5分）
            # 舒适度低于2.5分，认为天气不佳
            low_comfort = comfort_level < 2.5

            # 判断标准2：检查是否有降雨
            has_rain = False
            rain_probability = 0

            # 检查总体天气描述中是否包含雨
            summary = overall_weather.get("summary", "").lower()
            rain_keywords = ["雨", "rain", "雪", "snow", "冰雹", "hail"]
            for keyword in rain_keywords:
                if keyword in summary:
                    has_rain = True
                    break

            # 检查每日预报中的降雨概率
            if daily_forecast:
                total_rain_prob = 0
                for day in daily_forecast:
                    prob = day.get("rain_probability", 0)
                    if isinstance(prob, str):
                        # 如果是字符串格式如 "70%"，提取数字
                        prob = int(''.join(filter(str.isdigit, prob)) or 0)
                    total_rain_prob += prob

                # 计算平均降雨概率
                avg_rain_prob = total_rain_prob / len(daily_forecast)
                rain_probability = avg_rain_prob

                # 如果平均降雨概率超过50%，认为有雨
                if avg_rain_prob > 50:
                    has_rain = True

            # 判断标准3：检查是否有恶劣天气警告
            weather_warnings = weather_info.get("warnings", [])
            severe_weather_types = ["暴雨", "台风", "大风", "雷电", "冰雹", "暴雪"]
            has_severe_weather = any(
                warning.get("type", "") in severe_weather_types
                for warning in weather_warnings
            )

            # ========== 综合决策 ==========
            print(f"天气分析结果:")
            print(f"  - 舒适度评分: {comfort_level}/5")
            print(f"  - 天气概况: {summary}")
            print(f"  - 平均降雨概率: {rain_probability:.1f}%")
            print(f"  - 检测到降雨: {'是' if has_rain else '否'}")
            print(f"  - 恶劣天气警告: {'是' if has_severe_weather else '否'}")

            # 决策规则：
            # 1. 如果有恶劣天气警告 -> 室内
            # 2. 如果检测到降雨 -> 室内
            # 3. 如果舒适度太低 -> 室内
            # 4. 其他情况 -> 户外
            if has_severe_weather:
                print(f"\n✓ 决策结果: 推荐室内景点（原因：恶劣天气警告）")
                return "indoor"
            elif has_rain:
                print(f"\n✓ 决策结果: 推荐室内景点（原因：有降雨）")
                return "indoor"
            elif low_comfort:
                print(f"\n✓ 决策结果: 推荐室内景点（原因：舒适度较低）")
                return "indoor"
            else:
                print(f"\n✓ 决策结果: 推荐户外景点（原因：天气良好）")
                return "outdoor"

        except Exception as e:
            # 发生错误时，默认推荐户外景点
            print(f"\n⚠️  天气路由决策出错: {str(e)}，默认推荐户外景点")
            return "outdoor"

    async def check_weather(self, state: TravelInfo) -> Dict[str, Any]:
        """
        节点处理函数：检查天气信息

        调用 WeatherAgent 分析目的地的天气情况，并将结果更新到状态中。
        这是工作流的第一个节点，负责获取天气数据供后续节点使用。

        参数:
            state (TravelInfo): 当前工作流状态，包含目的地、日期等基本信息

        返回:
            Dict[str, Any]: 更新后的状态字典，包含天气分析结果

        工作流程:
            1. 从状态中提取目的地、日期、时长等信息
            2. 调用 WeatherAgent.analyze_weather 方法进行天气分析
            3. 将分析结果存储到状态的 weather_info 字段
            4. 打印进度信息
            5. 返回更新后的状态（LangGraph 会自动合并更新）
        """
        print("\n" + "=" * 60)
        print("【步骤 1/3】正在分析天气信息...")
        print("=" * 60)

        try:
            # 从状态中提取必要的参数
            destination = state.destination
            start_date = state.start_date
            duration = state.duration
            preferences = state.preferences

            print(f"目的地: {destination}")
            print(f"出发日期: {start_date}")
            print(f"旅行天数: {duration}天")

            # 调用 WeatherAgent 进行天气分析
            # analyze_weather 是异步方法，需要使用 await 等待结果
            weather_result = await self.weather_agent.analyze_weather(
                destination=destination,
                start_date=start_date,
                duration=duration,
                preferences=preferences
            )

            # 检查天气分析是否成功
            if weather_result.get("success"):
                # 提取天气分析详情
                weather_analysis = weather_result.get("weather_analysis", {})

                # 打印天气概况
                overall_weather = weather_analysis.get("overall_weather", {})
                summary = overall_weather.get("summary", "暂无天气概况")
                comfort_level = overall_weather.get("comfort_level", "未知")
                avg_temp = overall_weather.get("average_temperature", "未知")

                print(f"\n✓ 天气分析成功")
                print(f"  - 天气概况: {summary}")
                print(f"  - 平均温度: {avg_temp}")
                print(f"  - 舒适度: {comfort_level}/5分")

                # 返回更新的状态
                # LangGraph 会将这个字典与原有状态合并
                return {
                    "weather_info": weather_analysis
                }
            else:
                # 天气分析失败，记录错误信息
                error_message = weather_result.get("message", "未知错误")
                print(f"\n✗ 天气分析失败: {error_message}")

                # 返回空字典，保持原有状态不变
                return {}

        except Exception as e:
            # 捕获并处理异常
            print(f"\n✗ 天气检查过程中发生错误: {str(e)}")
            return {}

    async def recommend_indoor_attractions(self, state: TravelInfo) -> Dict[str, Any]:
        """
        节点处理函数：推荐室内景点（雨天备选方案）

        当天气不适合户外活动时，调用 AttractionAgent 推荐室内景点，
        如博物馆、美术馆、购物中心、室内娱乐场所等。

        参数:
            state (TravelInfo): 当前工作流状态，包含目的地、天气信息等

        返回:
            Dict[str, Any]: 更新后的状态字典，包含室内景点推荐列表

        工作流程:
            1. 从状态中提取目的地、天气信息、偏好等
            2. 构建室内景点专用的推荐提示
            3. 调用 AttractionAgent.recommend_attractions 方法
            4. 将推荐结果存储到状态的 attractions 字段
            5. 打印推荐的景点摘要
            6. 返回更新后的状态
        """
        print("\n" + "=" * 60)
        print("【步骤 2/3 - 雨天方案】正在推荐室内景点...")
        print("=" * 60)

        try:
            # 从状态中提取必要的参数
            destination = state.destination
            duration = state.duration
            preferences = state.preferences.copy()  # 复制偏好字典，避免修改原数据
            weather_info = state.weather_info

            print(f"目的地: {destination}")
            print(f"旅行天数: {duration}天")
            print(f"推荐策略: 室内景点优先（避雨方案）")

            # 如果有天气信息，显示天气概况
            if weather_info:
                overall_weather = weather_info.get("overall_weather", {})
                if overall_weather:
                    summary = overall_weather.get("summary", "")
                    print(f"天气状况: {summary}")

            # 修改用户偏好，强调室内活动
            preferences["indoor_preference"] = True
            preferences["attraction_type"] = "室内景点"
            preferences["special_note"] = "由于天气原因，优先推荐室内景点，如博物馆、美术馆、科技馆、购物中心、室内娱乐等"

            # 调用 AttractionAgent 进行景点推荐
            # 传入天气信息和室内偏好，让推荐更加智能化
            attraction_result = await self.attraction_agent.recommend_attractions(
                destination=destination,
                weather_info=weather_info,
                preferences=preferences,
                duration=duration
            )

            # 检查景点推荐是否成功
            if attraction_result.get("success"):
                # 提取景点列表
                attractions = attraction_result.get("attractions", [])
                total_count = len(attractions)
                summary = attraction_result.get("recommendation_summary", "")

                print(f"\n✓ 室内景点推荐成功")
                print(f"  - 推荐景点数量: {total_count}个")
                if summary:
                    print(f"  - 推荐理由: {summary}")

                # 打印每个景点的简要信息
                if attractions:
                    print(f"\n  推荐室内景点列表:")
                    for idx, attraction in enumerate(attractions, 1):
                        name = attraction.get("name", "未知景点")
                        attr_type = attraction.get("type", "未知类型")
                        rating = attraction.get("rating", 0)
                        indoor_tag = attraction.get("indoor", False)
                        indoor_indicator = "🏛️ 室内" if indoor_tag else ""
                        print(f"    {idx}. {name} ({attr_type}) - 推荐指数: {rating}/5 {indoor_indicator}")

                # 返回更新的状态，将景点列表存入状态
                return {
                    "attractions": attractions
                }
            else:
                # 景点推荐失败，记录错误信息
                error_message = attraction_result.get("message", "未知错误")
                print(f"\n✗ 室内景点推荐失败: {error_message}")

                # 返回空列表，避免后续节点出错
                return {
                    "attractions": []
                }

        except Exception as e:
            # 捕获并处理异常
            print(f"\n✗ 室内景点推荐过程中发生错误: {str(e)}")
            return {
                "attractions": []
            }

    async def recommend_outdoor_attractions(self, state: TravelInfo) -> Dict[str, Any]:
        """
        节点处理函数：推荐户外景点（晴天方案）

        当天气适合户外活动时，调用 AttractionAgent 推荐户外景点，
        如公园、自然风景区、户外古迹、登山步道等。

        参数:
            state (TravelInfo): 当前工作流状态，包含目的地、天气信息等

        返回:
            Dict[str, Any]: 更新后的状态字典，包含户外景点推荐列表

        工作流程:
            1. 从状态中提取目的地、天气信息、偏好等
            2. 构建户外景点专用的推荐提示
            3. 调用 AttractionAgent.recommend_attractions 方法
            4. 将推荐结果存储到状态的 attractions 字段
            5. 打印推荐的景点摘要
            6. 返回更新后的状态
        """
        print("\n" + "=" * 60)
        print("【步骤 2/3 - 晴天方案】正在推荐户外景点...")
        print("=" * 60)

        try:
            # 从状态中提取必要的参数
            destination = state.destination
            duration = state.duration
            preferences = state.preferences.copy()  # 复制偏好字典，避免修改原数据
            weather_info = state.weather_info

            print(f"目的地: {destination}")
            print(f"旅行天数: {duration}天")
            print(f"推荐策略: 户外景点优先（充分利用好天气）")

            # 如果有天气信息，显示天气概况
            if weather_info:
                overall_weather = weather_info.get("overall_weather", {})
                if overall_weather:
                    summary = overall_weather.get("summary", "")
                    print(f"天气状况: {summary}")

            # 修改用户偏好，强调户外活动
            preferences["indoor_preference"] = False
            preferences["attraction_type"] = "户外景点"
            preferences["special_note"] = "天气良好，优先推荐户外景点，如公园、自然风景、户外古迹、登山步道、海滨等"

            # 调用 AttractionAgent 进行景点推荐
            # 传入天气信息和户外偏好，让推荐更加智能化
            attraction_result = await self.attraction_agent.recommend_attractions(
                destination=destination,
                weather_info=weather_info,
                preferences=preferences,
                duration=duration
            )

            # 检查景点推荐是否成功
            if attraction_result.get("success"):
                # 提取景点列表
                attractions = attraction_result.get("attractions", [])
                total_count = len(attractions)
                summary = attraction_result.get("recommendation_summary", "")

                print(f"\n✓ 户外景点推荐成功")
                print(f"  - 推荐景点数量: {total_count}个")
                if summary:
                    print(f"  - 推荐理由: {summary}")

                # 打印每个景点的简要信息
                if attractions:
                    print(f"\n  推荐户外景点列表:")
                    for idx, attraction in enumerate(attractions, 1):
                        name = attraction.get("name", "未知景点")
                        attr_type = attraction.get("type", "未知类型")
                        rating = attraction.get("rating", 0)
                        outdoor_tag = attraction.get("outdoor", True)
                        outdoor_indicator = "🌳 户外" if outdoor_tag else ""
                        print(f"    {idx}. {name} ({attr_type}) - 推荐指数: {rating}/5 {outdoor_indicator}")

                # 返回更新的状态，将景点列表存入状态
                return {
                    "attractions": attractions
                }
            else:
                # 景点推荐失败，记录错误信息
                error_message = attraction_result.get("message", "未知错误")
                print(f"\n✗ 户外景点推荐失败: {error_message}")

                # 返回空列表，避免后续节点出错
                return {
                    "attractions": []
                }

        except Exception as e:
            # 捕获并处理异常
            print(f"\n✗ 户外景点推荐过程中发生错误: {str(e)}")
            return {
                "attractions": []
            }

    async def recommend_attractions(self, state: TravelInfo) -> Dict[str, Any]:
        """
        节点处理函数：推荐景点（已废弃，保留作为备用）

        注意：此方法已被 recommend_indoor_attractions 和 recommend_outdoor_attractions
        替代，但保留此方法以防需要回退到无条件推荐模式。

        参数:
            state (TravelInfo): 当前工作流状态，包含目的地、天气信息等

        返回:
            Dict[str, Any]: 更新后的状态字典，包含景点推荐列表
        """
        print("\n" + "=" * 60)
        print("【步骤 2/3】正在推荐景点...")
        print("=" * 60)

        try:
            # 从状态中提取必要的参数
            destination = state.destination
            duration = state.duration
            preferences = state.preferences
            weather_info = state.weather_info

            print(f"目的地: {destination}")
            print(f"旅行天数: {duration}天")

            # 如果有天气信息，显示天气概况
            if weather_info:
                overall_weather = weather_info.get("overall_weather", {})
                if overall_weather:
                    summary = overall_weather.get("summary", "")
                    print(f"天气状况: {summary}")

            # 调用 AttractionAgent 进行景点推荐
            # 传入天气信息，让推荐更加智能化
            attraction_result = await self.attraction_agent.recommend_attractions(
                destination=destination,
                weather_info=weather_info,
                preferences=preferences,
                duration=duration
            )

            # 检查景点推荐是否成功
            if attraction_result.get("success"):
                # 提取景点列表
                attractions = attraction_result.get("attractions", [])
                total_count = len(attractions)
                summary = attraction_result.get("recommendation_summary", "")

                print(f"\n✓ 景点推荐成功")
                print(f"  - 推荐景点数量: {total_count}个")
                if summary:
                    print(f"  - 推荐理由: {summary}")

                # 打印每个景点的简要信息
                if attractions:
                    print(f"\n  推荐景点列表:")
                    for idx, attraction in enumerate(attractions, 1):
                        name = attraction.get("name", "未知景点")
                        attr_type = attraction.get("type", "未知类型")
                        rating = attraction.get("rating", 0)
                        print(f"    {idx}. {name} ({attr_type}) - 推荐指数: {rating}/5")

                # 返回更新的状态，将景点列表存入状态
                return {
                    "attractions": attractions
                }
            else:
                # 景点推荐失败，记录错误信息
                error_message = attraction_result.get("message", "未知错误")
                print(f"\n✗ 景点推荐失败: {error_message}")

                # 返回空列表，避免后续节点出错
                return {
                    "attractions": []
                }

        except Exception as e:
            # 捕获并处理异常
            print(f"\n✗ 景点推荐过程中发生错误: {str(e)}")
            return {
                "attractions": []
            }

    async def plan_itinerary(self, state: TravelInfo) -> Dict[str, Any]:
        """
        节点处理函数：规划行程

        调用 ItineraryAgent 根据目的地、天气、景点和用户偏好生成详细的行程安排。
        这是工作流的最后一个节点，整合前面所有节点的结果生成最终方案。

        参数:
            state (TravelInfo): 当前工作流状态，包含完整的旅行信息

        返回:
            Dict[str, Any]: 更新后的状态字典，包含详细的行程规划

        工作流程:
            1. 从状态中提取所有旅行相关信息
            2. 构建完整的旅行信息字典
            3. 调用 ItineraryAgent.plan_itinerary 方法
            4. 将行程规划结果存储到状态的 itinerary 字段
            5. 打印行程概览
            6. 返回更新后的状态
        """
        print("\n" + "=" * 60)
        print("【步骤 3/3】正在规划详细行程...")
        print("=" * 60)

        try:
            # 从状态中提取所有必要信息
            destination = state.destination
            start_date = state.start_date
            duration = state.duration
            preferences = state.preferences
            weather_info = state.weather_info
            attractions = state.attractions

            print(f"目的地: {destination}")
            print(f"出发日期: {start_date}")
            print(f"旅行天数: {duration}天")
            print(f"已推荐景点数: {len(attractions)}个")

            # 构建完整的旅行信息字典
            # ItineraryAgent 需要这些信息来生成合理的行程
            travel_info = {
                "destination": destination,
                "start_date": start_date,
                "duration": duration,
                "preferences": preferences,
                "weather_info": weather_info,
                "attractions": attractions
            }

            # 调用 ItineraryAgent 进行行程规划
            itinerary_result = await self.itinerary_agent.plan_itinerary(travel_info)

            # 检查行程规划是否成功
            if itinerary_result.get("success"):
                # 提取行程详情
                itinerary = itinerary_result.get("itinerary", {})

                # 打印行程概览
                trip_overview = itinerary.get("trip_overview", {})
                title = trip_overview.get("title", "未命名行程")
                theme = trip_overview.get("theme", "未知主题")
                difficulty = trip_overview.get("difficulty_level", "未知强度")
                budget = trip_overview.get("estimated_budget", "待定")

                print(f"\n✓ 行程规划成功")
                print(f"  - 行程标题: {title}")
                print(f"  - 行程主题: {theme}")
                print(f"  - 行程强度: {difficulty}")
                print(f"  - 预估预算: {budget}")

                # 打印每日行程摘要
                daily_plans = itinerary.get("daily_plans", [])
                if daily_plans:
                    print(f"\n  每日行程安排:")
                    for day_plan in daily_plans:
                        day_num = day_plan.get("day", 0)
                        day_theme = day_plan.get("theme", "无主题")
                        schedule_count = len(day_plan.get("schedule", []))
                        print(f"    第{day_num}天: {day_theme} ({schedule_count}个活动)")

                # 返回更新的状态，将行程规划结果存入状态
                return {
                    "itinerary": itinerary
                }
            else:
                # 行程规划失败，记录错误信息
                error_message = itinerary_result.get("message", "未知错误")
                print(f"\n✗ 行程规划失败: {error_message}")

                # 返回空字典
                return {
                    "itinerary": {}
                }

        except Exception as e:
            # 捕获并处理异常
            print(f"\n✗ 行程规划过程中发生错误: {str(e)}")
            return {
                "itinerary": {}
            }

    async def run(self, initial_state: TravelInfo) -> TravelInfo:
        """
        运行旅行规划工作流

        调用编译后的工作流应用程序，异步执行整个旅行规划流程。
        这是工作流的统一入口点，简化了外部调用。

        参数:
            initial_state (TravelInfo): 初始状态，包含用户输入的旅行需求

        返回:
            TravelInfo: 完成规划后的最终状态，包含所有结果

        工作流程:
            1. 调用工作流的 ainvoke 方法异步执行
            2. 等待所有节点按顺序执行完成
            3. 将返回的字典转换为 TravelInfo 对象
            4. 返回包含完整规划结果的最终状态
        """
        print("\n" + "=" * 60)
        print("开始执行旅行规划工作流")
        print("=" * 60)

        # 调用工作流的 ainvoke 方法
        # ainvoke 是异步方法，会按顺序执行所有节点
        # 参数 initial_state 是初始状态
        # 重要：LangGraph 的 ainvoke 返回的是字典类型，不是 TravelInfo 对象
        final_state_dict = await self.app.ainvoke(initial_state)

        print("\n" + "=" * 60)
        print("旅行规划工作流执行完成！")
        print("=" * 60)

        # 将字典转换为 TravelInfo 对象
        # LangGraph 在工作流执行过程中会更新状态并返回字典
        # 我们需要将其转换回 TravelInfo 对象，以便后续用属性访问
        if isinstance(final_state_dict, dict):
            # 使用字典解包语法创建 TravelInfo 对象
            # ** 操作符会将字典的键值对作为关键字参数传递
            final_state = TravelInfo(**final_state_dict)
        else:
            # 如果已经是 TravelInfo 对象，直接使用
            final_state = final_state_dict

        return final_state


async def main():
    """
    主函数：演示旅行规划工作流的使用

    创建旅行助手实例，设置初始状态，调用工作流执行整个过程，
    并打印最终的规划结果。

    执行步骤:
        1. 创建 TravelPlannerWorkflow 实例
        2. 通过用户交互获取旅行需求（目的地、日期、偏好等）
        3. 调用工作流的 run 方法执行规划
        4. 打印最终的行程结果
    """
    print("=" * 60)
    print("       智能旅行规划系统")
    print("=" * 60)

    # ========== 第一步：创建旅行助手实例 ==========
    # 初始化工作流，这会自动创建所有 Agent 并配置工作流图
    planner = TravelPlannerWorkflow()

    # ========== 第二步：通过用户交互获取旅行需求 ==========
    print("\n请输入您的旅行需求：\n")

    # 获取目的地
    destination = input("📍 旅游目的地（例如：杭州、北京、上海）：").strip()
    while not destination:
        print("⚠️  目的地不能为空！")
        destination = input("📍 旅游目的地（例如：杭州、北京、上海）：").strip()

    # 获取出发日期
    start_date = input("📅 出发日期（格式：YYYY-MM-DD，例如：2025-04-20）：").strip()
    while not start_date:
        print("⚠️  出发日期不能为空！")
        start_date = input("📅 出发日期（格式：YYYY-MM-DD，例如：2025-04-20）：").strip()

    # 获取旅行天数
    duration_input = input("⏱️  旅行天数（例如：3）：").strip()
    while not duration_input:
        print("⚠️  旅行天数不能为空！")
        duration_input = input("⏱️  旅行天数（例如：3）：").strip()

    # 将输入的天数转换为整数，并进行验证
    try:
        duration = int(duration_input)
        if duration <= 0:
            raise ValueError("旅行天数必须大于0")
    except ValueError as e:
        print(f"⚠️  输入无效：{str(e)}，使用默认值 3 天")
        duration = 3

    # 获取兴趣爱好
    print("\n请选择您的兴趣偏好（可多选，用逗号分隔）：")
    print("  1. 自然风景")
    print("  2. 文化古迹")
    print("  3. 现代建筑")
    print("  4. 美食体验")
    print("  5. 购物娱乐")
    print("  6. 户外运动")
    print("  7. 艺术展览")
    print("  8. 拍照打卡")
    interest_options = {
        "1": "自然风景",
        "2": "文化古迹",
        "3": "现代建筑",
        "4": "美食体验",
        "5": "购物娱乐",
        "6": "户外运动",
        "7": "艺术展览",
        "8": "拍照打卡"
    }
    interest_input = input("请输入选项编号（例如：1,2,8）：").strip()
    if interest_input:
        # 解析用户选择的兴趣
        selected_interests = []
        for option in interest_input.split(","):
            option = option.strip()
            if option in interest_options:
                selected_interests.append(interest_options[option])
        interest = "和".join(selected_interests) if selected_interests else "综合旅游"
    else:
        interest = "综合旅游"

    # 获取旅行节奏
    print("\n请选择旅行节奏：")
    print("  1. 轻松 - 悠闲自在，景点较少")
    print("  2. 适中 - 平衡安排，推荐选择")
    print("  3. 紧凑 - 充实丰富，景点较多")
    pace_options = {
        "1": "轻松",
        "2": "适中",
        "3": "紧凑"
    }
    pace_input = input("请输入选项编号（1/2/3，默认为2）：").strip()
    pace = pace_options.get(pace_input, "适中")

    # 获取预算等级
    print("\n请选择预算等级：")
    print("  1. 经济 - 性价比高，节约开支")
    print("  2. 中等 - 舒适体验，合理消费")
    print("  3. 豪华 - 高端享受，品质优先")
    budget_options = {
        "1": "经济",
        "2": "中等",
        "3": "豪华"
    }
    budget_input = input("请输入选项编号（1/2/3，默认为2）：").strip()
    budget_level = budget_options.get(budget_input, "中等")

    # 获取饮食偏好
    food_preference = input("\n🍜 饮食偏好（例如：喜欢当地特色美食、素食、清淡口味）：").strip()
    if not food_preference:
        food_preference = "喜欢当地特色美食"

    # 获取特殊要求
    special_requirements = input("✨ 特殊要求（例如：希望有拍照打卡点、带老人小孩）：").strip()
    if not special_requirements:
        special_requirements = "无特殊要求"

    # 构建用户偏好字典
    preferences = {
        "interest": interest,
        "pace": pace,
        "budget_level": budget_level,
        "food_preference": food_preference,
        "special_requirements": special_requirements
    }

    # 显示用户输入的总结
    print("\n" + "=" * 60)
    print("您输入的旅行需求：")
    print("=" * 60)
    print(f"  📍 目的地: {destination}")
    print(f"  📅 出发日期: {start_date}")
    print(f"  ⏱️  旅行天数: {duration}天")
    print(f"  🎯 兴趣偏好: {interest}")
    print(f"  🚶 旅行节奏: {pace}")
    print(f"  💰 预算等级: {budget_level}")
    print(f"  🍜 饮食偏好: {food_preference}")
    print(f"  ✨ 特殊要求: {special_requirements}")
    print("=" * 60)

    # 确认是否开始规划
    confirm = input("\n是否开始生成旅行规划？（y/n，默认为y）：").strip().lower()
    if confirm == 'n':
        print("已取消旅行规划。")
        return

    # 创建初始状态对象
    initial_state = TravelInfo(
        destination=destination,
        start_date=start_date,
        duration=duration,
        preferences=preferences,
        weather_info={},
        attractions=[],
        itinerary={}
    )

    # ========== 第三步：调用工作流执行规划 ==========
    # 使用 await 等待异步工作流执行完成
    # final_state 包含所有节点的输出结果
    final_state = await planner.run(initial_state)

    # ========== 第四步：打印最终结果 ==========
    print("\n" + "=" * 60)
    print("       旅行规划结果")
    print("=" * 60)

    # 检查是否有行程规划结果
    if final_state.itinerary:
        # 提取行程概览信息
        trip_overview = final_state.itinerary.get("trip_overview", {})
        print(f"\n📋 行程概览:")
        print(f"   标题: {trip_overview.get('title', 'N/A')}")
        print(f"   主题: {trip_overview.get('theme', 'N/A')}")
        print(f"   强度: {trip_overview.get('difficulty_level', 'N/A')}")
        print(f"   预算: {trip_overview.get('estimated_budget', 'N/A')}")
        print(f"   评分: {final_state.itinerary.get('overall_rating', 'N/A')}/5")

        # 打印每日详细行程
        daily_plans = final_state.itinerary.get("daily_plans", [])
        if daily_plans:
            print(f"\n📅 每日行程安排:")
            for day_plan in daily_plans:
                day_num = day_plan.get("day", 0)
                day_date = day_plan.get("date", "未知日期")
                day_theme = day_plan.get("theme", "无主题")
                weather_note = day_plan.get("weather_note", "")

                print(f"\n   【第{day_num}天】{day_date}")
                print(f"   主题: {day_theme}")
                if weather_note:
                    print(f"   天气提示: {weather_note}")

                # 打印当天的活动时间表
                schedule = day_plan.get("schedule", [])
                if schedule:
                    print(f"   时间安排:")
                    for activity in schedule:
                        time_slot = activity.get("time_slot", "")
                        activity_name = activity.get("activity", "")
                        location = activity.get("location", "")
                        duration = activity.get("duration", "")

                        print(f"     ⏰ {time_slot}: {activity_name}")
                        print(f"        📍 地点: {location}")
                        print(f"        ⏱️  时长: {duration}")

                        # 如果有小贴士，也打印出来
                        tips = activity.get("tips", "")
                        if tips:
                            print(f"        💡 提示: {tips}")

                # 打印用餐建议
                meals = day_plan.get("meals", {})
                if meals:
                    print(f"   用餐建议:")
                    breakfast = meals.get("breakfast", "")
                    lunch = meals.get("lunch", "")
                    dinner = meals.get("dinner", "")
                    if breakfast:
                        print(f"     🌅 早餐: {breakfast}")
                    if lunch:
                        print(f"     ☀️  午餐: {lunch}")
                    if dinner:
                        print(f"     🌙 晚餐: {dinner}")

                # 打印当日总结
                daily_summary = day_plan.get("daily_summary", "")
                if daily_summary:
                    print(f"   总结: {daily_summary}")

        # 打印实用贴士
        practical_tips = final_state.itinerary.get("practical_tips", {})
        if practical_tips:
            print(f"\n💡 实用贴士:")

            # 交通建议
            transportation = practical_tips.get("transportation", {})
            if transportation:
                local_transport = transportation.get("local_transport", "")
                if local_transport:
                    print(f"   🚗 当地交通: {local_transport}")

            # 必备物品
            packing_list = practical_tips.get("packing_list", [])
            if packing_list:
                print(f"   🎒 必备物品: {', '.join(packing_list[:5])}")

            # 安全提示
            safety_notes = practical_tips.get("safety_notes", [])
            if safety_notes:
                print(f"   ⚠️  安全提示:")
                for note in safety_notes[:3]:
                    print(f"      - {note}")

        # 打印灵活选项
        flexible_options = final_state.itinerary.get("flexible_options", {})
        if flexible_options:
            print(f"\n🔄 灵活选项:")
            rainy_plan = flexible_options.get("rainy_day_plan", [])
            if rainy_plan:
                print(f"   ☔ 雨天备选: {', '.join(rainy_plan[:3])}")
    else:
        # 如果没有行程结果，显示错误提示
        print("\n❌ 未能生成行程规划，请检查输入信息或稍后重试。")

    print("\n" + "=" * 60)
    print("感谢使用智能旅行规划系统！祝您旅途愉快！")
    print("=" * 60)


# ========== 程序入口 ==========
# 当直接运行此脚本时，执行 main 函数
if __name__ == "__main__":
    # 使用 asyncio.run 运行异步主函数
    # 这会创建一个事件循环并运行 main() 协程
    asyncio.run(main())
