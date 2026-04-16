import streamlit as st
from datetime import datetime
import asyncio
from typing import Dict, Any

# Travel相关导入
from main import TravelPlannerWorkflow
from states import TravelInfo


class TravelPlanningAssistant:
    """
    智能旅行规划助手类

    提供基于LangGraph工作流的智能旅行规划功能，
    包括天气分析、景点推荐和行程规划。

    功能模块:
        1. 旅行需求收集 - 目的地、日期、偏好等信息
        2. 智能行程规划 - 天气分析、景点推荐、行程安排
        3. 结果展示 - 美观的Markdown格式行程单
        4. 聊天历史管理 - 保存和展示对话记录

    属性:
        travel_planner: 旅行规划工作流实例
        messages: 聊天消息历史列表
    """

    def __init__(self):
        """
        初始化旅行规划助手

        步骤说明:
            1. 初始化旅行规划工作流
            2. 设置会话状态变量
            3. 初始化用户界面
        """
        # ========== 第一步：初始化旅行规划组件 ==========
        self.travel_planner = None

        # ========== 第二步：初始化会话状态 ==========
        if 'travel_result' not in st.session_state:
            st.session_state.travel_result = None

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'planning_completed' not in st.session_state:
            st.session_state.planning_completed = False

        # ========== 第三步：初始化界面 ==========
        self.init_ui()

    def init_ui(self):
        """
        初始化用户界面

        设置页面配置、标题和整体布局。
        这是应用启动时的第一个可视化步骤。

        界面元素:
            - 页面标题: "智能旅行规划助手"
            - 页面图标: ✈️
            - 布局: 宽屏模式
            - 主标题和副标题
        """
        # 设置页面配置
        st.set_page_config(
            page_title="智能旅行规划助手",
            layout='wide',
            page_icon='✈️'
        )

        # 显示主标题和副标题
        st.title("✈️ 智能旅行规划助手")
        st.caption("基于LangGraph工作流 | 天气分析 | 景点推荐 | 行程规划 | AI驱动")

    def render_sidebar(self):
        """
        渲染侧边栏

        提供旅行需求输入表单，收集以下信息：
        1. 基本信息 - 目的地、出发日期、旅行天数
        2. 兴趣偏好 - 自然风景、文化古迹等多选
        3. 旅行节奏 - 轻松/适中/紧凑
        4. 预算等级 - 经济/中等/豪华
        5. 其他偏好 - 饮食、特殊要求等

        工作流程:
            用户填写表单 -> 点击生成 -> 调用工作流 -> 显示结果
        """
        with st.sidebar:
            st.header('⚙️ 旅行需求配置')

            with st.form("travel_form"):
                # ===== 基本信息区域 =====
                st.markdown("**📍 基本信息**")

                # 目的地输入
                destination = st.text_input(
                    "旅游目的地",
                    placeholder="例如：杭州、北京、上海",
                    help="请输入您要去的城市或地区"
                )

                # 日期和天数
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "出发日期",
                        value=datetime.now(),
                        help="选择您的出发日期"
                    )
                with col2:
                    duration = st.number_input(
                        "旅行天数",
                        min_value=1,
                        max_value=30,
                        value=3,
                        help="计划旅行的天数"
                    )

                # ===== 偏好设置区域 =====
                st.markdown("**🎯 兴趣偏好**")

                # 兴趣多选
                interest_options = [
                    "自然风景", "文化古迹", "现代建筑",
                    "美食体验", "购物娱乐", "户外运动",
                    "艺术展览", "拍照打卡"
                ]
                selected_interests = st.multiselect(
                    "选择您的兴趣（可多选）",
                    options=interest_options,
                    default=["自然风景"],
                    help="选择您感兴趣的旅游类型"
                )

                # 旅行节奏
                pace = st.select_slider(
                    "旅行节奏",
                    options=["轻松", "适中", "紧凑"],
                    value="适中",
                    help="选择您喜欢的旅行节奏"
                )

                # 预算等级
                budget_level = st.select_slider(
                    "预算等级",
                    options=["经济", "中等", "豪华"],
                    value="中等",
                    help="选择您的预算范围"
                )

                # ===== 其他偏好 =====
                st.markdown("**🍜 其他偏好**")

                food_preference = st.text_input(
                    "饮食偏好",
                    value="喜欢当地特色美食",
                    placeholder="例如：素食、清淡口味、海鲜等"
                )

                special_requirements = st.text_area(
                    "特殊要求",
                    value="无特殊要求",
                    placeholder="例如：带老人小孩、希望有拍照点等",
                    height=80
                )

                # 提交按钮
                submitted = st.form_submit_button(
                    "🚀 生成旅行规划",
                    use_container_width=True,
                    type="primary"
                )

            # 处理表单提交
            if submitted:
                if not destination:
                    st.error("❌ 请输入旅游目的地")
                else:
                    # 构建偏好字典
                    preferences = {
                        "interest": "和".join(selected_interests) if selected_interests else "综合旅游",
                        "pace": pace,
                        "budget_level": budget_level,
                        "food_preference": food_preference,
                        "special_requirements": special_requirements
                    }

                    # 显示加载状态
                    with st.spinner("🤖 AI正在为您规划旅行，请稍候..."):
                        try:
                            # 异步执行旅行规划工作流
                            travel_result = asyncio.run(
                                self.run_travel_planning(
                                    destination=destination,
                                    start_date=start_date.strftime("%Y-%m-%d"),
                                    duration=int(duration),
                                    preferences=preferences
                                )
                            )

                            # 保存结果到会话状态
                            st.session_state.travel_result = travel_result
                            st.session_state.planning_completed = True

                            # 在聊天区显示结果
                            if travel_result.get("success"):
                                # 构造AI回复消息
                                ai_message = self.format_travel_result(travel_result)

                                # 添加到聊天历史
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": f"请为我规划{destination}{duration}天的旅行"
                                })
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": ai_message
                                })

                                st.success("✅ 旅行规划生成成功！")
                                st.rerun()
                            else:
                                st.error(f"❌ 规划失败: {travel_result.get('message', '未知错误')}")

                        except Exception as e:
                            st.error(f"❌ 发生错误: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

            # ===== 通用功能区域 =====
            st.divider()
            st.subheader("🔧 通用功能")

            if st.button("🗑️ 清空聊天记录", use_container_width=True):
                st.session_state.messages = []
                st.session_state.travel_result = None
                st.session_state.planning_completed = False
                st.success('聊天记录已清空')
                st.rerun()

            # 显示使用提示
            st.divider()
            st.info("💡 **使用提示**:\n1. 填写左侧旅行需求\n2. 点击生成规划\n3. 查看详细行程\n4. 可继续提问咨询")

    async def run_travel_planning(
        self,
        destination: str,
        start_date: str,
        duration: int,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行旅行规划工作流

        异步调用TravelPlannerWorkflow，按照以下步骤执行：
        1. 检查天气情况
        2. 根据天气推荐景点（室内/户外）
        3. 生成详细行程安排

        参数:
            destination: 目的地名称
            start_date: 出发日期（YYYY-MM-DD格式）
            duration: 旅行天数
            preferences: 用户偏好字典

        返回:
            Dict包含完整的旅行规划结果
        """
        # 创建旅行规划工作流实例（如果尚未创建）
        if self.travel_planner is None:
            self.travel_planner = TravelPlannerWorkflow()

        # 构建初始状态
        initial_state = TravelInfo(
            destination=destination,
            start_date=start_date,
            duration=duration,
            preferences=preferences,
            weather_info={},
            attractions=[],
            itinerary={}
        )

        # 执行工作流
        final_state = await self.travel_planner.run(initial_state)

        # 提取结果
        result = {
            "success": bool(final_state.itinerary),
            "destination": destination,
            "duration": duration,
            "weather_info": final_state.weather_info,
            "attractions": final_state.attractions,
            "itinerary": final_state.itinerary
        }

        return result

    def format_travel_result(self, travel_result: Dict[str, Any]) -> str:
        """
        格式化旅行规划结果为可读文本

        将JSON格式的旅行规划结果转换为美观的Markdown文本，
        包含行程概览、每日安排、实用贴士等部分。

        参数:
            travel_result: 旅行规划结果字典

        返回:
            str: 格式化后的Markdown文本
        """
        if not travel_result.get("success"):
            return f"❌ 旅行规划失败: {travel_result.get('message', '未知错误')}"

        itinerary = travel_result.get("itinerary", {})
        if not itinerary:
            return "❌ 未生成行程规划"

        # ===== 构建Markdown输出 =====
        output = []

        # 行程概览
        trip_overview = itinerary.get("trip_overview", {})
        output.append("## 📋 行程概览")
        output.append(f"**标题**: {trip_overview.get('title', 'N/A')}")
        output.append(f"**主题**: {trip_overview.get('theme', 'N/A')}")
        output.append(f"**强度**: {trip_overview.get('difficulty_level', 'N/A')}")
        output.append(f"**预算**: {trip_overview.get('estimated_budget', 'N/A')}")
        output.append(f"**评分**: {'⭐' * int(itinerary.get('overall_rating', 0))}/5")
        output.append("")

        # 每日行程
        daily_plans = itinerary.get("daily_plans", [])
        if daily_plans:
            output.append("---")
            output.append("## 📅 每日行程安排")
            output.append("")

            for day_plan in daily_plans:
                day_num = day_plan.get("day", 0)
                day_date = day_plan.get("date", "未知日期")
                day_theme = day_plan.get("theme", "无主题")
                weather_note = day_plan.get("weather_note", "")

                output.append(f"### 【第{day_num}天】{day_date}")
                output.append(f"**主题**: {day_theme}")
                if weather_note:
                    output.append(f"**天气提示**: {weather_note}")
                output.append("")

                # 时间安排
                schedule = day_plan.get("schedule", [])
                if schedule:
                    output.append("**时间安排**:")
                    for activity in schedule:
                        time_slot = activity.get("time_slot", "")
                        activity_name = activity.get("activity", "")
                        location = activity.get("location", "")
                        duration_text = activity.get("duration", "")
                        tips = activity.get("tips", "")

                        output.append(f"- ⏰ **{time_slot}**: {activity_name}")
                        output.append(f"  - 📍 地点: {location}")
                        output.append(f"  - ⏱️ 时长: {duration_text}")
                        if tips:
                            output.append(f"  - 💡 提示: {tips}")
                    output.append("")

                # 用餐建议
                meals = day_plan.get("meals", {})
                if meals:
                    output.append("**用餐建议**:")
                    if meals.get("breakfast"):
                        output.append(f"- 🌅 早餐: {meals['breakfast']}")
                    if meals.get("lunch"):
                        output.append(f"- ☀️ 午餐: {meals['lunch']}")
                    if meals.get("dinner"):
                        output.append(f"- 🌙 晚餐: {meals['dinner']}")
                    output.append("")

                # 当日总结
                daily_summary = day_plan.get("daily_summary", "")
                if daily_summary:
                    output.append(f"**总结**: {daily_summary}")
                    output.append("")

                output.append("---")

        # 实用贴士
        practical_tips = itinerary.get("practical_tips", {})
        if practical_tips:
            output.append("## 💡 实用贴士")
            output.append("")

            # 交通建议
            transportation = practical_tips.get("transportation", {})
            if transportation.get("local_transport"):
                output.append(f"**🚗 当地交通**: {transportation['local_transport']}")

            # 必备物品
            packing_list = practical_tips.get("packing_list", [])
            if packing_list:
                output.append(f"**🎒 必备物品**: {', '.join(packing_list[:5])}")

            # 安全提示
            safety_notes = practical_tips.get("safety_notes", [])
            if safety_notes:
                output.append("**⚠️ 安全提示**:")
                for note in safety_notes[:3]:
                    output.append(f"- {note}")

            output.append("")

        # 灵活选项
        flexible_options = itinerary.get("flexible_options", {})
        if flexible_options:
            output.append("## 🔄 灵活选项")
            rainy_plan = flexible_options.get("rainy_day_plan", [])
            if rainy_plan:
                output.append(f"**☔ 雨天备选**: {', '.join(rainy_plan[:3])}")
            output.append("")

        output.append("---")
        output.append("*祝您旅途愉快！如有其他问题，欢迎继续咨询* 😊")

        return "\n".join(output)

    def render_chat(self):
        """
        渲染聊天界面

        显示旅行规划结果和对话历史。

        聊天流程:
            1. 显示历史消息
            2. 接收用户输入（可选扩展功能）
            3. 显示AI回复
        """
        # 如果没有规划结果，显示欢迎信息
        if not st.session_state.planning_completed:
            st.info("👈 请在左侧填写旅行需求，点击'生成旅行规划'开始规划您的旅程！")
            return

        # 显示聊天历史
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # 用户输入框（可扩展为后续问答功能）
        user_input = st.chat_input("关于这次旅行还有什么问题吗？")

        if user_input:
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(user_input)

            # 保存用户消息
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # 这里可以扩展为基于旅行结果的智能问答
            with st.chat_message("assistant"):
                with st.spinner('🤔 AI思考中...'):
                    try:
                        # 简单的回复（可以扩展为更智能的问答）
                        ai_reply = f"感谢您的提问！关于'{user_input}'，建议您参考上方的详细行程规划。如需更多帮助，请咨询专业旅行社。"

                        # 显示AI回复
                        st.markdown(ai_reply)

                        # 保存AI回复
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ai_reply
                        })

                    except Exception as e:
                        error_msg = f"抱歉，处理您的请求时出现错误: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    def run(self):
        """
        运行应用主循环

        这是应用的入口点，按顺序执行以下步骤：
        1. 渲染侧边栏（旅行需求表单）
        2. 渲染聊天界面（显示规划结果）
        3. 处理用户交互

        注意: Streamlit会在每次用户交互时重新运行整个脚本，
        因此所有状态都需要通过session_state管理。
        """
        # 渲染侧边栏
        self.render_sidebar()

        # 渲染聊天界面
        self.render_chat()


# ========== 应用入口点 ==========
if __name__ == '__main__':
    # 创建应用实例
    app = TravelPlanningAssistant()
    # 运行应用
    app.run()
