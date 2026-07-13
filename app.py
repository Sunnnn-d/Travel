import streamlit as st
from datetime import datetime
import asyncio
import json
import os
import glob
from typing import Dict, Any, List

# Travel相关导入
from main import TravelPlannerWorkflow
from states import TravelInfo


class SessionManager:
    """
    会话管理类，负责会话的持久化存储和加载

    功能:
        1. 将会话数据保存到JSON文件（覆盖式保存）
        2. 从JSON文件加载会话数据
        3. 列出所有已保存的会话（包括历史快照）
        4. 删除指定会话
        5. 自动保存当前会话

    文件存储结构:
        sessions/
            current_session.json      # 当前活动会话（覆盖式保存）
            history/
                {timestamp}_{destination}.json  # 历史快照
    """

    SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
    CURRENT_SESSION_FILE = os.path.join(SESSIONS_DIR, "current_session.json")
    HISTORY_DIR = os.path.join(SESSIONS_DIR, "history")

    def __init__(self):
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)
        os.makedirs(self.HISTORY_DIR, exist_ok=True)

    def save_session(self, messages: List[Dict], travel_result: Dict = None, planning_completed: bool = False) -> str:
        """
        保存会话到JSON文件

        只保存当前活动会话（覆盖式保存），不自动创建历史快照。
        历史快照只在新建会话或清空聊天时创建。

        参数:
            messages: 聊天消息列表
            travel_result: 旅行规划结果
            planning_completed: 是否已完成规划

        返回:
            保存的文件路径
        """
        # 构建会话数据
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "travel_result": travel_result,
            "planning_completed": planning_completed
        }

        # 覆盖保存当前活动会话
        with open(self.CURRENT_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        return self.CURRENT_SESSION_FILE

    def save_as_history(self, messages: List[Dict], travel_result: Dict = None, planning_completed: bool = False) -> str:
        """
        将当前会话保存为历史快照（用于新建会话或清空聊天前）

        参数:
            messages: 聊天消息列表
            travel_result: 旅行规划结果
            planning_completed: 是否已完成规划

        返回:
            保存的历史快照文件路径
        """
        # 构建会话数据
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "travel_result": travel_result,
            "planning_completed": planning_completed
        }

        # 创建历史快照
        self._create_snapshot(session_data)

        # 返回历史快照文件名用于显示
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = ""
        if travel_result and travel_result.get("destination"):
            destination = travel_result["destination"]
        destination = destination.replace("/", "_").replace("\\", "_").replace(":", "_")[:20]
        return f"{timestamp}_{destination}.json" if destination else f"{timestamp}.json"

    def _create_snapshot(self, session_data: Dict):
        """
        创建会话历史快照

        参数:
            session_data: 会话数据
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = session_data.get("travel_result", {}).get("destination", "")
        
        # 清理文件名中的非法字符
        destination = destination.replace("/", "_").replace("\\", "_").replace(":", "_")[:20]
        filename = f"{timestamp}_{destination}.json" if destination else f"{timestamp}.json"
        filepath = os.path.join(self.HISTORY_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def load_session(self, filepath: str) -> Dict:
        """
        从JSON文件加载会话

        参数:
            filepath: 会话文件路径

        返回:
            会话数据字典
        """
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        return session_data

    def list_sessions(self) -> List[Dict]:
        """
        列出所有已保存的会话

        返回:
            会话列表，包含当前活动会话和历史快照
        """
        sessions = []

        # 首先添加当前活动会话
        if os.path.exists(self.CURRENT_SESSION_FILE):
            try:
                with open(self.CURRENT_SESSION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                sessions.append({
                    "filepath": self.CURRENT_SESSION_FILE,
                    "filename": "current_session.json",
                    "timestamp": data.get("timestamp", ""),
                    "destination": data.get("travel_result", {}).get("destination", ""),
                    "message_count": len(data.get("messages", [])),
                    "planning_completed": data.get("planning_completed", False),
                    "is_current": True
                })
            except Exception:
                pass

        # 添加历史快照
        history_sessions = self._list_history_sessions()
        sessions.extend(history_sessions)

        return sessions

    def delete_session(self, filepath: str) -> bool:
        """
        删除指定会话

        参数:
            filepath: 会话文件路径

        返回:
            是否删除成功
        """
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def get_latest_session(self) -> Dict:
        """
        获取最近保存的会话

        优先加载当前活动会话文件（current_session.json），
        如果不存在则查找历史快照中最新的一个。

        返回:
            最近会话的数据，如果没有则返回None
        """
        # 优先加载当前活动会话
        if os.path.exists(self.CURRENT_SESSION_FILE):
            return self.load_session(self.CURRENT_SESSION_FILE)
        
        # 如果没有当前会话，查找历史快照
        history_sessions = self._list_history_sessions()
        if history_sessions:
            return self.load_session(history_sessions[0]["filepath"])
        
        return None

    def _list_history_sessions(self) -> List[Dict]:
        """
        列出历史快照会话

        返回:
            历史快照列表
        """
        sessions = []
        pattern = os.path.join(self.HISTORY_DIR, "*.json")
        files = sorted(glob.glob(pattern), reverse=True)

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                timestamp = data.get("timestamp", "")
                destination = data.get("travel_result", {}).get("destination", "")
                message_count = len(data.get("messages", []))

                sessions.append({
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "timestamp": timestamp,
                    "destination": destination,
                    "message_count": message_count,
                    "planning_completed": data.get("planning_completed", False),
                    "is_current": False
                })
            except Exception:
                continue

        return sessions


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
        5. 会话持久化 - 支持历史会话的保存和加载

    属性:
        travel_planner: 旅行规划工作流实例
        messages: 聊天消息历史列表
        session_manager: 会话管理器实例
    """

    # 类变量，用于实现单例模式
    _instance = None
    _travel_planner = None
    _semaphore = None
    _session_manager = None

    def __new__(cls):
        """
        实现单例模式，确保整个应用中只有一个 TravelPlanningAssistant 实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化信号量，限制并发请求数量
            cls._semaphore = asyncio.Semaphore(5)  # 最多同时处理5个请求
            # 初始化会话管理器
            cls._session_manager = SessionManager()
        return cls._instance

    def __init__(self):
        """
        初始化旅行规划助手

        步骤说明:
            1. 初始化旅行规划工作流
            2. 加载历史会话（如果存在）
            3. 设置会话状态变量
            4. 初始化用户界面
        """
        # 确保旅行规划工作流实例只创建一次
        if TravelPlanningAssistant._travel_planner is None:
            TravelPlanningAssistant._travel_planner = TravelPlannerWorkflow()

        # ========== 第二步：加载历史会话 ==========
        self.load_last_session()

        # ========== 第三步：初始化会话状态 ==========
        if 'travel_result' not in st.session_state:
            st.session_state.travel_result = None

        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 你好！我是您的智能旅行规划助手。\n\n请在左侧填写您的旅行需求（目的地、日期、偏好等），然后点击「生成旅行规划」按钮，我将为您生成一份详细的旅行方案。\n\n您也可以随时向我提问关于旅行的问题！😊"
                }
            ]

        if 'planning_completed' not in st.session_state:
            st.session_state.planning_completed = False

        # ========== 第四步：初始化界面 ==========
        self.init_ui()

    def load_last_session(self):
        """
        加载最近一次保存的会话

        如果存在历史会话，将其恢复到session_state中，
        实现"退出后再进入继续会话"的功能。

        注意：使用 session_loaded 标志确保只在首次启动时加载一次，
        避免每次Streamlit重新运行时覆盖当前状态。
        """
        if 'session_loaded' in st.session_state:
            return

        latest_session = TravelPlanningAssistant._session_manager.get_latest_session()
        if latest_session:
            # 恢复会话数据
            st.session_state.messages = latest_session.get("messages", [])
            st.session_state.travel_result = latest_session.get("travel_result", None)
            st.session_state.planning_completed = latest_session.get("planning_completed", False)

        # 标记已完成初始加载
        st.session_state.session_loaded = True

    def save_current_session(self):
        """
        保存当前会话到本地文件

        调用SessionManager将会话数据序列化保存，
        确保对话历史不会丢失。
        """
        TravelPlanningAssistant._session_manager.save_session(
            messages=st.session_state.messages,
            travel_result=st.session_state.travel_result,
            planning_completed=st.session_state.planning_completed
        )

    def start_new_session(self):
        """
        开始新会话

        如果当前会话有内容，先保存为历史快照，
        然后重置session_state开始一个新会话。
        """
        # 检查当前会话是否有内容（超过欢迎消息）
        if len(st.session_state.messages) > 1:
            # 将当前会话保存为历史快照
            TravelPlanningAssistant._session_manager.save_as_history(
                messages=st.session_state.messages,
                travel_result=st.session_state.travel_result,
                planning_completed=st.session_state.planning_completed
            )

        # 删除当前活动会话文件
        if os.path.exists(TravelPlanningAssistant._session_manager.CURRENT_SESSION_FILE):
            os.remove(TravelPlanningAssistant._session_manager.CURRENT_SESSION_FILE)

        # 重置session_state
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 你好！我是您的智能旅行规划助手。\n\n请在左侧填写您的旅行需求（目的地、日期、偏好等），然后点击「生成旅行规划」按钮，我将为您生成一份详细的旅行方案。\n\n您也可以随时向我提问关于旅行的问题！😊"
            }
        ]
        st.session_state.travel_result = None
        st.session_state.planning_completed = False
        st.session_state.session_loaded = True

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

                                # 自动保存会话
                                self.save_current_session()

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

            # 新建会话按钮
            if st.button("🆕 新建会话", use_container_width=True):
                self.start_new_session()
                st.success('已创建新会话，原会话已保存到历史')
                st.rerun()

            # 清空聊天记录按钮
            if st.button("🗑️ 清空聊天记录", use_container_width=True):
                st.session_state.messages = []
                st.session_state.travel_result = None
                st.session_state.planning_completed = False
                st.session_state.session_loaded = False
                
                # 删除当前活动会话文件
                if os.path.exists(TravelPlanningAssistant._session_manager.CURRENT_SESSION_FILE):
                    os.remove(TravelPlanningAssistant._session_manager.CURRENT_SESSION_FILE)
                
                # 删除所有历史快照文件
                history_sessions = TravelPlanningAssistant._session_manager._list_history_sessions()
                for session in history_sessions:
                    TravelPlanningAssistant._session_manager.delete_session(session["filepath"])
                
                st.success('聊天记录已清空')
                st.rerun()

            # ===== 历史会话管理 =====
            st.divider()
            st.subheader("📜 历史会话")

            sessions = TravelPlanningAssistant._session_manager.list_sessions()
            if sessions:
                for session in sessions[:5]:
                    # 当前活动会话显示不同的标签
                    session_label = f"📍 {session['destination'] or '未命名会话'} ({session['message_count']}条消息)"
                    if session.get("is_current"):
                        session_label = f"🔴 当前会话 - {session_label}"
                    
                    with st.expander(session_label):
                        st.write(f"创建时间: {session['timestamp']}")
                        if session.get("is_current"):
                            st.info("这是当前正在使用的会话")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📥 加载", key=f"load_{session['filename']}", use_container_width=True):
                                loaded = TravelPlanningAssistant._session_manager.load_session(session["filepath"])
                                if loaded:
                                    st.session_state.messages = loaded.get("messages", [])
                                    st.session_state.travel_result = loaded.get("travel_result", None)
                                    st.session_state.planning_completed = loaded.get("planning_completed", False)
                                    st.success(f"已加载会话: {session['destination']}")
                                    st.rerun()
                        with col2:
                            if not session.get("is_current"):
                                if st.button(f"🗑️ 删除", key=f"delete_{session['filename']}", use_container_width=True):
                                    TravelPlanningAssistant._session_manager.delete_session(session["filepath"])
                                    st.success(f"已删除会话")
                                    st.rerun()
                            else:
                                st.button(f"🗑️ 删除", key=f"delete_{session['filename']}", use_container_width=True, disabled=True, help="当前会话无法删除，请使用'清空聊天记录'")
            else:
                st.write("暂无历史会话")

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
        try:
            # 使用信号量控制并发请求
            async with TravelPlanningAssistant._semaphore:
                print(f"获取到并发控制信号量，当前剩余: {TravelPlanningAssistant._semaphore._value}")
                
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

                # 执行工作流（使用单例实例）
                final_state = await TravelPlanningAssistant._travel_planner.run(initial_state)

                # 提取结果
                result = {
                    "success": bool(final_state.itinerary),
                    "destination": destination,
                    "duration": duration,
                    "weather_info": final_state.weather_info,
                    "attractions": final_state.attractions,
                    "itinerary": final_state.itinerary
                }

                print(f"释放并发控制信号量，当前剩余: {TravelPlanningAssistant._semaphore._value + 1}")
                return result
        except Exception as e:
            # 捕获并返回具体的错误信息
            import traceback
            error_traceback = traceback.format_exc()
            print(f"旅行规划执行错误: {error_traceback}")
            return {
                "success": False,
                "message": f"旅行规划执行失败: {str(e)}",
                "error_details": error_traceback,
                "destination": destination,
                "duration": duration
            }

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
            1. 显示历史消息（累积显示所有对话）
            2. 接收用户输入
            3. 基于行程结果进行智能问答
            4. 显示AI回复
        """
        # 显示聊天历史（累积显示所有对话）
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # 如果没有规划结果，显示引导信息
        if not st.session_state.planning_completed and not st.session_state.messages:
            st.info("👈 请在左侧填写旅行需求，点击'生成旅行规划'开始规划您的旅程！")

        # 用户输入框（始终显示，支持多轮对话）
        user_input = st.chat_input("关于这次旅行有什么问题吗？例如：'第一天午餐推荐什么餐厅？'、'有哪些适合拍照的地方？'")

        if user_input:
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(user_input)

            # 保存用户消息到历史记录
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # 基于旅行结果的智能问答
            with st.chat_message("assistant"):
                with st.spinner('🤔 AI思考中...'):
                    try:
                        # 获取行程结果
                        travel_result = st.session_state.travel_result
                        
                        # 构建上下文信息
                        context = ""
                        if travel_result and travel_result.get("success"):
                            itinerary = travel_result.get("itinerary", {})
                            
                            # 添加行程概览
                            trip_overview = itinerary.get("trip_overview", {})
                            if trip_overview:
                                context += f"行程概览：\n"
                                context += f"- 标题：{trip_overview.get('title', 'N/A')}\n"
                                context += f"- 主题：{trip_overview.get('theme', 'N/A')}\n"
                                context += f"- 强度：{trip_overview.get('difficulty_level', 'N/A')}\n"
                                context += f"- 预算：{trip_overview.get('estimated_budget', 'N/A')}\n\n"
                            
                            # 添加每日行程
                            daily_plans = itinerary.get("daily_plans", [])
                            if daily_plans:
                                context += "每日行程安排：\n"
                                for day_plan in daily_plans:
                                    day_num = day_plan.get("day", 0)
                                    day_date = day_plan.get("date", "未知日期")
                                    day_theme = day_plan.get("theme", "无主题")
                                    context += f"\n第{day_num}天 ({day_date}) - {day_theme}：\n"
                                    
                                    # 添加时间安排
                                    schedule = day_plan.get("schedule", [])
                                    if schedule:
                                        context += "  时间安排：\n"
                                        for activity in schedule:
                                            time_slot = activity.get("time_slot", "")
                                            activity_name = activity.get("activity", "")
                                            location = activity.get("location", "")
                                            tips = activity.get("tips", "")
                                            context += f"    - {time_slot}：{activity_name}（{location}）"
                                            if tips:
                                                context += f" - 提示：{tips}"
                                            context += "\n"
                                    
                                    # 添加用餐建议
                                    meals = day_plan.get("meals", {})
                                    if meals:
                                        context += "  用餐建议：\n"
                                        if meals.get("breakfast"):
                                            context += f"    - 早餐：{meals['breakfast']}\n"
                                        if meals.get("lunch"):
                                            context += f"    - 午餐：{meals['lunch']}\n"
                                        if meals.get("dinner"):
                                            context += f"    - 晚餐：{meals['dinner']}\n"
                            
                            # 添加实用贴士
                            practical_tips = itinerary.get("practical_tips", {})
                            if practical_tips:
                                context += "\n实用贴士：\n"
                                if practical_tips.get("transportation", {}).get("local_transport"):
                                    context += f"- 当地交通：{practical_tips['transportation']['local_transport']}\n"
                                if practical_tips.get("packing_list"):
                                    context += f"- 必备物品：{', '.join(practical_tips['packing_list'][:5])}\n"
                                if practical_tips.get("safety_notes"):
                                    context += "- 安全提示：\n"
                                    for note in practical_tips['safety_notes'][:3]:
                                        context += f"  - {note}\n"
                        
                        # 构建提示词
                        prompt = f"你是一个智能旅行助手，根据以下行程信息回答用户的问题：\n\n"
                        prompt += f"【行程信息】\n{context}\n\n"
                        prompt += f"【用户问题】\n{user_input}\n\n"
                        prompt += "请基于上述行程信息，详细回答用户的问题。如果行程信息中没有相关内容，请诚实告知，并提供合理的建议。"
                        
                        # 使用现有的 LLM 模型回答问题
                        from langchain_openai import ChatOpenAI
                        import os
                        
                        # 从环境变量中获取 API 密钥
                        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-140a61693bda414eb1c4e4e2e3996083")
                        
                        # 创建模型实例
                        model = ChatOpenAI(
                            api_key=api_key,
                            temperature=0.3,
                            model="qwen-turbo",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )
                        
                        # 生成回答
                        response = model.invoke(prompt)
                        ai_reply = response.content

                        # 显示AI回复
                        st.markdown(ai_reply)

                        # 保存AI回复到历史记录（累积添加）
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ai_reply
                        })

                        # 自动保存会话
                        self.save_current_session()

                    except Exception as e:
                        error_msg = f"抱歉，处理您的请求时出现错误: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                        # 自动保存会话（即使出错也保存）
                        self.save_current_session()

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
    # 配置日志系统
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建应用实例
    app = TravelPlanningAssistant()
    # 运行应用
    app.run()
