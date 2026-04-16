from enum import Enum
from typing import List, Dict
from pydantic import BaseModel


class PlanningState(str, Enum):
    """
    旅游规划工作流的状态枚举

    定义了智能旅游规划系统中的各个处理阶段，用于LangGraph工作流的状态管理。
    每个状态代表规划流程中的一个特定步骤。

    状态说明:
        INITIAL: 初始状态，等待用户输入旅行需求
        COLLECTING_INFO: 正在收集用户的旅行偏好和基本信息
        CHECKING_WEATHER: 正在查询目的地的天气信息
        FINDING_ATTRACTIONS: 正在搜索和推荐旅游景点
        GENERATING_ITINERARY: 正在生成详细的行程安排
        COMPLETE: 规划完成，输出最终结果
        ERROR: 发生错误，需要异常处理

    使用示例:
        >>> current_state = PlanningState.INITIAL
        >>> if current_state == PlanningState.COLLECTING_INFO:
        ...     print("正在收集用户信息")

    注意:
        - 继承str和Enum以便更好地序列化和比较
        - 这些状态将用于LangGraph的状态机流转
    """
    INITIAL = "initial"
    COLLECTING_INFO = "collecting_info"
    CHECKING_WEATHER = "checking_weather"
    FINDING_ATTRACTIONS = "finding_attractions"
    GENERATING_ITINERARY = "generating_itinerary"
    COMPLETE = "complete"
    ERROR = "error"


class TravelInfo(BaseModel):
    """
    旅行信息数据模型

    使用Pydantic定义的旅行相关信息结构，用于在整个旅游规划工作流中
    存储和管理用户的旅行数据、偏好设置以及生成的行程信息。

    该模型作为LangGraph工作流中的核心数据结构，在各个状态节点之间传递
    和累积旅行规划所需的全部信息。

    属性:
        destination (str): 旅行目的地城市或地区名称
        start_date (str): 旅行开始日期，格式为YYYY-MM-DD
        duration (int): 旅行持续天数，以天为单位
        preferences (Dict): 用户旅行偏好设置，包含预算范围、兴趣类型、
                           交通方式等个性化配置信息
        weather_info (Dict): 目的地天气信息，包括温度、降水概率、
                            风力等气象数据
        attractions (List[Dict]): 推荐的旅游景点列表，每个景点包含名称、
                                 描述、开放时间、门票价格等详细信息
        itinerary (Dict): 生成的详细行程安排，包含每日具体活动、
                         时间节点、交通安排等规划内容

    使用示例:
        >>> travel = TravelInfo(
        ...     destination="北京",
        ...     start_date="2025-01-15",
        ...     duration=3,
        ...     attractions=[]
        ... )
        >>> print(travel.destination)
        北京

    注意:
        - 该类继承自Pydantic的BaseModel，自动提供数据验证功能
        - attractions字段为必填项，其他字段都有默认值或可选
        - 所有Dict类型字段建议使用明确的键值结构以提高代码可读性
    """
    destination: str
    start_date: str
    duration: int

    preferences: Dict = {}
    weather_info: Dict = {}
    attractions: List[Dict]
    itinerary: Dict = {}
