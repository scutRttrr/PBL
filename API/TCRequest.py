from pydantic import BaseModel, Field
from typing import List, Optional


class TCRequest(BaseModel):
    # 必填项：老师输入的 ILO 列表
    questions: List[str] = Field(..., description="教学目标(ILO)列表", min_items=1)

    # 选填项：班级信息（用于检索过滤或教案个性化）
    class_info: Optional[str] = Field(None, description="班级/学生背景信息")

    # 选填项：课程名称
    course_info: Optional[str] = Field(None, description="课程名称")
    #
    # # 功能项：生成模式
    # # new: 全新生成
    # # refine: 基于历史缓存微调
    # mode: str = Field("new", pattern="^(new|refine)$", description="生成模式：new(新建) 或 refine(微调)")
    #
    # # 如果有特定的 thread_id（用于 LangGraph 记忆追踪）
    # thread_id: Optional[str] = Field(None, description="对话线程ID，用于保持多轮修改的连贯性")
    #
    # class Config:
    #     # 示例数据，方便在 FastAPI 的 /docs 页面直接测试
    #     json_schema_extra = {
    #         "example": {
    #             "questions": ["描述高血压的病理生理", "列举常用的降压药分类"],
    #             "class_info": "临床医学三年级",
    #             "course_info": "内科学",
    #             "mode": "new"
    #         }
    #     }