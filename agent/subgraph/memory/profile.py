from pydantic import BaseModel, Field, RootModel
from typing import List


# ==========================================
# 1. 最底层的实体模型：备选项
# ==========================================
class Candidate(BaseModel):
    value: str = Field(description="具体的偏好内容描述（如：亲和、生动、充满鼓励）")
    weight: float = Field(default=1.0, description="该偏好的当前权重值")


# ==========================================
# 2. 维度模型：包含备选列表，并自带“裁判”逻辑
# ==========================================
class PreferenceDimension(BaseModel):
    candidates: List[Candidate] = Field(default_factory=list, description="该维度下的所有备选偏好列表")

    @property
    def current_best(self) -> str:
        """
        🚀 核心魔法：随时随地获取当前权重最高的偏好值。
        在组装教案生成的 Prompt 时，直接调用 this_obj.current_best 即可。
        """
        if not self.candidates:
            return "无特殊要求"  # 如果列表为空的兜底方案

        # 自动找出 weight 最大的那个 Candidate，并返回它的 value
        best_candidate = max(self.candidates, key=lambda x: x.weight)
        return best_candidate.value


# ==========================================
# 3. 顶层画像模型：组合各个维度
# ==========================================
class PrefContentStyle(BaseModel):
    tone: PreferenceDimension = Field(description="语气风格偏好")
    difficulty_level: PreferenceDimension = Field(description="认知难度偏好")
    pbl_methodology: PreferenceDimension = Field(description="PBL教学法偏好")

class PrefFormatStyle(BaseModel):
    length_preference: PreferenceDimension = Field(description="内容长度偏好")
    formatting_style: PreferenceDimension = Field(description="排版格式偏好")
    visual_requirements: PreferenceDimension = Field(description="视觉效果要求")

# 1. 确保有这个雷区实体模型
class NegativeConstraint(BaseModel):
    issue_description: str
    weight: float

# 2. 🌟 必须改成 List[NegativeConstraint]
class NegativeConstraints(RootModel):

    @property
    def active_rules(self) -> List[str]:
        """
        🚀 只提取权重 >= 5.0 的核心红线
        """
        return [
            f"- {c.issue_description}"
            for c in self.root
            if c.weight >= 5.0
        ]
    root: List[NegativeConstraint] = Field(default_factory=list)

    # 这样你依然可以定义好用的迭代方法
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class UserProfile(BaseModel):
    ContentPrefer:PrefContentStyle= Field(description="内容偏好")
    FormatPrefer: PrefFormatStyle = Field(description="格式偏好")
    NegativePrefer: NegativeConstraints = Field(description="厌恶内容")