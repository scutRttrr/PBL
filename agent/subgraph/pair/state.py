from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools

        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )


    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.


    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""

class TLCheckOutput(BaseModel):
    is_info_sufficient: bool = Field(description="ILO 信息是否充足")
    missing_dimensions: list = Field(description="缺失的维度")
    logic: str = Field(description="检测逻辑描述")


@dataclass(kw_only=True)
class TLState(InputState):
    is_info_sufficient: bool = Field(description="ILO 信息是否充足")
    missing_dimensions: list = Field(description="缺失的维度")
    documents: list[str] = Field(description="检索返回的文档")
    logic: str = Field(description="检测逻辑描述")
    ILO: str = Field(description="intended learning outcomes")
    TC: str = Field(description="Teaching Content")
    current_count: int = 0
    question: str =Field(description="主图传入的问题")