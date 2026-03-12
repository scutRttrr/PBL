import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from agent.subgraph.memory.profile import UserProfile
from agent.subgraph.memory.prompt import MODIFY_MEMORY_USER_PROMPT,MODIFY_MEMORY_SYSTEM_PROMPT
from agent.tool.postgresql.CRUD import read_user_profile, save_updated_profile
from config.settings import settings


def update_memory(user_id,context):
    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,
        tags=["get preference"]
    )

    a, b, c = read_user_profile("黄老师")
    old_profile = UserProfile(ContentPrefer=a,FormatPrefer=b,NegativePrefer=c)


    old_profile_json = old_profile.model_dump_json(indent=2)



    modelprofile = model.with_structured_output(UserProfile,method="function_calling")

    result = model.invoke( [
        {"role":"system","content":MODIFY_MEMORY_SYSTEM_PROMPT},
        {"role":"human","content":MODIFY_MEMORY_USER_PROMPT.format(user_profile=old_profile_json,user_feedback="教案要面向尖子生的高难度")},
    ])

    parser = PydanticOutputParser(pydantic_object=UserProfile)

    new_profile = parser.invoke(result.content)

    a = new_profile.ContentPrefer.model_dump_json(indent=2)
    b = new_profile.FormatPrefer.model_dump_json(indent=2)
    c = new_profile.NegativePrefer.model_dump_json(indent=2)

    save_updated_profile("黄老师",a,b,c)

    return a,b,c

update_memory("a","ddd")