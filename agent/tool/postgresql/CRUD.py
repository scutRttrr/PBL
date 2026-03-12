import psycopg2
from psycopg2.extras import Json
import json

# 1. 数据库连接配置 (根据你的 DBeaver 设置修改)
DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "postgres",
    "user": "gzlzhao",
    "password": "",  # 替换为你安装时设置的密码
    "port": "5432"
}


def init_db():
    """初始化数据库表结构"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 创建用户表，注意 ai_memory_profile 是 JSONB 类型
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        pref_content_style JSONB DEFAULT '{}'::jsonb,
        pref_format_style JSONB DEFAULT '{}'::jsonb,
        negative_constraints JSONB DEFAULT '{}'::jsonb
        
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ 数据库表初始化成功！")


def create_initial_user(username):
    """为新用户创建一个初始化的画像模版"""
    # 这是一个符合我们 Prompt 要求的初始结构
    pref_content_style={
        "tone": {
            "candidates": [
                {
                    "value": "亲和、生动、充满鼓励",
                    "weight": 5.0
                },
                {
                    "value": "专业、严谨、客观",
                    "weight": 1.2,
                }
            ]
        },
        "difficulty_level": {
            "candidates": [
                            {"value":"注重基础概念扫盲",
                            "weight": 8.0},]
        },
        "pbl_methodology": {
            "candidates": [
                {"value":"重度依赖小组合作与动手实验",
                "weight": 3.5,}
            ]
        }
    }

    pref_format_style={
        "length_preference":
        {"candidates":
            [
                {"value":"中等长度，理论框架简略，但学生活动步骤必须提供详细执行指南",
                "weight": 3.5,}
            ]
        },
        "formatting_style":
            {"candidates":
                [
                    {"value": "必须使用 markdown 表格呈现评价量规，多用序号和列表，排斥密集长段落",
                     "weight": 3.5, }
                ]
            },
        "visual_requirements":
            {"candidates":
                [
                    {"value": "每个关键实验步骤都需要附带生成对应的 AI 绘图提示词",
                     "weight": 3.5, }
                ]
            }
    }
    negative_constraints=[
        {
            "issue_description": "不要设计需要课外花费大量金钱购买材料的实验",
            "weight": 8.0,
        },
        {
            "issue_description": "不需要课后书面作业，请完全去除该环节",
            "weight": 5.0,
        }
    ]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, pref_content_style,pref_format_style,negative_constraints) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (username, Json(pref_content_style),Json(pref_format_style),Json(negative_constraints))
        )
        conn.commit()
        print(f"✅ 用户 {username} 初始化画像已存入！")
    except Exception as e:
        print(f"❌ 插入失败: {e}")
    finally:
        cur.close()
        conn.close()

def read_user_profile(username):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    get_query = """
    SELECT pref_content_style,pref_format_style,negative_constraints
    FROM users
    WHERE username = %s
    """

    cur.execute(get_query, (username,))


    conn.commit()
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row

# ---------------------------------------------------------
# 模拟：当 LLM 返回了更新后的 JSON 字符串时，如何存入数据库
# ---------------------------------------------------------
def save_updated_profile(username, pref_content_style, pref_format_style, negative_constraints):
    """将 LLM 处理完的字典存回数据库"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    update_query = """
    UPDATE users 
    SET pref_content_style = %s, pref_format_style = %s, negative_constraints = %s
    WHERE username = %s
    """
    cur.execute(update_query, (Json(pref_content_style),Json(pref_format_style),Json(negative_constraints) ,username))
    conn.commit()
    cur.close()
    conn.close()
    print(f"🚀 用户 {username} 的画像已完成 AI 语义更新！")


if __name__ == "__main__":
    # 第一步：建表
    init_db()

    # 第二步：初始化一个测试用户
    create_initial_user("黄老师")

    #read_user_profile("张老师")

    # 第三步：模拟 LLM 处理后的更新（演示数据）

    #save_updated_profile("张老师", {"aa":"bb","cc":"dd"},"","")