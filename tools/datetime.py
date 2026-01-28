# tools/datetime.py
import datetime

def get_current_datetime() -> str:
    """
    获取当前系统时间（权威事实）。

    重要规则：
    - 所有涉及“今天 / 现在 / 去年 / 明年 / 日期 / 星期 / 时间推算”的问题，
      必须先调用此工具。
    - 返回值是唯一可信的当前时间来源，禁止模型自行猜测或修改。

    返回格式：
    - YYYY-MM-DD HH:MM:SS（24小时制）
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
