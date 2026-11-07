"""
config.py

一句话说明这个模块是干什么的。
"""
# MinerU 模型版本配置（vlm = 视觉语言模型,适合pdf/图片高精度解析
MINERU_MODEL_VERSION = "vlm"
# MinerU 任务轮询最大超时时间（单位：秒）
MINERU_POLL_TIMEOUT_SECONDS = 600
# MinerU 任务轮询间隔时间（单位：秒）
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒）
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30
