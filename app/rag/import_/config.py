# MinerU 模型版本配置（vlm = 视觉语言模型,适合pdf/图片高精度解析
MINERU_MODEL_VERSION = "vlm"
# MinerU 任务轮询最大超时时间（单位：秒）
MINERU_POLL_TIMEOUT_SECONDS = 600
# MinerU 任务轮询间隔时间（单位：秒）
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒）
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30
# 图片常见的后缀名
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
# 截取图片上下文的字符数
IMAGE_CONTEXT_SUB_CHARS = 100
# 代码块连续未闭合的最大容忍行数，超过则强制退出代码块状态（防止围栏标记缺失导致整篇塌成一个chunk）
CODE_FENCE_UNCLOSED_MAX_LINES = 200
# 拼接chunk标题时最多保留的标题层级数，取最近的层级（层级越深语义越具体）
TITLE_PATH_MAX_LEVELS = 4

# 文本切换最大长度：单个文本最多包含1000个字符
CHUNK_MAX_SIZE = 1000
# 文本切分默认长度：单个文本最多包含600个字符
CHUNK_SIZE = 600
# 文本切分重叠长度：相邻文本切分的重叠字符数
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为断碎片，需尝试合并
CHUNK_MIN = 400

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

# Milvus 向量维度（BGE-M3 稠密向量维度）
MILVUS_VECTOR_DIM = 1024
# Milvus VARCHAR 字段最大长度
MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512

# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5