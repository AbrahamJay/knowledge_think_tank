# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1

# 向量数据库每路查询的chunk的数量
MILVUS_CHUNK_TOP_K = 5

# 动态截取的最大值和最小值
RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 3

# 检查断崖的参数
RERANK_GAP_RATIO: float = 0.2
RERANK_GAP_ABS: float = 0.2

# reranker输入的最大token数
RERANK_MAX_INPUT_TOKENS: int = 512

# 经验值 算token
RERANK_SUMMARY_CHAR_RATIO: float = 1.3

# 最低压缩的字符数量
RERANK_MIN_SUMMARY_CHARS: int = 50