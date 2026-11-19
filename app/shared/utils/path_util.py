"""
应用主包 / 工具模块中的 path_util 模块，负责承载对应场景的具体实现逻辑。
"""
# app/shared/utils/path_util.py
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps:int = 0)->Path:
    """
    pathlib.Path 提供了 parents 属性，这是一个有序的路径上级目录迭代器，直接通过索引取值就能快速获取「上 N 级目录」，完美解决多层 .parent 繁琐的问题，这也是官方推荐的简化写法！
    核心规则：parents[N] 索引对应「向上的层级数」
    parents[0] → 等价于 .parent（当前路径的上 1 级目录）
    parents[1] → 等价于 .parent.parent（当前路径的上 2 级目录）
    parents[2] → 等价于 .parent.parent.parent（当前路径的上 3 级目录）
    以此类推，parents[N] → 直接获取上 N+1 级目录，索引越⼤，层级越靠上
    :param ps:
    :return:
    """
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    # 第一步：优先读取环境变量（生产环境用）
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # 第二步：加载根目录的.env文件（为了后续逻辑，也可省略）
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # 第三步：递归查找标识（兜底，开发环境用）
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量PROJECT_ROOT未配置")


PROJECT_ROOT = get_project_root(".env")

# 模型快照目录的判定标志文件，transformers 只有在该目录下找到它才会走本地加载
_MODEL_CONFIG_FILE = "config.json"


def resolve_model_path(raw_path: str, env_name: str = "") -> str:
    """
    归一化本地模型路径，兼容 ModelScope / HuggingFace 缓存的层级差异。

    背景：底层 from_pretrained 只有在「传入目录里真的有 config.json」时才认为是本地模型；
    否则会把这个字符串当成 HuggingFace 仓库 id 处理，抛出与真实原因（路径写错）毫无关系的
    HFValidationError 或 "Unrecognized model in ..." 异常，非常难定位。这里提前把目录补齐并校验。

    支持两种写法：
    1. 快照目录本身：.../BAAI--bge-m3/snapshots/master
    2. 模型根目录：.../BAAI--bge-m3（自动向下寻找含 config.json 的 revision 目录）

    :param raw_path: 环境变量中配置的模型路径，可为仓库 id（如 BAAI/bge-m3）或空值
    :param env_name: 环境变量名，仅用于拼装可读的报错信息
    :return: 可直接传给模型加载器的路径字符串；非本地路径时原样返回
    """
    value = (raw_path or "").strip()
    if not value:
        return value

    # 形如 BAAI/bge-m3 的仓库 id 或 http 地址不属于本地目录，交给底层库自行解析
    candidate = Path(value).expanduser()
    looks_local = (
        candidate.is_absolute()
        or value.startswith((".", "~", "/", "\\"))
        or "\\" in value
        or candidate.exists()
    )
    if not looks_local:
        return value

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在{'（配置项：' + env_name + '）' if env_name else ''}：{candidate}"
        )

    # 已经是完整快照目录，直接返回
    if (candidate / _MODEL_CONFIG_FILE).is_file():
        return str(candidate)

    # 只配到了模型根目录：在 snapshots 下挑选最新的 revision 目录
    snapshots_dir = candidate / "snapshots"
    if snapshots_dir.is_dir():
        # revision 目录名不固定（master / v1 / commit 哈希），按修改时间倒序取第一个合法的
        revisions = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for revision in revisions:
            if (revision / _MODEL_CONFIG_FILE).is_file():
                return str(revision)

    raise FileNotFoundError(
        f"模型目录不完整{'（配置项：' + env_name + '）' if env_name else ''}：{candidate} 及其 "
        f"snapshots 子目录中都没有 {_MODEL_CONFIG_FILE}；ModelScope 下载后的完整快照层级为 "
        f".../models/<org>--<name>/snapshots/<revision>，需要配置到最后一层"
    )
