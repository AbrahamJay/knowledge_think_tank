"""
main_graph.py

一句话说明这个模块是干什么的。
"""
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.process.import_.agent.nodes import (
    node_pdf_to_md,
    node_md_img,
    node_document_split,
    node_item_name_recognition,
    node_bge_embedding,
    node_import_milvus
)
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.state import ImportGraphState

load_dotenv(override=True)

# 1.创建主图构建和编译对象 StateGraph
import_graph_builder = StateGraph(ImportGraphState)
# 2.添加节点
import_graph_builder.add_node(node_entry)
import_graph_builder.add_node(node_pdf_to_md)
import_graph_builder.add_node(node_md_img)
import_graph_builder.add_node(node_document_split)
import_graph_builder.add_node(node_item_name_recognition)
import_graph_builder.add_node(node_bge_embedding)
import_graph_builder.add_node(node_import_milvus)
# 3.添加边
# import_graph_builder.add_edge(START, "node_entry")
import_graph_builder.set_entry_point("node_entry")


# 条件边
def after_node_entry(state: ImportGraphState) -> str:
    # state -> is_md_read_enabled | is_pdf_read_enabled
    if state['is_md_read_enabled']:
        return "node_md_img"
    elif state['is_pdf_read_enabled']:
        return "node_pdf_to_md"
    else:
        return END


import_graph_builder.add_conditional_edges(
    "node_entry",
    after_node_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)
import_graph_builder.add_edge("node_pdf_to_md", "node_md_img")
import_graph_builder.add_edge("node_md_img", "node_document_split")
import_graph_builder.add_edge("node_document_split", "node_item_name_recognition")
import_graph_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
import_graph_builder.add_edge("node_bge_embedding", "node_import_milvus")
import_graph_builder.add_edge("node_import_milvus", END)

# 4.编译主图对象
import_app = import_graph_builder.compile()
