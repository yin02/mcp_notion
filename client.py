import asyncio
import os
import platform
import re
import time
from typing import Any

import mcp.types as types
from dotenv import load_dotenv
from mcp.client.websocket import websocket_client
from mcp.server.lowlevel import Server
from notion_client import Client

load_dotenv()

server = Server(
    "NotionBridge",
    instructions=(
        "You are a dialogue assistant with STRICT mandatory MCP tool usage. "
        "On EVERY user turn, call import_mcp_context first using the user's utterance, then answer. "
        "For any Notion-related intent (notion/page/document/notes/read/summarize), "
        "you MUST call one of these tools before answering: import_mcp_context, study_notion_notes, "
        "read_notion_page, get_notion_page, list_notion_blocks. "
        "Never claim you cannot access notes unless tool call fails. "
        "你是对话助手，必须严格使用 MCP 工具。每一轮对话先调用 import_mcp_context（传用户原话）再回答。"
    ),
)
notion: Client | None = None

READ_NOTION_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_id": {"type": "string"},
        "query": {"type": "string"},
        "keyword": {"type": "string"},
        "utterance": {"type": "string"},
    },
    "anyOf": [
        {"required": ["page_id"]},
        {"required": ["query"]},
        {"required": ["keyword"]},
        {"required": ["utterance"]},
    ],
    "additionalProperties": False,
}

LIST_NOTION_BLOCKS_SCHEMA = {
    "type": "object",
    "properties": {
        "page_id": {"type": "string"},
        "query": {"type": "string"},
        "keyword": {"type": "string"},
        "utterance": {"type": "string"},
    },
    "anyOf": [
        {"required": ["page_id"]},
        {"required": ["query"]},
        {"required": ["keyword"]},
        {"required": ["utterance"]},
    ],
    "additionalProperties": False,
}


def _extract_rich_text(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    block_payload = block.get(block_type, {}) if block_type else {}
    rich_text = block_payload.get("rich_text", [])
    return "".join(item.get("plain_text", "") for item in rich_text).strip()


def _extract_title(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    for value in properties.values():
        if isinstance(value, dict) and value.get("type") == "title":
            title_items = value.get("title", [])
            title = "".join(item.get("plain_text", "") for item in title_items).strip()
            if title:
                return title
    return "Untitled"


def _tokenize(text: str) -> list[str]:
    parts = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
    return [p for p in parts if p]


def _score_title_match(title: str, query_text: str) -> int:
    title_tokens = set(_tokenize(title))
    query_tokens = _tokenize(query_text)
    score = 0
    for token in query_tokens:
        if token in title_tokens:
            score += 3
        elif token and token in title.lower():
            score += 1
    # Prioritize resume-study intents.
    if ("resume" in query_text.lower() or "简历" in query_text) and ("resume" in title.lower() or "简历" in title):
        score += 4
    return score


def _search_page_by_text(query_text: str) -> tuple[str, str]:
    if notion is None:
        raise RuntimeError("Notion client is not initialized")

    resp = notion.search(
        query=query_text,
        filter={"value": "page", "property": "object"},
        page_size=10,
    )
    results = resp.get("results", [])
    if not results:
        raise ValueError(f"No Notion page found for query: {query_text}")

    best_page: dict[str, Any] | None = None
    best_score = -1
    for page in results:
        title = _extract_title(page)
        score = _score_title_match(title, query_text)
        if score > best_score:
            best_score = score
            best_page = page
    chosen = best_page or results[0]
    chosen_id = chosen.get("id")
    if not chosen_id:
        raise ValueError(f"Found result without page id for query: {query_text}")
    return chosen_id, _extract_title(chosen)


def _resolve_page_id(arguments: dict[str, Any]) -> tuple[str, str | None]:
    page_id = arguments.get("page_id")
    if isinstance(page_id, str) and page_id.strip():
        return page_id.strip(), None

    query_text = arguments.get("query") or arguments.get("keyword") or arguments.get("utterance")
    if isinstance(query_text, str) and query_text.strip():
        resolved_page_id, matched_title = _search_page_by_text(query_text.strip())
        return resolved_page_id, matched_title

    raise ValueError("Please provide page_id or query/keyword")


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        text = _extract_rich_text(block)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _fetch_all_blocks(page_id: str) -> list[dict[str, Any]]:
    if notion is None:
        raise RuntimeError("Notion client is not initialized")

    blocks: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        blocks_resp = notion.blocks.children.list(block_id=page_id, page_size=100, start_cursor=cursor)
        blocks.extend(blocks_resp.get("results", []))
        if not blocks_resp.get("has_more"):
            break
        cursor = blocks_resp.get("next_cursor")
    return blocks


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_notion_page",
            description=(
                "读取 Notion 页面内容（标题+正文纯文本）/ Read Notion page content (title + plain text). "
                "触发词 Trigger words: Notion, 页面, page, 笔记, 文档, 读取内容, summarize page. "
                "参数支持 page_id 或 query/keyword/utterance（自动搜索页面）/ Supports page_id or query/keyword/utterance."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="read_notion_page",
            description=(
                "get_notion_page 的语义别名（同功能）/ Semantic alias of get_notion_page. "
                "触发词 Trigger words: 读取Notion页面, read notion page, 查看页面内容, page content."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="study_notion_notes",
            description=(
                "按自然语言学习意图读取相关 Notion 笔记 / Read relevant Notion notes from natural-language intent. "
                "示例 Example: '我要复习 Pinterest 的简历笔记'."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="import_mcp_context",
            description=(
                "每轮对话先调用本工具导入 Notion 上下文 / MUST be called first each turn to import Notion context. "
                "请传 utterance（用户原话）/ Pass utterance (raw user message)."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="read_page_content",
            description=(
                "read_notion_page 的别名（同功能）/ Alias of read_notion_page (same behavior). "
                "触发词 Trigger words: 读页面, 读取文档, summarize notion doc."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="summarize_notion_page",
            description=(
                "先读取页面内容再供模型总结 / Read page content first for downstream summarization. "
                "触发词 Trigger words: 总结页面, summarize, extract key points."
            ),
            inputSchema=READ_NOTION_PAGE_SCHEMA,
        ),
        types.Tool(
            name="list_notion_blocks",
            description=(
                "列出 Notion 页面 block 结构用于调试 / List Notion block structure for debugging. "
                "触发词 Trigger words: block, 结构, 调试, 类型, why empty."
            ),
            inputSchema=LIST_NOTION_BLOCKS_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if notion is None:
        raise RuntimeError("Notion client is not initialized")

    page_read_aliases = {
        "get_notion_page",
        "read_notion_page",
        "read_page_content",
        "summarize_notion_page",
        "study_notion_notes",
        "import_mcp_context",
    }
    if name not in page_read_aliases | {"list_notion_blocks"}:
        raise ValueError(f"Unknown tool: {name}")

    page_id, matched_title = _resolve_page_id(arguments)
    blocks = _fetch_all_blocks(page_id)

    if name in page_read_aliases:
        page = notion.pages.retrieve(page_id=page_id)
        result = {
            "page_id": page_id,
            "title": _extract_title(page),
            "content": _extract_text_from_blocks(blocks),
        }
        if matched_title:
            result["matched_by_query_title"] = matched_title
        return result

    # Debug-friendly output to inspect the page structure quickly.
    result_blocks: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        text = _extract_rich_text(block)
        result_blocks.append(
            {
                "index": idx,
                "id": block.get("id"),
                "type": block.get("type"),
                "has_children": block.get("has_children", False),
                "text": text,
            }
        )
    return {
        "page_id": page_id,
        "matched_by_query_title": matched_title,
        "count": len(result_blocks),
        "blocks": result_blocks,
    }


def _resolve_endpoint() -> str | None:
    # Keep compatibility with both naming styles.
    return os.getenv("XIAOZHI_WSS") or os.getenv("MCP_ENDPOINT")


def _print_local_context() -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    print(f"[context] local_time={ts}")
    print(f"[context] platform={platform.system()} {platform.release()}")
    print(f"[context] has_NOTION_TOKEN={bool(os.getenv('NOTION_TOKEN'))}")
    print(f"[context] has_endpoint={bool(_resolve_endpoint())}")


async def _run_once(endpoint: str) -> None:
    print(f"Connecting to XiaoZhi MCP: {endpoint}")
    async with websocket_client(endpoint) as (read_stream, write_stream):
        print("Connected transport, waiting for MCP initialization...")
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def main() -> None:
    global notion

    _print_local_context()
    token = os.getenv("NOTION_TOKEN")
    endpoint = _resolve_endpoint()
    if not token:
        raise RuntimeError("Missing NOTION_TOKEN")
    if not endpoint:
        raise RuntimeError("Missing XIAOZHI_WSS or MCP_ENDPOINT")
    notion = Client(auth=token)

    delay_seconds = 1
    while True:
        try:
            await _run_once(endpoint)
            # server.run normally blocks; if it exits, reset backoff and retry.
            delay_seconds = 1
        except Exception as exc:
            print(f"Connection error: {exc}")
            print(f"Retrying in {delay_seconds}s...")
            await asyncio.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 60)


if __name__ == "__main__":
    asyncio.run(main())
