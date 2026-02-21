import asyncio
import os
import platform
import time
from typing import Any

import mcp.types as types
from dotenv import load_dotenv
from mcp.client.websocket import websocket_client
from mcp.server.lowlevel import Server
from notion_client import Client

load_dotenv()

server = Server("NotionBridge")
notion: Client | None = None

GET_NOTION_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_id": {"type": "string"},
    },
    "required": ["page_id"],
    "additionalProperties": False,
}


def _extract_title(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    for value in properties.values():
        if isinstance(value, dict) and value.get("type") == "title":
            title_items = value.get("title", [])
            title = "".join(item.get("plain_text", "") for item in title_items).strip()
            if title:
                return title
    return "Untitled"


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        block_payload = block.get(block_type, {}) if block_type else {}
        rich_text = block_payload.get("rich_text", [])
        text = "".join(item.get("plain_text", "") for item in rich_text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_notion_page",
            description="Get title and plain text content from a Notion page.",
            inputSchema=GET_NOTION_PAGE_SCHEMA,
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if notion is None:
        raise RuntimeError("Notion client is not initialized")

    if name != "get_notion_page":
        raise ValueError(f"Unknown tool: {name}")

    page_id = arguments["page_id"]
    page = notion.pages.retrieve(page_id=page_id)
    blocks: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        blocks_resp = notion.blocks.children.list(block_id=page_id, page_size=100, start_cursor=cursor)
        blocks.extend(blocks_resp.get("results", []))
        if not blocks_resp.get("has_more"):
            break
        cursor = blocks_resp.get("next_cursor")

    return {
        "title": _extract_title(page),
        "content": _extract_text_from_blocks(blocks),
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
