from dotenv import load_dotenv
from notion_client import Client
from mcp.server.fastmcp import FastMCP
import os

load_dotenv()

mcp = FastMCP(
    "NotionAgent",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)

notion = Client(auth=os.getenv("NOTION_TOKEN"))


@mcp.tool()
def get_page(page_id: str):
    page = notion.pages.retrieve(page_id=page_id)
    return page


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
