"""
企业技术支持智研知识库（PRD-KB）—— Web API 启动入口

使用 Uvicorn 启动 FastAPI 服务，对外提供 RESTful 对话接口与流式 SSE 推送。
启动方式：
    python run_api.py
"""
import sys
from pathlib import Path

project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)

import uvicorn
from api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)