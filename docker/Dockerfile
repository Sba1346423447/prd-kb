FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --target /app/deps \
    -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.12-slim

WORKDIR /app

# 生产镜像不含 pip 缓存与构建工具，仅拷贝依赖与源码
COPY --from=builder /app/deps /app/deps
COPY . .

ENV PYTHONPATH=/app:/app/deps \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 启动前先执行数据库迁移（幂等），再拉起 API 服务
# 用 python -m 调用，不依赖 --target 安装未链接进 PATH 的 console scripts
CMD ["sh", "-c", "python -m alembic upgrade head && python run_api.py"]
