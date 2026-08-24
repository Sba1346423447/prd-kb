"""批次4：Docker 配置静态验证脚本（compose 结构 + healthcheck 语法 + Dockerfile 关键行）"""
import ast
import pathlib
import sys

import yaml

root = pathlib.Path(__file__).resolve().parents[1]

# 1) compose YAML 语法 + 结构断言
cfg = yaml.safe_load((root / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
svc = cfg["services"]
assert "mysql" in svc and "rag" in svc
assert svc["rag"]["healthcheck"]["start_period"] == "120s"
assert svc["mysql"]["healthcheck"]["test"][0] == "CMD-SHELL"
assert set(cfg["volumes"]) == {"mysql_data", "app_logs"}
assert svc["rag"]["depends_on"]["mysql"]["condition"] == "service_healthy"
assert svc["rag"]["logging"]["options"]["max-size"] == "10m"
ro = [v for v in svc["rag"]["volumes"] if v.endswith(":ro")]
assert len(ro) == 2, ro
print("compose OK: 2 services / healthchecks / volumes / logging")

# 2) healthcheck 内嵌 python 语句语法合法
hc = svc["rag"]["healthcheck"]["test"][1]
code = hc.split("python -c ")[1].strip().strip('"')
ast.parse(code)
print("healthcheck python OK:", code[:60])

# 3) Dockerfile 关键行核对
df = (root / "docker" / "Dockerfile").read_text(encoding="utf-8")
for kw in [
    "FROM python:3.12-slim AS builder",
    "--target /app/deps",
    "COPY --from=builder",
    "PYTHONPATH=/app:/app/deps",
    "python -m alembic upgrade head",
    "whl/cpu",
]:
    assert kw in df, kw
print("Dockerfile OK: multistage / cpu-torch / module-cmd")
print("ALL DOCKER STATIC CHECKS PASSED")
sys.exit(0)
