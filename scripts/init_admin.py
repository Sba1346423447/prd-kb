"""
初始化管理员账号脚本

用法：python scripts/init_admin.py <用户名> <密码>
示例：python scripts/init_admin.py admin Admin@123

数据库连接参数与后端一致，通过环境变量 DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME 控制。
须先执行 alembic upgrade head 完成建表。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db import SessionLocal  # noqa: E402
from core.models import User  # noqa: E402
from core.security import hash_password  # noqa: E402
from utils.logger import logger  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("用法：python scripts/init_admin.py <用户名> <密码>")

    username, password = sys.argv[1], sys.argv[2]

    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise SystemExit(f"用户 [{username}] 已存在，跳过创建")
        db.add(User(username=username, password_hash=hash_password(password), role="admin"))
        db.commit()

    logger.info(f"管理员账号 [{username}] 创建成功")


if __name__ == "__main__":
    main()
