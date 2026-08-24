"""messages.content 扩展为 MEDIUMTEXT，容纳多模态图片 base64 历史

背景：content 原为 TEXT（MySQL 上限 64KB），多模态图片 base64 数据常超
该上限，落库报 DataError(1406)。生产 MySQL 升级为 MEDIUMTEXT（16MB）；
SQLite 无长度限制，不受影响。

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "messages",
        "content",
        existing_type=sa.Text(),
        type_=sa.dialects.mysql.MEDIUMTEXT(),
    )


def downgrade() -> None:
    op.alter_column(
        "messages",
        "content",
        existing_type=sa.dialects.mysql.MEDIUMTEXT(),
        type_=sa.Text(),
    )