"""
权限位模型模块

权限不是布尔开关而是权限位集合：每个动作对应一个权限位，角色是权限位的集合，
用户挂角色。接口校验统一走 require(Permission.X)，不硬编码角色字符串，
新增功能权限无需改表结构（MaxKB 借鉴点 2.1）。
"""
from enum import Enum
from typing import Set


class Permission(str, Enum):
    """权限位定义：一个动作一个权限位"""
    KB_MANAGE = "kb.manage"             # 知识库管理（上传文档/重建索引）
    CHAT = "chat"                       # 问答对话
    SESSION_MANAGE = "session.manage"   # 会话管理


ROLE_PERMISSIONS = {
    "admin": {Permission.KB_MANAGE, Permission.CHAT, Permission.SESSION_MANAGE},
    "user": {Permission.CHAT, Permission.SESSION_MANAGE},
}


def get_role_permissions(role: str) -> Set[Permission]:
    """查询角色对应的权限位集合，未知角色返回空集（默认拒绝）"""
    return ROLE_PERMISSIONS.get(role, set())
