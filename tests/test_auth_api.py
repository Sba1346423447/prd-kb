"""
批次3：认证接口测试

覆盖登录成败、令牌解析与权限位返回，使用 SQLite 内存库预置用户。
"""


class TestLogin:
    def test_login_success_returns_token_and_role(self, client, users):
        resp = client.post("/auth/login", json={"username": "alice", "password": "alice-pwd"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["role"] == "user"
        assert data["token_type"] == "bearer"
        assert data["access_token"]

    def test_login_wrong_password_rejected(self, client):
        resp = client.post("/auth/login", json={"username": "alice", "password": "bad"})
        assert resp.status_code == 401

    def test_login_unknown_user_rejected_with_same_message(self, client):
        """不存在的用户与密码错误返回相同文案，防账号枚举"""
        resp = client.post("/auth/login", json={"username": "nobody", "password": "x"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "用户名或密码错误"


class TestMe:
    def test_me_without_token_rejected(self, client):
        assert client.get("/auth/me").status_code == 401

    def test_me_with_garbage_token_rejected(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
        assert resp.status_code == 401

    def test_me_returns_user_permissions(self, client, tokens):
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['alice']}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert "chat" in data["permissions"]
        assert "kb.manage" not in data["permissions"]

    def test_me_admin_has_kb_manage(self, client, tokens):
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['admin']}"})
        assert resp.status_code == 200
        assert "kb.manage" in resp.json()["permissions"]
