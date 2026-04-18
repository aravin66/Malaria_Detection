from app import parse_mysql_url, resolve_mysql_settings


def test_parse_mysql_url():
    parsed = parse_mysql_url("mysql://root:secret@metro.proxy.rlwy.net:21559/railway")
    assert parsed == {
        "host": "metro.proxy.rlwy.net",
        "port": 21559,
        "user": "root",
        "password": "secret",
        "database": "railway",
    }


def test_resolve_mysql_settings_prefers_mysql_url():
    base_config, database = resolve_mysql_settings(
        {
            "MYSQL_URL": "mysql://root:secret@metro.proxy.rlwy.net:21559/railway",
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "root",
            "MYSQL_PASSWORD": "wrong",
            "MYSQL_DATABASE": "malaria_database",
        }
    )
    assert base_config == {
        "host": "metro.proxy.rlwy.net",
        "port": 21559,
        "user": "root",
        "password": "secret",
    }
    assert database == "railway"


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.data == b"OK"


def test_index(app, client):
    res = client.get("/")
    assert res.status_code == 200


def test_index_does_not_hit_auth_lookup(monkeypatch, client):
    def fail_if_called():
        raise AssertionError("auth lookup should not run for GET /")

    monkeypatch.setattr("app.ensure_auth_storage", fail_if_called)

    res = client.get("/")
    assert res.status_code == 200


def test_head_index_skips_auth_lookup(monkeypatch, client):
    def fail_if_called():
        raise AssertionError("auth lookup should not run for HEAD /")

    monkeypatch.setattr("app.ensure_auth_storage", fail_if_called)

    res = client.head("/")
    assert res.status_code == 200
    assert res.data == b""


def test_form(app, client):
    res = client.get("/form")
    assert res.status_code == 200


def test_result(app, client):
    res = client.get("/result")
    assert res.status_code == 200
