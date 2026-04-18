def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.data == b"OK"


def test_index(app, client):
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
