from app import config
from app import download_models


def test_download_all_fetches_every_model_in_order():
    calls = []

    def fake_download(repo_id, token):
        calls.append((repo_id, token))
        return f"/cache/{repo_id}"

    paths = download_models.download_all(download=fake_download, token="tok")
    assert calls == [
        (config.MODEL_ID, "tok"),
        (config.PE_T2I_ID, "tok"),
        (config.PE_I2I_ID, "tok"),
    ]
    assert paths == [f"/cache/{m}" for m in download_models.MODEL_IDS]


def test_main_passes_hf_token_from_env(monkeypatch):
    calls = []
    monkeypatch.setattr(download_models, "snapshot_download",
                        lambda repo_id, token: calls.append(token) or repo_id)
    monkeypatch.setenv("HF_TOKEN", "secret")
    download_models.main()
    assert calls == ["secret"] * 3


def test_main_without_token_passes_none(monkeypatch):
    calls = []
    monkeypatch.setattr(download_models, "snapshot_download",
                        lambda repo_id, token: calls.append(token) or repo_id)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    download_models.main()
    assert calls == [None] * 3
