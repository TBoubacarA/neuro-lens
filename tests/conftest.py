import pytest


@pytest.fixture(autouse=True)
def _dossier_de_travail_jetable(tmp_path, monkeypatch):
    """Chaque analyse écrit un projet sur le disque : pendant les tests, jamais dans le vrai dossier."""
    monkeypatch.setenv("NEURO_LENS_HOME", str(tmp_path / "neuro-lens"))
