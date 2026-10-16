from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from circuit_lantern.services.hub import CURATED_MODELS, search_models


def model(name, likes=20, age=2):
    return SimpleNamespace(
        id=name,
        downloads=1000,
        likes=likes,
        gated=False,
        last_modified=datetime.now(UTC) - timedelta(days=age),
        created_at=datetime.now(UTC) - timedelta(days=age + 10),
        safetensors=SimpleNamespace(total=135000000),
        library_name="transformers",
    )


def test_recommendations_are_available_without_network_and_cannot_be_mutated():
    with patch("circuit_lantern.services.hub.HfApi") as api:
        results = search_models("")
    api.assert_not_called()
    results[0]["id"] = "changed"
    assert CURATED_MODELS[0]["id"] != "changed"


@pytest.mark.parametrize(
    "sort,expected",
    [
        ("trending", "trending_score"),
        ("likes", "likes"),
        ("downloads", "downloads"),
        ("updated", "last_modified"),
        ("newest", "created_at"),
    ],
)
def test_hub_ranking_is_delegated_to_hugging_face(sort, expected):
    with patch("circuit_lantern.services.hub.HfApi") as api:
        api.return_value.list_models.return_value = [model("author/model")]
        results = search_models("", sort=sort)
    kwargs = api.return_value.list_models.call_args.kwargs
    assert kwargs["sort"] == expected and kwargs["direction"] == -1
    assert kwargs["pipeline_tag"] == "text-generation"
    assert results[0]["likes"] == 20 and results[0]["parameters"] == "135.0M"


def test_time_likes_author_and_limit_filters_keep_hub_order():
    with patch("circuit_lantern.services.hub.HfApi") as api:
        api.return_value.list_models.return_value = [
            model("a/old", age=31),
            model("a/unliked", likes=0),
            model("a/one"),
            model("a/two"),
        ]
        results = search_models("model", sort="likes", days=7, min_likes=10, author="a", limit=1)
    assert [entry["id"] for entry in results] == ["a/one"]
    kwargs = api.return_value.list_models.call_args.kwargs
    assert kwargs["limit"] == 500 and kwargs["author"] == "a"


def test_newest_time_window_uses_creation_date():
    with patch("circuit_lantern.services.hub.HfApi") as api:
        api.return_value.list_models.return_value = [model("a/recently-updated")]
        assert not search_models("", sort="newest", days=7)


@pytest.mark.parametrize(
    "options", [{"sort": "bad"}, {"days": 2}, {"min_likes": -1}, {"limit": 97}]
)
def test_discovery_rejects_invalid_limits(options):
    with pytest.raises(ValueError):
        search_models("", **options)


def test_browser_query_strings_parse_time_windows_correctly():
    from fastapi.testclient import TestClient

    from circuit_lantern.main import app

    client = TestClient(app)
    response = client.get("/api/models?sort=recommended&days=0&min_likes=0&limit=24")
    assert response.status_code == 200
    assert response.json()[0]["id"] == CURATED_MODELS[0]["id"]
    with patch("circuit_lantern.services.hub.HfApi") as api:
        api.return_value.list_models.return_value = [model("a/recent")]
        response = client.get("/api/models?sort=trending&days=7&min_likes=10&author=a")
    assert response.status_code == 200 and response.json()[0]["id"] == "a/recent"
