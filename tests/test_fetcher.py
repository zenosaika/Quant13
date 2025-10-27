from __future__ import annotations

from unittest.mock import Mock, patch

from src.data import fetcher


def test_fetch_news_scrapes_article_content():
    mock_ticker = Mock()
    mock_ticker.news = [{"title": "Headline", "link": "https://example.com/article"}]

    mock_response = Mock()
    mock_response.text = "<html><body><article><p>Market news body.</p></article></body></html>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status = Mock()

    with patch.object(fetcher, "_create_ticker", return_value=mock_ticker), \
        patch("src.data.fetcher.requests.get", return_value=mock_response):

        articles = fetcher.fetch_news("TEST", limit=1)

    assert articles[0]["scraped_content"].startswith("Market news body")
    summary = articles[0].get("summary")
    assert summary
    assert len(summary) <= 280
