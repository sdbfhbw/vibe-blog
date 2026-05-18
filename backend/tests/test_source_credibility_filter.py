from services.blog_generator.services.source_credibility_filter import (
    SourceCredibilityFilter,
)


class _FakeLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, **kwargs):
        return self.response


def _results(count: int):
    return [
        {
            "title": f"Result {index}",
            "url": f"https://example.com/{index}",
            "source": "example",
            "content": "body",
        }
        for index in range(count)
    ]


def test_zero_thresholds_are_respected():
    service = SourceCredibilityFilter(_FakeLLM("[]"), max_results=0, min_score=0)

    assert service.max_results == 0
    assert service.min_score == 0


def test_curate_applies_scores_and_sorts_descending():
    service = SourceCredibilityFilter(
        _FakeLLM(
            '[{"index":2,"total_score":8.4,"authority":8},'
            '{"index":1,"total_score":6.2,"authority":7}]'
        ),
        min_score=5,
    )

    curated = service.curate("query", _results(6), max_results=2)

    assert [item["url"] for item in curated] == [
        "https://example.com/1",
        "https://example.com/0",
    ]
    assert curated[0]["credibility_detail"]["authority"] == 8


def test_parse_response_accepts_wrapped_results():
    parsed = SourceCredibilityFilter._parse_response(
        '```json\n{"results":[{"index":1,"total_score":7.5}]}\n```'
    )

    assert parsed == [{"index": 1, "total_score": 7.5}]
