"""LLM-assisted filtering for merged search results."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WEIGHTS = {
    "authority": 0.30,
    "freshness": 0.25,
    "relevance": 0.30,
    "depth": 0.15,
}

DEFAULT_MAX_RESULTS = 10
DEFAULT_MIN_SCORE = 5.0
SKIP_THRESHOLD = 5

SearchResult = Dict[str, Any]
ScoreResult = Dict[str, Any]


class SourceCredibilityFilter:
    """Filter search results by LLM-scored credibility."""

    def __init__(
        self,
        llm_client,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None,
    ):
        self.llm = llm_client
        self.max_results = max_results if max_results is not None else int(
            os.environ.get("SOURCE_CREDIBILITY_MAX_RESULTS", DEFAULT_MAX_RESULTS)
        )
        self.min_score = min_score if min_score is not None else float(
            os.environ.get("SOURCE_CREDIBILITY_MIN_SCORE", DEFAULT_MIN_SCORE)
        )

    def curate(
        self,
        query: str,
        search_results: List[SearchResult],
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Score and filter merged search results, with graceful fallback."""
        if not search_results:
            return []
        if len(search_results) <= SKIP_THRESHOLD:
            logger.info(
                "Only %s search results available; skipping credibility scoring",
                len(search_results),
            )
            return search_results

        effective_max = self.max_results if max_results is None else max_results
        try:
            response = self.llm.chat(
                messages=[
                    {
                        "role": "user",
                        "content": self._build_prompt(query, search_results, effective_max),
                    }
                ],
                caller="source_credibility_filter",
            )
            if not response:
                logger.warning("Credibility scoring returned an empty response; using raw results")
                return search_results

            scores = self._parse_response(response)
            if not scores:
                logger.warning("Credibility scoring response was not usable; using raw results")
                return search_results

            filtered = self._apply_scores(search_results, scores)
            filtered.sort(
                key=lambda item: item.get("credibility_score", 0),
                reverse=True,
            )
            result = filtered[:effective_max]
            logger.info(
                "Credibility filter reduced results from %s to %s",
                len(search_results),
                len(result),
            )
            return result
        except Exception as exc:
            logger.error("Credibility filtering failed; using raw results: %s", exc)
            return search_results

    def _apply_scores(
        self,
        search_results: List[SearchResult],
        scores: List[ScoreResult],
    ) -> List[SearchResult]:
        filtered = []
        for item in scores:
            idx = int(item.get("index", 0) or 0) - 1
            total_score = float(item.get("total_score", 0) or 0)
            if not (0 <= idx < len(search_results)) or total_score < self.min_score:
                continue

            result = search_results[idx].copy()
            result["credibility_score"] = total_score
            result["credibility_detail"] = {
                "authority": item.get("authority", 0),
                "freshness": item.get("freshness", 0),
                "relevance": item.get("relevance", 0),
                "depth": item.get("depth", 0),
                "reason": item.get("reason", ""),
            }
            filtered.append(result)
        return filtered

    def _build_prompt(
        self,
        query: str,
        results: List[SearchResult],
        max_results: int,
    ) -> str:
        items = []
        for index, result in enumerate(results, 1):
            content_preview = (result.get("content", "") or "")[:500]
            items.append(
                f"[{index}] Title: {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Source: {result.get('source', 'Unknown')}\n"
                f"Publish date: {result.get('publish_date', 'Unknown')}\n"
                f"Content preview: {content_preview}"
            )

        return (
            "You are evaluating the credibility of search results for a research workflow.\n\n"
            f"Research query: {query}\n\n"
            f"Search results ({len(results)} total):\n"
            + "\n---\n".join(items)
            + "\n\n"
            "Score each result from 1 to 10 for:\n"
            "1. authority\n"
            "2. freshness\n"
            "3. relevance\n"
            "4. depth\n\n"
            f"Return at most {max_results} results as JSON only, sorted by total_score descending.\n"
            '[{"index":1,"authority":8,"freshness":9,"relevance":10,"depth":7,'
            '"total_score":8.6,"reason":"one sentence"}]\n\n'
            "Use total_score = authority*0.30 + freshness*0.25 + "
            "relevance*0.30 + depth*0.15."
        )

    @staticmethod
    def _parse_response(response: str) -> List[ScoreResult]:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            return parsed["results"]
        return []
