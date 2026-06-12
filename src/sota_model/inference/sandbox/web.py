"""Allowlisted web tool with offline cache.

Replaces the placeholder `_stub_web_search` and `_stub_web_fetch` from
`inference/tools.py` with a real but conservative implementation:

  - Domain allowlist enforced before any network call
  - On-disk cache keyed by URL+query so repeat calls are deterministic
  - Soft request budgets per session (modelcard 8.8 BrowseComp / DeepSearchQA
    benchmark grading depends on bounded tool budgets)
  - Optional offline mode: when network is forbidden, only cache hits return

The real production stack would proxy through an operator-managed search
service (Brave/Bing/equivalent) — this module models that contract with a
pluggable `searcher` callable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class WebFetchResult:
    url: str
    status_code: int
    content_type: str
    text: str
    truncated: bool
    cached: bool

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class WebSearchResult:
    query: str
    items: list[dict]
    cached: bool

    def to_dict(self) -> dict:
        return self.__dict__


class AllowlistedWebTool:
    """A controlled web tool that the model can drive via tool calls.

    Pass `searcher` to plug in a real search backend; pass `fetcher` to
    plug in a real HTTP client. Without those, the tool degrades to its
    cache, which is deterministic — useful for evals and CI.
    """

    DEFAULT_ALLOW_DOMAINS: tuple[str, ...] = (
        "example.com",
        "wikipedia.org", "en.wikipedia.org",
        # Operators add the production allowlist here.
    )
    DEFAULT_DENY_DOMAINS: tuple[str, ...] = (
        # Modelcard 8.8.1 / 9.2 contamination URL list mirror — never let
        # the model fetch eval text during inference.
        "huggingface.co", "hf.co", "lastexam.ai", "agi.safe.ai",
        "github.com/centerforaisafety/hle", "github.com/supaihq/hle",
    )

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        allow_domains: Optional[tuple[str, ...]] = None,
        deny_domains: Optional[tuple[str, ...]] = None,
        max_text_bytes: int = 32 * 1024,
        max_search_results: int = 10,
        request_budget: int = 32,
        offline: bool = False,
        searcher: Optional[Callable[[str, int], list[dict]]] = None,
        fetcher: Optional[Callable[[str], tuple[int, str, str]]] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_domains = tuple(allow_domains) if allow_domains else self.DEFAULT_ALLOW_DOMAINS
        self.deny_domains = tuple(deny_domains) if deny_domains else self.DEFAULT_DENY_DOMAINS
        self.max_text_bytes = max_text_bytes
        self.max_search_results = max_search_results
        self.request_budget = request_budget
        self.offline = offline
        self.searcher = searcher
        self.fetcher = fetcher

        self._calls = 0

    # --- public API matching the tool signatures used in the registry ---

    def search(self, q: str, n: int = 5) -> list[dict]:
        return self._do_search(q, n).items

    def fetch(self, url: str) -> str:
        return self._do_fetch(url).text

    # --- registry adapters ---

    def make_search_callable(self) -> Callable[..., list[dict]]:
        def _call(q: str, n: int = 5) -> list[dict]:
            return self.search(q, n=n)
        return _call

    def make_fetch_callable(self) -> Callable[..., str]:
        def _call(url: str) -> str:
            return self.fetch(url)
        return _call

    # --- internals ---

    def _do_search(self, q: str, n: int) -> WebSearchResult:
        n = min(max(1, n), self.max_search_results)
        cache_path = self._cache_path("search", q, str(n))
        cached = self._read_cache(cache_path)
        if cached is not None:
            return WebSearchResult(query=q, items=cached, cached=True)
        if self.offline or self.searcher is None:
            return WebSearchResult(
                query=q,
                items=[{
                    "title": "(no result)",
                    "url": "",
                    "snippet": f"web_search offline: no cached results for {q!r}",
                }],
                cached=False,
            )
        self._charge()
        items = self.searcher(q, n)
        items = [it for it in items if self._domain_allowed(it.get("url", ""))]
        items = items[:n]
        self._write_cache(cache_path, items)
        return WebSearchResult(query=q, items=items, cached=False)

    def _do_fetch(self, url: str) -> WebFetchResult:
        if not self._domain_allowed(url):
            return WebFetchResult(
                url=url, status_code=451, content_type="text/plain",
                text=f"domain not allowed: {urllib.parse.urlparse(url).netloc}",
                truncated=False, cached=False,
            )
        cache_path = self._cache_path("fetch", url)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return WebFetchResult(
                url=url, status_code=cached["status_code"],
                content_type=cached["content_type"], text=cached["text"],
                truncated=cached.get("truncated", False), cached=True,
            )
        if self.offline or self.fetcher is None:
            return WebFetchResult(
                url=url, status_code=503, content_type="text/plain",
                text="web_fetch offline: no cached body", truncated=False, cached=False,
            )
        self._charge()
        status, content_type, text = self.fetcher(url)
        truncated = len(text) > self.max_text_bytes
        text = text[: self.max_text_bytes]
        result = {"status_code": status, "content_type": content_type, "text": text, "truncated": truncated}
        self._write_cache(cache_path, result)
        return WebFetchResult(url=url, **result, cached=False)

    def _domain_allowed(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        if not host:
            return False
        if any(deny in host for deny in self.deny_domains):
            return False
        return any(host == d or host.endswith("." + d) for d in self.allow_domains)

    def _charge(self) -> None:
        self._calls += 1
        if self._calls > self.request_budget:
            raise RuntimeError(f"web tool budget exceeded: {self.request_budget} calls")

    def _cache_path(self, kind: str, *parts: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(json.dumps([kind, *parts], sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / kind / f"{h}.json"

    def _read_cache(self, path: Optional[Path]):
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())

    def _write_cache(self, path: Optional[Path], data) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
