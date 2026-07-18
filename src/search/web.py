"""
Web research module for fetching latest data to ground animations in real-world context.

Pipeline:
  1. LLM generates diverse, targeted search queries from the user's topic
  2. Queries run in parallel against DuckDuckGo and Wikipedia
  3. Top-5 URLs scraped concurrently with httpx + BeautifulSoup
  4. LLM synthesises a structured research brief used as extra context by the code-gen node

All IO is async; the public entry-point is ``research_topic()``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (compatible; ManimResearchBot/1.0; +https://github.com/manim-agent)"
)
_WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"
_SCRAPE_TIMEOUT = 12.0          # seconds per URL
_MAX_PAGE_CHARS = 6_000         # chars kept from each page
_MAX_RETRIES = 2
_CONCURRENCY = 5                 # parallel HTTP requests


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WebSource:
    url: str
    title: str
    snippet: str
    content: str = ""             # extracted body text (trimmed)
    source_type: str = "web"      # "web" | "wikipedia"


@dataclass
class ResearchBrief:
    """Structured output returned to the code-gen node."""
    topic: str
    queries_used: list[str] = field(default_factory=list)
    sources: list[WebSource] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    synthesis: str = ""
    error: Optional[str] = None   # set if research partially/fully failed

    def to_prompt_block(self) -> str:
        """Format the brief as a readable block for the code-gen LLM."""
        if self.error and not self.synthesis:
            return f"[Web research failed: {self.error}]"

        parts = [
            "## Latest Web Research",
            f"**Topic:** {self.topic}",
            f"**Queries used:** {', '.join(self.queries_used)}",
            "",
        ]

        if self.key_facts:
            parts.append("### Key Facts / Recent Data")
            for fact in self.key_facts:
                parts.append(f"- {fact}")
            parts.append("")

        if self.synthesis:
            parts.append("### Research Summary")
            parts.append(self.synthesis)
            parts.append("")

        if self.sources:
            parts.append("### Sources")
            for i, src in enumerate(self.sources, 1):
                parts.append(f"{i}. **{src.title}** — {src.url}")
                if src.snippet:
                    parts.append(f"   > {src.snippet[:200]}")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 1 — LLM-generated search queries
# ---------------------------------------------------------------------------

_QUERY_GEN_PROMPT = """\
You are a research assistant specialising in finding the LATEST, most relevant information for educational animations.

Given the animation topic below, generate {n} diverse search queries that will return up-to-date, factual content useful for creating an accurate and informative animation.

Requirements:
- Mix general overview queries with specific detail queries
- Include at least one "latest 2024 2025" or "recent" query for timely data
- Prefer queries likely to return Wikipedia, news, or authoritative sources
- Keep each query concise (3–8 words)

Topic: {topic}

Respond with ONLY a JSON array of strings, e.g.:
["query one", "query two", "query three"]
"""


def _gen_search_queries(topic: str, llm_chat: Callable, n: int = 4) -> list[str]:
    """Ask the LLM to produce n diverse search queries for the topic."""
    prompt = _QUERY_GEN_PROMPT.format(topic=topic, n=n)
    try:
        raw = llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        if not raw:
            raise ValueError("empty LLM response")

        # Extract JSON array (may be wrapped in markdown fences)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON array found")

        queries: list[str] = json.loads(match.group(0))
        return [str(q).strip() for q in queries if q.strip()][:n]
    except Exception as exc:
        print(f"[WebResearch] Query generation failed: {exc} — using topic directly")
        # Fallback: simple variants of the topic
        return [
            topic,
            f"{topic} explained",
            f"{topic} latest 2025",
            f"{topic} Wikipedia",
        ][:n]


# ---------------------------------------------------------------------------
# Step 2 — DuckDuckGo search
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = 5) -> list[WebSource]:
    """Search DuckDuckGo and return result snippets (synchronous, no JS)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore

        results: list[WebSource] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    WebSource(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        source_type="web",
                    )
                )
        return results
    except Exception as exc:
        print(f"[WebResearch] DDG search failed for '{query}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 3 — Wikipedia API search
# ---------------------------------------------------------------------------

def _wikipedia_search(query: str, max_pages: int = 2) -> list[WebSource]:
    """Search Wikipedia and return page summaries via the REST v1 API."""
    sources: list[WebSource] = []
    try:
        # First: get search hit page titles
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_pages,
            "format": "json",
            "srinfo": "totalhits",
            "srprop": "snippet",
        }
        with httpx.Client(timeout=8.0, headers={"User-Agent": _USER_AGENT}) as client:
            resp = client.get(_WIKIPEDIA_SEARCH, params=params)
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            if not title:
                continue

            # Fetch a clean summary from the REST v1 endpoint
            try:
                with httpx.Client(timeout=8.0, headers={"User-Agent": _USER_AGENT}) as client:
                    summary_resp = client.get(
                        _WIKIPEDIA_REST.format(quote(title.replace(" ", "_")))
                    )
                    summary_resp.raise_for_status()
                    summary = summary_resp.json()

                sources.append(
                    WebSource(
                        url=summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        title=summary.get("title", title),
                        snippet=summary.get("description", ""),
                        content=summary.get("extract", "")[:_MAX_PAGE_CHARS],
                        source_type="wikipedia",
                    )
                )
            except Exception as exc:
                print(f"[WebResearch] Wikipedia page '{title}' fetch failed: {exc}")

    except Exception as exc:
        print(f"[WebResearch] Wikipedia search failed for '{query}': {exc}")

    return sources


# ---------------------------------------------------------------------------
# Step 4 — Async web scraping
# ---------------------------------------------------------------------------

def _extract_text(html: str) -> str:
    """Extract readable text from raw HTML via BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "header", "footer",
                      "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer article / main content containers
    for selector in ("article", "main", '[role="main"]', ".content", "#content", "body"):
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator=" ", strip=True)
            break
    else:
        text = soup.get_text(separator=" ", strip=True)

    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text[:_MAX_PAGE_CHARS]


async def _scrape_url(url: str, client: httpx.AsyncClient) -> str:
    """Fetch a URL and extract its text content."""
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = await client.get(url, timeout=_SCRAPE_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" not in ct and "text" not in ct:
            return ""
        return _extract_text(resp.text)
    except Exception as exc:
        print(f"[WebResearch] Scrape failed for {url}: {exc}")
        return ""


async def _enrich_sources(sources: list[WebSource]) -> list[WebSource]:
    """
    Concurrently scrape up to _CONCURRENCY sources that don't already have content.
    Wikipedia sources already carry an extract so we skip them.
    """
    sem = asyncio.Semaphore(_CONCURRENCY)
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

    async with httpx.AsyncClient(headers=headers, max_redirects=5) as client:

        async def _do(src: WebSource) -> WebSource:
            if src.content or src.source_type == "wikipedia":
                return src
            async with sem:
                src.content = await _scrape_url(src.url, client)
            return src

        enriched = await asyncio.gather(*[_do(s) for s in sources], return_exceptions=False)

    return list(enriched)


# ---------------------------------------------------------------------------
# Step 5 — LLM synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a research analyst preparing a briefing for an AI that will create an educational animation.

Below is raw content gathered from web searches about: **{topic}**

Your job:
1. Extract 5–10 KEY FACTS that are concrete, specific, and animation-friendly (numbers, comparisons, timelines, mechanisms, recent developments).
2. Write a 2–4 sentence SYNTHESIS PARAGRAPH summarising what is most important and visually interesting to animate.

Sources:
{sources_text}

Respond ONLY with valid JSON in this exact format:
{{
  "key_facts": ["fact 1", "fact 2", ...],
  "synthesis": "Two to four sentence paragraph here."
}}
"""


def _synthesise(topic: str, sources: list[WebSource], llm_chat: Callable) -> tuple[list[str], str]:
    """Have the LLM distil source content into key facts and a summary."""
    sources_text_parts = []
    for i, src in enumerate(sources, 1):
        body = src.content or src.snippet
        if not body:
            continue
        sources_text_parts.append(
            f"[Source {i}] {src.title} ({src.source_type})\n{body[:1500]}"
        )

    if not sources_text_parts:
        return [], ""

    prompt = _SYNTHESIS_PROMPT.format(
        topic=topic,
        sources_text="\n\n".join(sources_text_parts),
    )

    try:
        raw = llm_chat([{"role": "user", "content": prompt}], temperature=0.1)
        if not raw:
            raise ValueError("empty LLM response")

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("` ")
        data = json.loads(raw)
        facts = [str(f) for f in data.get("key_facts", [])]
        synthesis = str(data.get("synthesis", ""))
        return facts, synthesis
    except Exception as exc:
        print(f"[WebResearch] Synthesis failed: {exc}")
        # Fallback: collect snippets
        snippets = [s.snippet for s in sources if s.snippet][:5]
        return snippets, " ".join(snippets[:2])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def research_topic(
    topic: str,
    llm_chat: Callable,
    n_queries: int = 4,
    max_sources: int = 5,
) -> ResearchBrief:
    """
    Full research pipeline:
      1. Generate search queries via LLM
      2. Run DDG + Wikipedia searches (in thread executor to not block event loop)
      3. Deduplicate & pick top sources by relevance
      4. Scrape source pages concurrently
      5. Synthesise with LLM

    Args:
        topic:       The animation topic / user prompt.
        llm_chat:    The llm_chat() callable from nodes.py.
        n_queries:   How many search queries to generate.
        max_sources: Maximum unique sources to scrape and include.

    Returns:
        A ResearchBrief ready to be injected into the code-gen prompt.
    """
    print(f"[WebResearch] Starting research for: {topic!r}")
    t0 = time.monotonic()
    brief = ResearchBrief(topic=topic)

    # --- Step 1: generate queries ---
    queries = _gen_search_queries(topic, llm_chat, n=n_queries)
    brief.queries_used = queries
    print(f"[WebResearch] Generated queries: {queries}")

    # --- Step 2: search in parallel threads ---
    loop = asyncio.get_event_loop()

    async def _ddg_query(q: str) -> list[WebSource]:
        return await loop.run_in_executor(None, _ddg_search, q, 5)

    async def _wiki_query(q: str) -> list[WebSource]:
        return await loop.run_in_executor(None, _wikipedia_search, q, 2)

    search_coros = [_ddg_query(q) for q in queries] + [_wiki_query(queries[0])]
    search_results = await asyncio.gather(*search_coros, return_exceptions=True)

    all_sources: list[WebSource] = []
    for batch in search_results:
        if isinstance(batch, list):
            all_sources.extend(batch)

    # --- Step 3: Deduplicate by URL, prioritise Wikipedia ---
    seen_urls: set[str] = set()
    deduped: list[WebSource] = []
    # Wikipedia first for accuracy
    for src in sorted(all_sources, key=lambda s: 0 if s.source_type == "wikipedia" else 1):
        if src.url not in seen_urls and src.url:
            seen_urls.add(src.url)
            deduped.append(src)
        if len(deduped) >= max_sources:
            break

    print(f"[WebResearch] {len(deduped)} unique sources selected for scraping")

    # --- Step 4: Scrape ---
    try:
        enriched = await _enrich_sources(deduped)
    except Exception as exc:
        print(f"[WebResearch] Enrichment error: {exc}")
        enriched = deduped

    brief.sources = [s for s in enriched if s.content or s.snippet]

    # --- Step 5: Synthesise ---
    if brief.sources:
        brief.key_facts, brief.synthesis = _synthesise(topic, brief.sources, llm_chat)

    elapsed = time.monotonic() - t0
    print(f"[WebResearch] Done in {elapsed:.1f}s — {len(brief.sources)} sources, "
          f"{len(brief.key_facts)} facts")

    if not brief.sources:
        brief.error = "No content retrieved from any source"

    return brief
