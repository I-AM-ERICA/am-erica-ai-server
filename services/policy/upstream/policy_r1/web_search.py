"""SerpAPI search with labeled snippets and optional supported official HTML excerpts."""
from collections import OrderedDict
import hashlib
from http.client import HTTPException
import json
import math
import os
import re
import socket
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from .official_pages import official_url, public_page_url, fetch_official_page, fetch_public_page, PageUnavailable


class SearchUnavailable(RuntimeError):
    CODES = {"SEARCH_ERROR", "TIMEOUT", "TLS_CERTIFICATE_ERROR", "TLS_ERROR", "DNS_ERROR",
             "CONNECTION_ERROR", "HTTP_PROTOCOL_ERROR", "NETWORK_ERROR", "INVALID_JSON",
             "INVALID_RESPONSE", "INVALID_RESULTS", "PROVIDER_ERROR", "RESPONSE_TOO_LARGE", "DEADLINE"}

    def __init__(self, message, *, code="SEARCH_ERROR", search_id=None):
        super().__init__(message)
        self.code = code if isinstance(code, str) and (code in self.CODES or re.fullmatch(r"HTTP_[1-5][0-9]{2}", code)) else "SEARCH_ERROR"
        self.search_id = search_id if isinstance(search_id, str) and re.fullmatch(r"[a-f0-9]{24}", search_id) else None


def network_failure_code(error):
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, TimeoutError):
        return "TIMEOUT"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "TLS_CERTIFICATE_ERROR"
    if isinstance(reason, ssl.SSLError):
        return "TLS_ERROR"
    if isinstance(reason, socket.gaierror):
        return "DNS_ERROR"
    if isinstance(reason, ConnectionError):
        return "CONNECTION_ERROR"
    if isinstance(reason, HTTPException):
        return "HTTP_PROTOCOL_ERROR"
    return "NETWORK_ERROR"


class NoRedirect(HTTPRedirectHandler):
    # Never forward an API-key-bearing request to a different endpoint.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_link(value):
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            return None
        if any(ord(char) < 32 for char in value):
            return None
        return value
    except ValueError:
        return None


def unusable_evidence(url, snippet):
    """Reject navigation wrappers and Google's missing-description notices."""
    host = urlsplit(url).hostname.lower().rstrip(".")
    google = any(host == domain or host.endswith("." + domain)
                 for domain in ("google.com", "google.co.kr"))
    if google and urlsplit(url).path.rstrip("/") in {"/goto", "/url"}:
        # Do not guess a destination from an abbreviated title or fetch redirects.
        return "google_redirect"
    normalized = " ".join(snippet.casefold().replace("’", "'").split())
    notices = ("이 페이지에 관한 정보가 없습니다", "이 페이지에 대한 정보가 없습니다",
               "no information is available for this page", "no information available for this page",
               "a description for this result is not available")
    if any(notice in normalized for notice in notices):
        return "missing_description"
    return None


class SerpAPIRetriever:
    def __init__(self, api_key=None, timeout=15.0, cache_size=128, cache_ttl=300.0, opener=None,
                 engine="google", page_fetcher=None, require_page_evidence=False):
        if engine not in {"google", "naver"}:
            raise ValueError("Search engine must be google or naver")
        self.engine = engine
        self.require_page_evidence = require_page_evidence
        self._fetch_page = page_fetcher or (fetch_public_page if require_page_evidence else fetch_official_page)
        self._key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        if not self._key.strip():
            raise ValueError("Set SERPAPI_API_KEY on the GPU server")
        if not math.isfinite(timeout) or timeout <= 0 or cache_size < 0 or cache_ttl < 0:
            raise ValueError("Invalid search timeout/cache configuration")
        self.timeout = timeout
        self._cache_size, self._cache_ttl = cache_size, cache_ttl
        self._cache = OrderedDict()
        self._pages = OrderedDict()
        self._lock = threading.Lock()
        self._open = opener or build_opener(NoRedirect()).open
        self.last_diagnostics = {}

    def _read_page(self, url, timeout):
        if timeout <= 0:
            raise PageUnavailable("deadline")
        if not public_page_url(url) or unusable_evidence(url, ""):
            raise PageUnavailable("unsupported_url")
        with self._lock:
            cached = self._pages.get(url)
            if cached and time.monotonic() - cached[0] < self._cache_ttl:
                self._pages.move_to_end(url)
                return cached[1]
        body = self._fetch_page(url, timeout=timeout)
        if not isinstance(body, str) or len(body.strip()) < 120:
            raise PageUnavailable("no_html_body")
        body = body.strip()[:5000]
        if self._cache_size:
            with self._lock:
                self._pages[url] = (time.monotonic(), body)
                self._pages.move_to_end(url)
                while len(self._pages) > self._cache_size:
                    self._pages.popitem(last=False)
        return body

    def read_pages(self, urls, timeout=24):
        """Re-read earlier source URLs; caller-supplied history is not evidence."""
        deadline = time.monotonic() + timeout
        hits = []
        for url in list(dict.fromkeys(urls))[:3]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                body = self._read_page(url, min(8, remaining))
            except PageUnavailable:
                continue
            title = urlsplit(url).hostname
            kind = "official_page_excerpt" if official_url(url) else "page_excerpt"
            hits.append({"id": "web_" + hashlib.sha256(url.encode()).hexdigest()[:16],
                         "title": title, "url": url, "snippet": body, "score": 1.0,
                         "evidence_kind": kind, "contents": f"제목: {title}\nHTML 본문 발췌(일부; 첨부파일 미확인): {body}"})
        return hits

    def search(self, query, top_k=3, timeout=None):
        query = query.strip()
        if not query or len(query) > 1000 or not 1 <= top_k <= 5:
            raise ValueError("Search query must be 1..1000 characters and top_k 1..5")
        key = (self.engine, query, top_k)
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < self._cache_ttl:
                self._cache.move_to_end(key)
                self.last_diagnostics = {"engine": self.engine, "cache_hit": True, "returned": len(cached[1])}
                return [dict(hit) for hit in cached[1]]
            self._cache.pop(key, None)
        query_field = "query" if self.engine == "naver" else "q"
        result_field = "web_results" if self.engine == "naver" else "organic_results"
        params = {"engine": self.engine, query_field: query, "api_key": self._key}
        params.update({"where": "nexearch"} if self.engine == "naver" else {"hl": "ko", "gl": "kr"})
        request = Request("https://serpapi.com/search.json?" + urlencode(params),
                          headers={"Accept": "application/json", "User-Agent": "Search-R1-Policy/1.0"})
        budget = min(self.timeout, timeout) if timeout is not None else self.timeout
        if budget <= 0:
            raise SearchUnavailable("Search deadline exceeded", code="DEADLINE")
        started = time.monotonic()
        try:
            with self._open(request, timeout=budget) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise SearchUnavailable("Search response exceeded size limit", code="RESPONSE_TOO_LARGE")
            payload = json.loads(raw)
        except HTTPError as error:
            # Do not print URLs, response bodies or provider exceptions: they may contain the key.
            raise SearchUnavailable(f"SerpAPI HTTP_{error.code} after {time.monotonic() - started:.1f}s",
                                    code=f"HTTP_{error.code}") from None
        except (URLError, OSError, HTTPException) as error:
            code = network_failure_code(error)
            raise SearchUnavailable(
                f"SerpAPI {code} after {time.monotonic() - started:.1f}s (socket timeout={budget:g}s)", code=code) from None
        except (ValueError, UnicodeError):
            raise SearchUnavailable(f"SerpAPI INVALID_JSON after {time.monotonic() - started:.1f}s", code="INVALID_JSON") from None
        if not isinstance(payload, dict):
            raise SearchUnavailable("SerpAPI returned an invalid response", code="INVALID_RESPONSE")
        metadata = payload.get("search_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        search_id = SearchUnavailable("", search_id=metadata.get("id")).search_id
        # SerpAPI can return status=Success plus an error field for an empty search.
        # Only recognize explicit empty-result messages; never hide account/API failures.
        empty_message = payload.get("error") in {
            "Google hasn't returned any results for this query.",
            "Naver hasn't returned any results for this query.",
        } if isinstance(payload.get("error"), str) else False
        empty_success = metadata.get("status") == "Success" and empty_message and not payload.get(result_field)
        if payload.get("error") and not empty_success:
            raise SearchUnavailable("SerpAPI returned an error; inspect the search in your SerpAPI dashboard",
                                    code="PROVIDER_ERROR", search_id=search_id)
        organic = [] if empty_success else payload.get(result_field, [])
        if not isinstance(organic, list):
            raise SearchUnavailable("SerpAPI returned an invalid results structure", code="INVALID_RESULTS", search_id=search_id)
        hits, seen, rejected, page_failures = [], set(), {}, {}
        page_attempts = 0
        def reject(reason):
            rejected[reason] = rejected.get(reason, 0) + 1
        # Read likely official sources first; never mistake a snippet for a page.
        ranked = list(enumerate(organic, 1))
        if self.require_page_evidence:
            ranked.sort(key=lambda pair: not (isinstance(pair[1], dict) and official_url(pair[1].get("link", ""))))
        for rank, row in ranked:
            if not isinstance(row, dict):
                reject("invalid_result")
                continue
            url = public_link(row.get("link"))
            title, snippet = row.get("title"), row.get("snippet", "")
            if not url or url in seen or not isinstance(title, str) or not title.strip():
                reject("invalid_or_duplicate_link_or_title")
                continue
            evidence_kind = "search_snippet"
            if self.require_page_evidence:
                if not public_page_url(url) or unusable_evidence(url, ""):
                    reject("unreadable_url")
                    continue
                if page_attempts >= 3:
                    reject("page_budget")
                    continue
                page_attempts += 1
                remaining = (timeout - (time.monotonic() - started)) if timeout is not None else 24 - (page_attempts - 1) * 8
                try:
                    snippet = self._read_page(url, min(8, remaining))
                    evidence_kind = "official_page_excerpt" if official_url(url) else "page_excerpt"
                except PageUnavailable as error:
                    page_failures[str(error)] = page_failures.get(str(error), 0) + 1
                    reject("page_unavailable")
                    continue
            elif (not isinstance(snippet, str) or not snippet.strip()) and official_url(url) and page_attempts < 2:
                page_attempts += 1
                remaining = (timeout - (time.monotonic() - started)) if timeout is not None else 16 - (page_attempts - 1) * 8
                try:
                    snippet = self._fetch_page(url, timeout=min(8, remaining))
                    evidence_kind = "official_page_excerpt"
                except PageUnavailable as error:
                    page_failures[str(error)] = page_failures.get(str(error), 0) + 1
            if not isinstance(snippet, str) or not snippet.strip():
                reject("empty_snippet")
                continue
            reason = unusable_evidence(url, snippet)
            if reason:
                reject(reason)
                continue
            seen.add(url)
            is_page = evidence_kind in {"official_page_excerpt", "page_excerpt"}
            title, snippet = title.strip()[:300], snippet.strip()[:5000 if is_page else 2000]
            label = (("공식 HTML" if evidence_kind == "official_page_excerpt" else "HTML")
                     + " 본문 발췌(일부; 첨부파일 미확인)" if is_page else "검색 요약(원문 미확인)")
            hits.append({"id": "web_" + hashlib.sha256(url.encode()).hexdigest()[:16],
                         "title": title, "url": url, "snippet": snippet, "score": 1.0 / rank,
                         "evidence_kind": evidence_kind,
                         "contents": f"제목: {title}\nURL: {url}\n{label}: {snippet}"})
        params_echo = payload.get("search_parameters")
        self.last_diagnostics = {
            "engine": self.engine, "result_field": result_field, "cache_hit": False,
            "search_id": search_id, "provider_empty_result": empty_success,
            "result_count": len(organic), "usable_results": len(hits),
            "rejected": rejected,
            "page_fetch_attempts": page_attempts, "page_fetch_failures": page_failures,
            "query_matches_response": params_echo.get(query_field) == query if isinstance(params_echo, dict) else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        hits = hits[:top_k]
        if self._cache_size:
            with self._lock:
                self._cache[key] = (time.monotonic(), tuple(dict(hit) for hit in hits))
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return hits
