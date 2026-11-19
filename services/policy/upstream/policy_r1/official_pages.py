"""Bounded HTML excerpts from explicitly supported public policy websites."""
from html.parser import HTMLParser
import ipaddress
import re
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from urllib.parse import urlsplit


OFFICIAL_HOSTS = frozenset({"www.semas.or.kr", "semas.or.kr", "ols.semas.or.kr",
                            "www.mss.go.kr", "www.bizinfo.go.kr"})


class PageUnavailable(RuntimeError):
    pass


def public_page_url(url):
    """Syntax gate; DNS addresses are separately checked and pinned before fetching."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if (parts.scheme != "https" or parts.port not in (None, 443) or parts.username
                or parts.password or not host or "." not in host or len(url) > 4096
                or any(ord(c) < 32 for c in url)):
            return False
        try:
            ipaddress.ip_address(host)
            return False  # only named websites, never literal addresses
        except ValueError:
            return not host.endswith((".localhost", ".local", ".internal"))
    except (ValueError, TypeError):
        return False


def official_url(url):
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        return (parts.scheme == "https" and (host in OFFICIAL_HOSTS or host.endswith(".go.kr"))
                and parts.port in (None, 443) and not parts.username and not parts.password
                and not any(ord(c) < 32 for c in url) and len(url) <= 4096)
    except (ValueError, TypeError):
        return False


class ContentParser(HTMLParser):
    """Prefer a content container; never substitute whole navigation-only pages."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.candidates = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = (tag in self.SKIP or "hidden" in attrs or attrs.get("aria-hidden") == "true"
                  or any(node[1] for node in self.stack))
        parent = self.stack[-1][0] if self.stack else None
        section_start = tag in {"h1", "h2", "h3", "h4", "h5", "h6", "dt"} or (
            tag == "strong" and parent in {"dd", "div", "section"})
        if section_start and not hidden:
            for _, _, active in self.stack:
                if active is not None and active["parts"] and active["parts"][-1] != "":
                    active["parts"].append("")
        classes = set((attrs.get("class") or "").split())
        rank = (3 if tag in {"main", "article"} else
                2 if classes & {"contents", "board_view", "view_cont", "view_content"} else
                1 if attrs.get("id") in {"content", "contents", "sub_contents"} else 0)
        candidate = None
        if rank and not hidden:
            candidate = {"rank": rank, "parts": [], "length": 0}
            self.candidates.append(candidate)
        if tag not in self.VOID:
            self.stack.append((tag, hidden, candidate))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not self.stack or self.stack[-1][1]:
            return
        text = " ".join(data.split())
        if text:
            for _, _, candidate in self.stack:
                if candidate is not None and candidate["length"] < 20000:
                    candidate["parts"].append(text)
                    candidate["length"] += len(text)


def extract_html(raw):
    # Supported sites use UTF-8 or Korean legacy encodings; no replacement-decoding.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            html = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise PageUnavailable("unsupported_encoding")
    parser = ContentParser()
    parser.feed(html)
    candidates = [(c["rank"], re.sub(r"\n{3,}", "\n\n", "\n".join(c["parts"])).strip()) for c in parser.candidates]
    candidates = [(rank, text) for rank, text in candidates if len(text) >= 120]
    if not candidates:
        raise PageUnavailable("no_html_body")
    return max(candidates, key=lambda c: (c[0], len(c[1])))[1][:5000]


def fetch_official_page(url, timeout=8):
    if not official_url(url):
        raise PageUnavailable("unsupported_host")
    return fetch_public_page(url, timeout)


def fetch_public_page(url, timeout=8):
    if not public_page_url(url):
        raise PageUnavailable("unsupported_url")
    if timeout <= 0:
        raise PageUnavailable("deadline")
    curl = shutil.which("curl")
    if not curl:
        raise PageUnavailable("curl_missing")
    host = urlsplit(url).hostname
    try:
        addresses = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
        ips = [row[4][0] for row in addresses]
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise PageUnavailable("nonpublic_address")
        # Pin the validated IP; keep hostname for TLS validation. No proxies, cookies,
        # curl config, redirects, API credentials or insecure TLS options.
        with tempfile.TemporaryDirectory(prefix="policy-page-") as tmp:
            output = Path(tmp) / "page.html"
            result = subprocess.run([
                curl, "--disable", "--silent", "--show-error", "--noproxy", "*",
                "--proto", "=https", "--resolve", f"{host}:443:{ips[0]}",
                "--max-time", str(min(timeout, 8)), "--max-filesize", "1048576",
                "--output", str(output), "--write-out", "%{http_code}\n%{content_type}", url,
            ], capture_output=True, timeout=min(timeout, 8) + 1,
               env={"PATH": "/usr/bin:/bin", "LANG": "C"})
            if result.returncode:
                raise PageUnavailable("page_request_failed")
            status, _, content_type = result.stdout.decode("ascii", "replace").partition("\n")
            if status != "200" or content_type.split(";")[0].strip().lower() not in {"text/html", "application/xhtml+xml"}:
                raise PageUnavailable("not_html_200")
            if output.stat().st_size > 1048576:
                raise PageUnavailable("page_too_large")
            return extract_html(output.read_bytes())
    except (OSError, subprocess.SubprocessError):
        raise PageUnavailable("page_request_failed") from None
