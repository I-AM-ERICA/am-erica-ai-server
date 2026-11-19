from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    internal_key: str
    api_key: str = ""
    data_dir: Path = Path("outputs/promotion")
    public_base_url: str = "http://127.0.0.1:8081"
    timeout: float = 70
    demo: bool = False
    studio: bool = False
    font_path: str = ""
    bold_font_path: str = ""

    def __post_init__(self):
        if not self.internal_key:
            raise ValueError("AI_INTERNAL_KEY를 설정해 주세요.")
        if not self.demo and not self.api_key:
            raise ValueError("CLOVA_STUDIO_API_KEY를 설정하거나 --demo로 실행해 주세요.")
        url = urlparse(self.public_base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.query or url.fragment:
            raise ValueError("POSTER_PUBLIC_BASE_URL은 http(s) 기본 주소여야 합니다.")
        if not 5 <= self.timeout <= 85:
            raise ValueError("POSTER_TIMEOUT_SECONDS는 5~85 사이여야 합니다.")

    @classmethod
    def from_env(cls, *, demo=False, studio=False):
        return cls(
            internal_key=os.getenv("AI_INTERNAL_KEY", "local-internal-key" if demo else ""),
            api_key=os.getenv("CLOVA_STUDIO_API_KEY", ""),
            data_dir=Path(os.getenv("POSTER_DATA_DIR", "outputs/promotion")),
            public_base_url=os.getenv("POSTER_PUBLIC_BASE_URL", "http://127.0.0.1:8081").rstrip("/"),
            timeout=float(os.getenv("POSTER_TIMEOUT_SECONDS", "70")),
            font_path=os.getenv("POSTER_FONT_PATH", ""),
            bold_font_path=os.getenv("POSTER_BOLD_FONT_PATH", ""),
            demo=demo, studio=studio,
        )
