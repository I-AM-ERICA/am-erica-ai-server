import hmac
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from .clova import ProviderError
from .models import Brief, Revision
from .storage import NotFound, Conflict
from .templates import TEMPLATES, default_font
from .font_catalog import public_fonts, FONT_BY_ID

STATIC = Path(__file__).parent / "static"
MAX_BODY = 22 * 1024 * 1024


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, count = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            count += len(body)
            if count > MAX_BODY:
                return await JSONResponse({"detail": "요청은 22MB 이하여야 합니다."}, 413)(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body"):
                break
        consumed = False
        async def replay():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        await self.app(scope, replay, send)


def create_app(engine):
    app = FastAPI(title="Promotion-AI", version="0.1.0",
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit)
    studio_token = secrets.token_urlsafe(32)

    def local(request):
        return request.client and request.client.host in ("127.0.0.1", "::1", "testclient") and request.url.hostname in ("127.0.0.1", "localhost", "::1", "testserver")

    def authorize(request: Request):
        given = request.headers.get("x-internal-key", "")
        if hmac.compare_digest(given.encode(), engine.settings.internal_key.encode()):
            return
        if engine.settings.studio and local(request):
            origin = request.headers.get("origin", "")
            parsed = urlsplit(origin) if origin else None
            same_origin = not parsed or (parsed.scheme == request.url.scheme and parsed.netloc == request.url.netloc)
            if same_origin and hmac.compare_digest(request.headers.get("x-studio-token", "").encode(), studio_token.encode()):
                return
        raise HTTPException(401, "서버 인증이 필요합니다.")

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        return JSONResponse({"detail": "입력 형식을 확인해 주세요.", "fields": [
            {"loc": list(e["loc"]), "message": e["msg"]} for e in error.errors()]}, 422)

    @app.exception_handler(NotFound)
    async def missing(request, error):
        return JSONResponse({"detail": str(error)}, 404)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, 409)

    @app.exception_handler(ProviderError)
    async def provider(request, error):
        status = {"DEADLINE": 504, "BUSY": 429, "RATE_LIMIT": 429,
                  "FACT_EDIT": 422, "DEMO": 422, "AUTH": 503}.get(error.code, 502)
        return JSONResponse({"detail": str(error), "code": error.code}, status)

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({"detail": str(error)}, 422)

    @app.get("/health")
    def health():
        return {"status": "ready", "mode": "demo" if engine.settings.demo else "clova",
                "provider_checked": False}

    @app.get("/v1/templates", dependencies=[Depends(authorize)])
    def templates():
        return {"templates": [t.__dict__ for t in TEMPLATES]}

    @app.get("/v1/fonts", dependencies=[Depends(authorize)])
    def fonts():
        return {"fonts": public_fonts()}

    @app.post("/v1/posters/drafts", dependencies=[Depends(authorize)])
    def create(brief: Brief):
        return engine.create(brief)

    @app.get("/v1/posters/{draft_id}", dependencies=[Depends(authorize)])
    def get(draft_id: str, store_id: str):
        return engine.get(draft_id, store_id)

    @app.post("/v1/posters/{draft_id}/revisions", dependencies=[Depends(authorize)])
    def revise(draft_id: str, revision: Revision):
        return engine.revise(draft_id, revision)

    @app.post("/v1/posters", dependencies=[Depends(authorize)])
    def legacy(brief: Brief):
        result = engine.create_legacy(brief, True)
        result["imageUrl"] = result.pop("image_url")
        return result

    @app.get("/images/{name}")
    def image(name: str, download: bool = False):
        # Unguessable capability URL, intended for sharing to the existing frontend.
        response = FileResponse(engine.repository.image(name), media_type="image/png",
                                filename="poster.png" if download else None)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.get("/")
    def studio(request: Request):
        if not engine.settings.studio or not local(request):
            raise HTTPException(404)
        config = json.dumps({"token": studio_token, "mode": "demo" if engine.settings.demo else "clova", "fonts": public_fonts(), "templateFonts": {t.id: default_font(t.id) for t in TEMPLATES}})
        html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__CONFIG__", config)
        return HTMLResponse(html, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                          "Referrer-Policy": "no-referrer"})

    @app.get("/studio/{asset}")
    def assets(asset: str, request: Request):
        if not engine.settings.studio or not local(request) or asset not in ("studio.css", "studio.js", "sample-product.png"):
            raise HTTPException(404)
        return FileResponse(STATIC / asset)

    @app.get("/studio/font-preview/{font_id}.png")
    def font_preview(font_id: str, request: Request):
        if not engine.settings.studio or not local(request) or font_id not in FONT_BY_ID:
            raise HTTPException(404)
        return Response(engine.renderer.font_preview(font_id), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    return app
