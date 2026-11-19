import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
from .api import create_app
from .clova import ClovaDesigner, DemoDesigner
from .config import Settings
from .engine import Engine
from .grpc_service import build_server
from .renderer import Renderer
from .storage import Repository


def main():
    parser = argparse.ArgumentParser(description="학습 없이 실행하는 CLOVA 홍보 비서")
    parser.add_argument("--demo", action="store_true", help="API를 호출하지 않는 체험 모드")
    parser.add_argument("--studio", action="store_true", help="로컬 테스트용 편집 화면 활성화")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP listen host")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--grpc-address", default="127.0.0.1:50052")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-cutout", action="store_true", help="배경 제거 모델을 불러오지 않음")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    if args.studio and args.host not in ("127.0.0.1", "::1", "localhost"):
        parser.error("--studio는 로컬 주소에서만 실행할 수 있습니다.")
    # Runtime state and environments must NOT be synchronized by Google Drive.
    runtime = Path(os.environ.get("LOCALAPPDATA", str(Path.home()/".local/share"))) / "Promotion-AI"
    os.environ.setdefault("POSTER_DATA_DIR", str(runtime / "data"))
    os.environ.setdefault("POSTER_PUBLIC_BASE_URL", f"http://127.0.0.1:{args.port}")
    try:
        settings = Settings.from_env(demo=args.demo, studio=args.studio)
        renderer = Renderer(settings.font_path, settings.bold_font_path)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    designer = DemoDesigner() if settings.demo else ClovaDesigner(settings.api_key)
    engine = Engine(settings, designer, renderer, Repository(settings.data_dir))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.no_cutout:
        engine.processor.unavailable = True
    else:
        logging.info("상품 배경 제거 모델을 준비합니다. 최초 실행에는 U2Net 다운로드가 필요합니다.")
        try:
            engine.processor.warmup()
        except Exception as error:
            engine.processor.unavailable = True
            logging.warning("배경 제거 준비 실패 (%s). 원본 사진으로 계속 실행합니다.", type(error).__name__)
    server, _ = build_server(engine, args.grpc_address)
    server.start()
    logging.info("Promotion gRPC: %s; mode=%s; data=%s", args.grpc_address,
                 "DEMO (no AI calls)" if args.demo else "CLOVA", settings.data_dir)
    try:
        uvicorn.run(create_app(engine), host=args.host, port=args.port, access_log=False)
    finally:
        server.stop(5).wait()
        designer.close()


if __name__ == "__main__":
    main()
