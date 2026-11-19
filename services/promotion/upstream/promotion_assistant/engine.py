from contextlib import contextmanager
from datetime import datetime, timezone
import threading
import time
import uuid
from .clova import ProviderError
from .models import Brief, Design, validate_copy
from .renderer import normalize_image, vision_image
from .storage import Conflict
from .product_image import ProductProcessor
from .templates import diverse_designs, reference_family
from .visual_profile import visual_profile, is_product_set


class Engine:
    def __init__(self, settings, designer, renderer, repository, processor=None):
        self.settings, self.designer = settings, designer
        self.renderer, self.repository = renderer, repository
        self.slots = threading.BoundedSemaphore(2)
        self.processor = processor or ProductProcessor()

    @contextmanager
    def slot(self):
        if not self.slots.acquire(blocking=False):
            raise ProviderError("BUSY", "다른 포스터를 제작 중입니다. 잠시 후 다시 시도해 주세요.")
        try:
            yield
        finally:
            self.slots.release()

    def check(self, deadline, cancelled):
        if cancelled() or time.monotonic() >= deadline:
            raise ProviderError("DEADLINE", "생성이 취소됐거나 제한 시간을 초과했습니다.")

    def expose(self, record):
        result = {k: v for k, v in record.items() if k not in ("image_name", "project_id")}
        result["image_url"] = self.settings.public_base_url + "/images/" + record["image_name"]
        return result

    def caption(self, brief, design):
        tags = ["#" + "".join(c for c in value if c.isalnum()) for value in [brief.store.name, brief.product]]
        tags = [t[:40] for t in tags if len(t) > 1]
        body = "\n".join(v for v in [design.subhead, brief.message,
                        f"{brief.price:,}원" if brief.price is not None else "",
                        brief.period, brief.contact, design.cta] if v)
        return {"headline": design.headline, "body": body, "hashtags": list(dict.fromkeys(tags))}

    def record(self, brief, design, *, draft_id=None, version=1):
        return {
            "id": draft_id or uuid.uuid4().hex, "version": version,
            "design": design.model_dump(), "caption": self.caption(brief, design),
            "mode": "demo" if self.settings.demo else "clova",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "width": 1080, "height": 1350,
        }

    def create(self, brief: Brief, *, deadline=None, cancelled=lambda: False):
        deadline = min(deadline or float("inf"), time.monotonic() + self.settings.timeout)
        with self.slot():
            self.check(deadline, cancelled)
            photo, logo = normalize_image(brief.product_image), normalize_image(brief.logo)
            reference = normalize_image(brief.style_reference)
            mode = "keep" if brief.background_mode == "auto" and is_product_set(brief.product) else brief.background_mode
            processed = self.processor.process(photo, mode)
            if mode == "keep" and brief.background_mode == "auto" and photo:
                processed.warning = "세트 상품은 구성을 보존하도록 원본 사진을 사용했습니다. 배경을 분리하려면 '배경 제거하기'를 선택해 주세요."
            self.check(deadline, cancelled)
            style = self.designer.analyze_style(vision_image(reference), deadline) if reference else ""
            profile = visual_profile(processed.data, brief)
            profile["style_family"] = reference_family(style)
            designs = diverse_designs(self.designer.designs(brief, deadline, style=style, visual_context=profile), profile)
            prepared = []
            for design in designs:
                self.check(deadline, cancelled)
                design = design.model_copy(update={"hide_footer": brief.hide_footer})
                design = validate_copy(design, brief)
                png = self.renderer.render(brief, design, processed.data, logo)
                record = self.record(brief, design)
                record["image_processing"] = {"background": processed.status, "warning": processed.warning}
                prepared.append((record, png))
            self.check(deadline, cancelled)
            records = self.repository.create(brief, photo, logo, style, prepared,
                                             cutout=processed.data if processed.status == "removed" else None)
            return {"drafts": [self.expose(r) for r in records], "mode": "demo" if self.settings.demo else "clova"}

    def create_legacy(self, brief, generate_image, **kwargs):
        if generate_image:
            first = self.create(brief, **kwargs)["drafts"][0]
            return first["caption"] | {"image_url": first["image_url"]}
        deadline = min(kwargs.get("deadline") or float("inf"), time.monotonic()+self.settings.timeout)
        with self.slot():
            design = self.designer.designs(brief, deadline)[0]
            self.check(deadline, kwargs.get("cancelled", lambda: False))
            return self.caption(brief, validate_copy(design, brief)) | {"image_url": ""}

    def get(self, draft_id, store_id):
        record, *_ = self.repository.get(draft_id, store_id)
        return self.expose(record)

    def revise(self, draft_id, revision, *, deadline=None, cancelled=lambda: False):
        deadline = min(deadline or float("inf"), time.monotonic()+self.settings.timeout)
        with self.slot():
            old, facts, photo, logo, style = self.repository.get(draft_id, revision.store_id)
            if old["version"] != revision.expected_version:
                raise Conflict("시안이 이미 수정됐습니다. 최신 시안을 다시 열어 주세요.")
            brief = Brief.model_validate(facts)
            design = self.designer.designs(brief, deadline, style=style,
                                          current=Design.model_validate(old["design"]),
                                          instruction=revision.instruction)[0]
            design = validate_copy(design, brief)
            self.check(deadline, cancelled)
            png = self.renderer.render(brief, design, photo, logo)
            self.check(deadline, cancelled)
            record = self.record(brief, design, draft_id=draft_id, version=old["version"]+1)
            record["project_id"] = old["project_id"]
            record["image_processing"] = old.get("image_processing", {"background": "kept", "warning": ""})
            return self.expose(self.repository.revise(record, png, revision.expected_version))
