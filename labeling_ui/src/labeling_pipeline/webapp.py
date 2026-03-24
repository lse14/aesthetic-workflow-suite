import logging
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from .service import LabelingService


class NextRequest(BaseModel):
    weights: dict[str, float] | None = None
    avoid_sample_ids: list[int] | None = None
    after_sample_id: int | None = Field(default=None, ge=1)


class AnnotateRequest(BaseModel):
    sample_id: int
    aesthetic: int | None = Field(default=None, ge=1, le=5)
    composition: int | None = Field(default=None, ge=1, le=5)
    color: int | None = Field(default=None, ge=1, le=5)
    sexual: int | None = Field(default=None, ge=1, le=5)
    in_domain: int = Field(default=1, ge=0, le=1)
    content_type: str | None = "anime_illust"
    exclude_from_score_train: int = Field(default=0, ge=0, le=1)
    exclude_from_cls_train: int = Field(default=0, ge=0, le=1)
    exclude_reason: str | None = None
    note: str | None = None


class SkipRequest(BaseModel):
    sample_id: int
    in_domain: int = Field(default=1, ge=0, le=1)
    content_type: str | None = "anime_illust"
    exclude_from_score_train: int = Field(default=0, ge=0, le=1)
    exclude_from_cls_train: int = Field(default=0, ge=0, le=1)
    exclude_reason: str | None = None
    note: str | None = None


class AnnotateDimRequest(BaseModel):
    sample_id: int
    dim: str
    score: int | None = Field(default=None, ge=1, le=5)
    in_domain: int = Field(default=1, ge=0, le=1)
    content_type: str | None = "anime_illust"
    exclude_from_score_train: int = Field(default=0, ge=0, le=1)
    exclude_from_cls_train: int = Field(default=0, ge=0, le=1)
    exclude_reason: str | None = None
    note: str | None = None


class SettingsSaveRequest(BaseModel):
    config: dict


def _resample_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


@lru_cache(maxsize=4096)
def _load_thumbnail_bytes(path_str: str, mtime_ns: int, file_size: int, max_side: int) -> bytes:
    _ = (mtime_ns, file_size)
    with Image.open(path_str) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((max_side, max_side), _resample_filter())
        buf = BytesIO()
        rgb.save(buf, format="WEBP", quality=82, method=4)
        return buf.getvalue()


def create_app(config_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="标注 WebUI", version="0.1.0")
    log = logging.getLogger("labeling.webapp")
    service = LabelingService(config_path=config_path)
    app_root = Path(__file__).resolve().parents[2]
    index_file = app_root / "static" / "index.html"

    def require_localhost(req: Request) -> None:
        host = (req.client.host if req.client else "").strip().lower()
        if host in {"127.0.0.1", "::1", "localhost"}:
            return
        raise HTTPException(
            status_code=403,
            detail="settings endpoints are only available from localhost",
        )

    @app.on_event("startup")
    def startup_source_health_check():
        def _run_health_check() -> None:
            try:
                service.refresh_source_health(log_result=True)
                log.info("启动图源连通性检查已完成")
            except Exception as e:
                log.warning("启动图源连通性检查失败: %s", e)

        # Run health checks in background so startup is not blocked by network/local indexing.
        t = threading.Thread(target=_run_health_check, name="labeling-source-health", daemon=True)
        t.start()

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "labeling"}

    @app.get("/api/config")
    def config():
        return service.get_public_config()

    @app.get("/api/settings")
    def settings(request: Request):
        require_localhost(request)
        return service.get_full_config(redact_secrets=True)

    @app.post("/api/settings/save")
    def settings_save(req: SettingsSaveRequest, request: Request):
        require_localhost(request)
        try:
            out = service.save_and_apply_config(req.config)
            log.info("通过 UI 更新配置成功")
            return out
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/stats")
    def stats():
        return service.stats()

    @app.get("/api/source-health")
    def source_health(refresh: int = Query(default=0, ge=0, le=1)):
        try:
            return service.get_source_health(refresh=bool(refresh))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/reindex-local")
    def reindex_local():
        try:
            out = service.reindex_local()
            log.info("通过 UI 触发本地索引重建")
            return out
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/next")
    def next_sample(req: NextRequest):
        try:
            return service.next_sample(
                override_weights=req.weights,
                avoid_sample_ids=req.avoid_sample_ids,
                after_sample_id=req.after_sample_id,
            )
        except Exception as e:
            log.warning("获取下一样本失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/sample/{sample_id}")
    def sample_by_id(sample_id: int):
        try:
            return service.get_sample(sample_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.delete("/api/sample/{sample_id}")
    def sample_delete(sample_id: int, delete_image: bool = Query(default=True)):
        try:
            out = service.delete_sample(sample_id=sample_id, delete_image=delete_image)
            log.info("通过 UI 删除样本成功。id=%s", sample_id)
            return out
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/last-reviewed")
    def last_reviewed(status: str | None = Query(default=None)):
        try:
            row = service.get_last_reviewed_sample(status=status)
            if row is None:
                return {"sample": None}
            return {"sample": row}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/sources")
    def sources():
        try:
            return service.list_sources()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/samples")
    def samples(
        page: int = Query(default=1, ge=1),
        size: int = Query(default=30, ge=1, le=200),
        status: str = Query(default="all"),
        source: str | None = Query(default=None),
        order: str = Query(default="desc"),
        in_domain: int | None = Query(default=None, ge=0, le=1),
        content_type: str | None = Query(default=None),
        score_dim: str | None = Query(default=None),
        score_value: int | None = Query(default=None, ge=1, le=5),
        after_id: int | None = Query(default=None, ge=1),
    ):
        try:
            return service.list_samples(
                page=page,
                size=size,
                status=status,
                source=source,
                order=order,
                in_domain=in_domain,
                content_type=content_type,
                score_dim=score_dim,
                score_value=score_value,
                after_id=after_id,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/annotate")
    def annotate(req: AnnotateRequest):
        try:
            service.annotate(
                sample_id=req.sample_id,
                aesthetic=req.aesthetic,
                composition=req.composition,
                color=req.color,
                sexual=req.sexual,
                in_domain=req.in_domain,
                content_type=req.content_type,
                exclude_from_score_train=req.exclude_from_score_train,
                exclude_from_cls_train=req.exclude_from_cls_train,
                exclude_reason=req.exclude_reason,
                note=req.note,
            )
            return {"ok": True}
        except Exception as e:
            log.warning("提交标注失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/annotate-dim")
    def annotate_dim(req: AnnotateDimRequest):
        try:
            service.annotate_dim(
                sample_id=req.sample_id,
                dim=req.dim,
                score=req.score,
                in_domain=req.in_domain,
                content_type=req.content_type,
                exclude_from_score_train=req.exclude_from_score_train,
                exclude_from_cls_train=req.exclude_from_cls_train,
                exclude_reason=req.exclude_reason,
                note=req.note,
            )
            return {"ok": True}
        except Exception as e:
            log.warning("单维标注失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/skip")
    def skip(req: SkipRequest):
        try:
            service.skip(
                sample_id=req.sample_id,
                in_domain=req.in_domain,
                content_type=req.content_type,
                exclude_from_score_train=req.exclude_from_score_train,
                exclude_from_cls_train=req.exclude_from_cls_train,
                exclude_reason=req.exclude_reason,
                note=req.note,
            )
            return {"ok": True}
        except Exception as e:
            log.warning("跳过样本失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/image/{filename}")
    def image(
        filename: str,
        thumb: int = Query(default=0, ge=0, le=1),
        thumb_size: int = Query(default=480, ge=96, le=1280),
    ):
        try:
            path = service.image_path(filename)
            if thumb:
                stat = path.stat()
                data = _load_thumbnail_bytes(
                    str(path),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                    int(thumb_size),
                )
                return Response(
                    content=data,
                    media_type="image/webp",
                    headers={"Cache-Control": "public, max-age=31536000, immutable"},
                )
            return FileResponse(
                str(path),
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/")
    def root():
        if not index_file.exists():
            raise HTTPException(status_code=500, detail=f"Missing UI file: {index_file}")
        return FileResponse(str(index_file))

    return app
