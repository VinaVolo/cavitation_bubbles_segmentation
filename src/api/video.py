import logging
import os
import shutil
import tempfile
import uuid
import zipfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from src.api.auth import get_current_user
from src.ml.processing import VideoProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_SIZE_MB = 500
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

MODEL_PATH = "hf_model_repo/model.pt"
video_processor = VideoProcessor(MODEL_PATH)


@router.post("/process_video/")
async def process_video_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    tmp_dir = tempfile.mkdtemp()
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(tmp_dir, unique_filename)

        size = 0
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB",
                    )
                buffer.write(chunk)

        output_video_path = os.path.join(tmp_dir, f"processed_{unique_filename}.mp4")
        csv_path = os.path.join(tmp_dir, f"data_{unique_filename.split('.')[0]}.csv")

        speed_hist_file, area_hist_file, hist_data_file = video_processor.process_video(
            input_path, output_video_path, csv_path, tmp_dir
        )

        zip_path = os.path.join(tmp_dir, "results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_video_path, "output_video.mp4")
            zf.write(csv_path, "data.csv")
            if speed_hist_file:
                zf.write(speed_hist_file, "histogram_speed.png")
            if area_hist_file:
                zf.write(area_hist_file, "histogram_area.png")
            if hist_data_file:
                zf.write(hist_data_file, "histogram_data.json")

        def _stream_and_cleanup():
            try:
                with open(zip_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return StreamingResponse(
            _stream_and_cleanup(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
