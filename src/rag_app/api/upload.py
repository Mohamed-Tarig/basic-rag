import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from rag_app.ingestion.pipeline import run_ingestion

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

COLLECTION_PREFIX = os.getenv(
    "RAG_UPLOAD_COLLECTION_PREFIX",
    "basic_rag",
)


@router.post("/documents/upload")
async def upload_document(
    file: Annotated[UploadFile, File()],
) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")


    try:
        with TemporaryDirectory() as temp_dir:
            original_name = Path(file.filename.replace("\\", "/")).name
            pdf_path = Path(temp_dir) / original_name

            size = 0
            with pdf_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)

                    if size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF exceeds the 20 MB limit",
                        )

                    output.write(chunk)

            if size == 0:
                raise HTTPException(status_code=400, detail="Empty PDF file")

            with pdf_path.open("rb") as saved_file:
                header = saved_file.read(5)

            if header != b"%PDF-":
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file",
                )

            result = await run_in_threadpool(
                run_ingestion,
                pdf_path=pdf_path,
                collection_prefix=COLLECTION_PREFIX,
            )

            return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion failed",
        ) from exc
    finally:
        await file.close()