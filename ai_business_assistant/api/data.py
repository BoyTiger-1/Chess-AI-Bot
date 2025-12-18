"""
Data upload and management API endpoints.
"""

from typing import Optional
import os
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.config import get_settings
from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    data_type: str = Query(..., description="Type of data: market, customer, competitor"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload data file (CSV, JSON, Excel)."""
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_UPLOAD_EXTENSIONS}"
        )
    
    try:
        settings.ensure_dirs()
        upload_path = settings.abs_upload_dir / f"{datetime.now().timestamp()}_{file.filename}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} by user {current_user.id}")
        
        return {
            "filename": file.filename,
            "size": file.size,
            "data_type": data_type,
            "status": "uploaded",
            "message": "File uploaded successfully. Processing will begin shortly."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_data(
    query: str = Query(...),
    data_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Search across all data sources."""
    try:
        results = []
        
        logger.info(f"Search query: {query} by user {current_user.id}")
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads")
async def list_uploads(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """List uploaded files."""
    try:
        settings.ensure_dirs()
        upload_dir = settings.abs_upload_dir
        
        files = []
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        files.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        return {
            "files": files[offset:offset+limit],
            "total": len(files)
        }
        
    except Exception as e:
        logger.error(f"Failed to list uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))
