"""
Data export API endpoints (CSV, JSON, Excel, PDF).
"""

import json
import csv
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import io

from ai_business_assistant.config import get_settings
from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/csv/{data_type}")
async def export_csv(
    data_type: str,
    filters: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export data as CSV."""
    try:
        data = []
        
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/json/{data_type}")
async def export_json(
    data_type: str,
    filters: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export data as JSON."""
    try:
        data = []
        
        json_data = json.dumps(data, default=str, indent=2)
        
        return StreamingResponse(
            iter([json_data]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={data_type}_{datetime.now().strftime('%Y%m%d')}.json"
            }
        )
        
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/excel/{data_type}")
async def export_excel(
    data_type: str,
    filters: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export data as Excel."""
    try:
        settings.ensure_dirs()
        export_path = settings.abs_export_dir / f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        logger.info(f"Excel export requested for {data_type}")
        
        return {
            "message": "Excel export feature requires openpyxl library",
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdf/{report_type}")
async def export_pdf(
    report_type: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export report as PDF."""
    try:
        logger.info(f"PDF export requested for {report_type}")
        
        return {
            "message": "PDF export feature requires reportlab library",
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
