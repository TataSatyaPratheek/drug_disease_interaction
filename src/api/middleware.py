# src/api/middleware.py - NEW FILE
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

async def error_handling_middleware(request: Request, call_next: Callable):
    """Global error handling middleware"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error processing {request.url}: {e}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "path": str(request.url),
                "timestamp": time.time()
            }
        )

