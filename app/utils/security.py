from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

import app.utils.config as config

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the API key if it's configured

    If no API key is configured in the settings, this is a no-op.
    Otherwise, it checks that the request contains a valid API key.
    """
    if config.API_KEY and config.API_KEY != "":
        if api_key != config.API_KEY:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
            )
    return api_key
