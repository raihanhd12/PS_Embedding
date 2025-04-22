from fastapi import Request

async def session_middleware(request: Request, call_next):
    # Get session ID from cookie or header
    session_id = request.cookies.get("session_id") or request.headers.get(
        "X-Session-ID"
    )

    # Store in request state (use a default if none provided)
    request.state.session_id = session_id or "default-session"

    # Continue processing the request
    response = await call_next(request)
    return response
