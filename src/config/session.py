from fastapi import Request


async def get_session_id(request: Request) -> str:
    """Extracts the session ID from the request state"""
    return request.state.session_id
