from fastapi import HTTPException, Request, status

from notarius_core.domain.identity import ActorContext


async def browser_actor(request: Request) -> ActorContext:
    if "authorization" in request.headers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Browser routes accept cookie authentication only",
        )
    return await request.app.state.auth_service.require_browser_actor(request)
