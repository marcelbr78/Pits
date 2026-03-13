import os
import subprocess
from fastapi import APIRouter, Header, HTTPException, Body
from fastapi.responses import JSONResponse
from .state import SystemState

router = APIRouter(prefix="/mcp")

API_KEY = "pits-claude-bridge-2026"

def verify_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

@router.get("/status")
async def mcp_status(x_api_key: str = Header(None), shared_state: SystemState = None):
    verify_key(x_api_key)
    if not shared_state:
        raise HTTPException(status_code=500, detail="State not initialized")
    return shared_state.get_full_state()

@router.get("/files/{path:path}")
async def mcp_read_file(path: str, x_api_key: str = Header(None)):
    verify_key(x_api_key)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"path": path, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/write")
async def mcp_write_file(payload: dict = Body(...), x_api_key: str = Header(None)):
    verify_key(x_api_key)
    path = payload.get("path")
    content = payload.get("content")
    if not path:
        raise HTTPException(status_code=400, detail="Path required")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run")
async def mcp_run_command(payload: dict = Body(...), x_api_key: str = Header(None)):
    verify_key(x_api_key)
    command = payload.get("command")
    if not command:
        raise HTTPException(status_code=400, detail="Command required")
    try:
        # Warning: Security risk, but requested by user
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "code": result.returncode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt")
async def mcp_prompt(payload: dict = Body(...), x_api_key: str = Header(None)):
    verify_key(x_api_key)
    # Placeholder for instruction execution
    # In a real scenario, this might integrate with an internal agent
    instruction = payload.get("instruction")
    return {"status": "received", "instruction": instruction, "message": "Instruction received by PITS Bridge"}

@router.get("/git/log")
async def mcp_git_log(x_api_key: str = Header(None)):
    verify_key(x_api_key)
    try:
        result = subprocess.run("git log -n 10 --oneline", shell=True, capture_output=True, text=True)
        return {"log": result.stdout.splitlines()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/git/push")
async def mcp_git_push(payload: dict = Body(...), x_api_key: str = Header(None)):
    verify_key(x_api_key)
    message = payload.get("message", "update via MCP Bridge")
    try:
        subprocess.run("git add .", shell=True, check=True)
        subprocess.run(f'git commit -m "{message}"', shell=True, check=True)
        subprocess.run("git push", shell=True, check=True)
        return {"status": "success", "message": f"Committed and pushed: {message}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def setup_mcp_bridge(app, shared_state: SystemState):
    # Inject state into endpoints that need it
    # This is a simple way without complex DI for a bridge
    @app.middleware("http")
    async def add_state_to_request(request: Request, call_next):
        request.state.shared_state = shared_state
        return await call_next(request)
    
    app.include_router(router)
