import sys
import os
import logging
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectrum_server.api import radios_work
from spectrum_server.api import sdr_control
from spectrum_server.api import spectrum_analysis
from spectrum_server.api import iq_capture
from spectrum_server.api import dsp_filters
from spectrum_server.api import dashboard
from spectrum_server.spectrum_stream import spectrum_stream
from spectrum_server.sdr_control import init_sdr_controller, get_sdr_controller
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from fastmcp.server.openapi import RouteMap, MCPType

logger = logging.getLogger(__name__)

fastapi_app = FastAPI(
    title="Spectrum Server MCP",
    description="MCP-enabled spectrum analysis server for Ettus B210 SDR with GPU-accelerated processing",
    version="1.1.0",
)

# Register API routers with MCP exposure
fastapi_app.include_router(radios_work.router, prefix="/radio", tags=["radio"])
fastapi_app.include_router(sdr_control.router, prefix="/sdr", tags=["sdr-control"])
fastapi_app.include_router(spectrum_analysis.router, prefix="/spectrum", tags=["spectrum-analysis"])
fastapi_app.include_router(iq_capture.router, prefix="/iq", tags=["iq-capture"])
fastapi_app.include_router(dsp_filters.router, prefix="/dsp", tags=["dsp-filters"])
fastapi_app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])


@fastapi_app.get("/health")
async def health():
    return "ok"


@fastapi_app.websocket("/ws/spectrum")
async def websocket_spectrum(websocket: WebSocket):
    """WebSocket channel that streams spectrum frames to all connected clients."""
    await spectrum_stream.connect(websocket)
    await websocket.send_json({"type": "info", "message": "connected to spectrum stream"})
    try:
        while True:
            try:
                # Keep the receive loop open to detect disconnects; replies to pings for liveness.
                payload = await websocket.receive_text()
                if payload.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    finally:
        await spectrum_stream.disconnect(websocket)


mcp = FastMCP.from_fastapi(app=fastapi_app, name="Spectrum MCP", route_maps=[RouteMap(pattern=r"^/health", mcp_type=MCPType.EXCLUDE)])
mcp_app = mcp.http_app()


@asynccontextmanager
async def combined_lifespan(app):
    """Combined lifespan that initializes both MCP and SDR controller."""
    logger.info("Starting spectrum server initialization...")

    # Initialize SDR controller
    try:
        logger.info("Initializing SDR controller...")
        controller = await init_sdr_controller()
        logger.info(f"SDR controller initialized: connected={controller._state.connected}, device={controller._state.device_name}")
    except Exception as e:
        logger.error(f"SDR controller initialization failed: {e}")

    # Start spectrum stream
    try:
        logger.info("Starting spectrum stream...")
        await spectrum_stream.start()
        logger.info("Spectrum stream started")
    except Exception as e:
        logger.error(f"Spectrum stream start failed: {e}")

    # Run the MCP app's lifespan
    async with mcp_app.lifespan(app):
        logger.info("Spectrum server ready")
        yield

    # Shutdown
    logger.info("Shutting down spectrum server...")
    try:
        await spectrum_stream.stop()
        controller = get_sdr_controller()
        await controller.close()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    logger.info("Spectrum server shutdown complete")


app = Starlette(
    routes=[
        Mount("/llm", app=mcp_app),
        Mount("/", app=fastapi_app),
    ],
    lifespan=combined_lifespan,
)
