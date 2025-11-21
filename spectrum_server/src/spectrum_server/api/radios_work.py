from fastapi import APIRouter, Query
from spectrum_server.radios_work import Radio
from spectrum_server.schema.radios import mongoOutput, timeWaitOutput, timeDict, tuneFrequencyOutput
from spectrum_server.spectrum_stream import SpectrumFrame, spectrum_stream

router = APIRouter()

@router.get(
    "/get-time",
    response_model=timeDict,
    description="get the time",
    operation_id="get_the_time",
    response_model_exclude_none=True, response_model_exclude_unset=True, response_model_exclude_defaults=True
)
async def get_time():
    radio = Radio()
    return radio.get_time()

@router.get(
    "/wait",
    response_model=timeWaitOutput,
    description="time to wait in seconds",
    operation_id="get_wait_success",
    response_model_exclude_none=True, response_model_exclude_unset=True, response_model_exclude_defaults=True
)
async def wait_time(seconds_to_wait: int = Query(..., description="Seconds to wait")):
    radio = Radio()
    return radio.wait_time(seconds_to_wait)

@router.get(
    "/freq",
    response_model=tuneFrequencyOutput,
    description="Tune Radio Frequency Receiver to Frequency in Hz",
    operation_id="get_tune_success",
    response_model_exclude_none=True, response_model_exclude_unset=True, response_model_exclude_defaults=True
)
async def tune_freq(center_freq_hz: int = Query(..., description="Frequnecy in Hz")):
    radio = Radio()
    return radio.tune_freq(center_freq_hz)
    
@router.get(
    "/mongo",
    response_model=mongoOutput,
    description="Retrieve a list of the last 10 reciever measurement results from mongo database.",
    operation_id="get_tune_results",
    response_model_exclude_none=True, response_model_exclude_unset=True, response_model_exclude_defaults=True
)
async def get_results(center_freq_hz: int = Query(..., description="Frequnecy in Hz")):
    radio = Radio()
    return radio.get_results(center_freq_hz)


@router.post(
    "/spectrum/publish",
    description="Publish a spectrum frame to all connected WebSocket subscribers.",
    operation_id="publish_spectrum_frame",
    status_code=202,
)
async def publish_spectrum_frame(frame: SpectrumFrame):
    """
    Ingest a computed spectrum frame (e.g., from the radio daemon) and broadcast it over the websocket stream.
    """
    await spectrum_stream.publish_frame(frame)
    return {"queued": True, "subscribers": spectrum_stream.connection_count}
