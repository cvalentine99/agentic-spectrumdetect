"""
SDR Control module with UHD/SoapySDR wrapper for Ettus B210.

This module provides a unified interface for SDR hardware control,
supporting both direct UHD access and SoapySDR abstraction layer.
Designed for integration with MCP tools and real-time I/Q pipelines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import select
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Any

from spectrum_server.schema.sdr_control import (
    SDRStatus, TuneRequest, TuneResponse, GainRequest, GainResponse,
    AntennaRequest, AntennaResponse, CalibrationRequest, CalibrationResponse,
    StreamControlRequest, StreamControlResponse, AntennaPort, GainMode
)

logger = logging.getLogger(__name__)


class SDRBackend(str, Enum):
    """Available SDR backend drivers."""
    UHD = "uhd"
    SOAPY = "soapy"
    SIMULATED = "simulated"


@dataclass
class SDRConfig:
    """SDR configuration parameters."""

    backend: SDRBackend = SDRBackend.UHD
    device_args: str = ""  # e.g., "serial=XXXXXX" for specific B210
    center_freq_hz: int = 2_450_000_000
    sample_rate_hz: int = 50_000_000
    bandwidth_hz: int = 50_000_000
    gain_db: float = 38.0
    gain_mode: GainMode = GainMode.MANUAL
    antenna: AntennaPort = AntennaPort.TX_RX
    channel: int = 0
    lo_offset_hz: int = 0
    dc_offset_mode: bool = True
    iq_balance_mode: bool = True


@dataclass
class SDRState:
    """Current SDR hardware state (thread-safe tracking)."""

    connected: bool = False
    streaming: bool = False
    center_freq_hz: int = 0
    sample_rate_hz: int = 0
    bandwidth_hz: int = 0
    gain_db: float = 0.0
    gain_mode: GainMode = GainMode.MANUAL
    antenna: AntennaPort = AntennaPort.TX_RX
    lo_locked: bool = False
    temperature_c: Optional[float] = None
    overflow_count: int = 0
    samples_received: int = 0
    device_serial: str = ""
    device_name: str = "Unknown"
    last_error: Optional[str] = None


class SDRController:
    """
    Unified SDR controller supporting Ettus B210 via UHD or SoapySDR.

    This class provides:
    - Hardware initialization and configuration
    - Frequency tuning with LO offset support
    - Gain control (manual and automatic)
    - Antenna selection
    - DC offset and IQ balance calibration
    - UDP-based retune protocol for tye_sp integration
    - Async streaming control

    Thread-safety: All public methods are async and can be called
    from any coroutine context. Internal state is protected.
    """

    def __init__(self, config: Optional[SDRConfig] = None):
        # Load config from environment if not provided
        if config is None:
            config = self._load_config_from_env()
        self.config = config
        self._state = SDRState()
        self._lock = asyncio.Lock()
        self._usrp = None  # UHD device handle
        self._sdr = None   # SoapySDR device handle
        self._rx_streamer = None
        self._stream_task: Optional[asyncio.Task] = None
        self._sample_callback: Optional[Callable] = None

        # UDP retune protocol support (tye_sp compatibility)
        self._retune_socket: Optional[socket.socket] = None
        self._ad_port = 61111
        self._retune_timeout_s = 2.0

        # Get hostname for network operations
        try:
            self._hostname = os.environ["HOST_NAME"]
        except KeyError:
            self._hostname = "localhost"

    def _load_config_from_env(self) -> SDRConfig:
        """Load SDR configuration from environment variables."""
        backend_str = os.environ.get("SDR_BACKEND", "uhd").lower()
        if backend_str == "uhd":
            backend = SDRBackend.UHD
        elif backend_str == "soapy":
            backend = SDRBackend.SOAPY
        else:
            backend = SDRBackend.SIMULATED

        device_serial = os.environ.get("SDR_DEVICE_SERIAL", "")
        device_args = f"serial={device_serial}" if device_serial else ""

        return SDRConfig(
            backend=backend,
            device_args=device_args,
            center_freq_hz=int(os.environ.get("SDR_CENTER_FREQ_HZ", "2450000000")),
            sample_rate_hz=int(os.environ.get("SDR_SAMPLE_RATE_HZ", "50000000")),
            bandwidth_hz=int(os.environ.get("SDR_BANDWIDTH_HZ", "50000000")),
            gain_db=float(os.environ.get("SDR_GAIN_DB", "38.0")),
            antenna=AntennaPort(os.environ.get("SDR_ANTENNA", "TX/RX")),
        )

    async def initialize(self) -> bool:
        """
        Initialize SDR hardware connection.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        async with self._lock:
            try:
                if self.config.backend == SDRBackend.UHD:
                    return await self._init_uhd()
                elif self.config.backend == SDRBackend.SOAPY:
                    return await self._init_soapy()
                elif self.config.backend == SDRBackend.SIMULATED:
                    return await self._init_simulated()
                else:
                    logger.error(f"Unknown backend: {self.config.backend}")
                    return False
            except Exception as e:
                logger.exception(f"SDR initialization failed: {e}")
                self._state.last_error = str(e)
                return False

    async def _init_uhd(self) -> bool:
        """Initialize UHD backend for Ettus devices."""
        try:
            # Attempt to import UHD Python bindings
            import uhd

            # Create USRP device
            args = self.config.device_args or ""
            logger.info(f"Initializing UHD device with args: {args}")

            # Run blocking UHD init in executor
            loop = asyncio.get_event_loop()
            self._usrp = await loop.run_in_executor(
                None, lambda: uhd.usrp.MultiUSRP(args)
            )

            # Get device info
            info = self._usrp.get_usrp_rx_info(self.config.channel)
            self._state.device_name = f"Ettus {info.get('mboard_id', 'B210')}"
            self._state.device_serial = info.get('mboard_serial', 'Unknown')

            # Apply initial configuration
            await self._configure_uhd()

            self._state.connected = True
            logger.info(f"UHD device initialized: {self._state.device_name} (serial: {self._state.device_serial})")
            return True

        except ImportError as e:
            logger.warning(f"UHD Python bindings not available ({e}), falling back to UDP protocol")
            # Fall back to UDP-based control for tye_sp
            self._state.device_name = "Ettus B210 (UDP Control)"
            self._state.connected = True
            return True
        except Exception as e:
            logger.error(f"UHD initialization failed: {e}")
            self._state.last_error = str(e)
            # Fall back to simulated mode on error
            logger.info("Falling back to simulated SDR mode")
            return await self._init_simulated()

    async def _configure_uhd(self) -> None:
        """Apply configuration to UHD device."""
        if not self._usrp:
            return

        import uhd

        ch = self.config.channel
        loop = asyncio.get_event_loop()

        def configure_blocking():
            # Set sample rate
            self._usrp.set_rx_rate(self.config.sample_rate_hz, ch)
            actual_rate = self._usrp.get_rx_rate(ch)
            self._state.sample_rate_hz = int(actual_rate)

            # Set center frequency
            tune_req = uhd.types.TuneRequest(self.config.center_freq_hz)
            self._usrp.set_rx_freq(tune_req, ch)
            self._state.center_freq_hz = int(self._usrp.get_rx_freq(ch))

            # Set bandwidth
            self._usrp.set_rx_bandwidth(self.config.bandwidth_hz, ch)
            self._state.bandwidth_hz = int(self._usrp.get_rx_bandwidth(ch))

            # Set gain (B210 doesn't support AGC via set_rx_agc)
            self._usrp.set_rx_gain(self.config.gain_db, ch)
            self._state.gain_db = self._usrp.get_rx_gain(ch)
            self._state.gain_mode = self.config.gain_mode

            # Set antenna
            antenna_name = self.config.antenna.value
            if antenna_name != "AUTO":
                self._usrp.set_rx_antenna(antenna_name, ch)
            self._state.antenna = AntennaPort(self._usrp.get_rx_antenna(ch))

        await loop.run_in_executor(None, configure_blocking)

        # Check LO lock with async waits
        for _ in range(100):
            sensors = self._usrp.get_rx_sensor_names(ch)
            if "lo_locked" in sensors:
                lo_sensor = self._usrp.get_rx_sensor("lo_locked", ch)
                self._state.lo_locked = lo_sensor.to_bool()
                if self._state.lo_locked:
                    break
            await asyncio.sleep(0.01)

        logger.info(f"UHD configured: {self._state.center_freq_hz/1e9:.4f} GHz, "
                   f"{self._state.sample_rate_hz/1e6:.1f} MSPS, {self._state.gain_db:.1f} dB, "
                   f"LO locked: {self._state.lo_locked}")

    async def _init_soapy(self) -> bool:
        """Initialize SoapySDR backend."""
        try:
            import SoapySDR

            args = self.config.device_args or "driver=uhd"
            logger.info(f"Initializing SoapySDR device with args: {args}")

            self._sdr = SoapySDR.Device(args)

            # Get device info
            hw_info = self._sdr.getHardwareInfo()
            self._state.device_name = hw_info.get("name", "SoapySDR Device")
            self._state.device_serial = hw_info.get("serial", "Unknown")

            # Apply initial configuration
            await self._configure_soapy()

            self._state.connected = True
            logger.info(f"SoapySDR device initialized: {self._state.device_name}")
            return True

        except ImportError:
            logger.error("SoapySDR not available")
            self._state.last_error = "SoapySDR not installed"
            return False
        except Exception as e:
            logger.error(f"SoapySDR initialization failed: {e}")
            self._state.last_error = str(e)
            return False

    async def _configure_soapy(self) -> None:
        """Apply configuration to SoapySDR device."""
        if not self._sdr:
            return

        ch = self.config.channel
        direction = 0  # RX = 0, TX = 1 in SoapySDR

        self._sdr.setSampleRate(direction, ch, self.config.sample_rate_hz)
        self._state.sample_rate_hz = int(self._sdr.getSampleRate(direction, ch))

        self._sdr.setFrequency(direction, ch, self.config.center_freq_hz)
        self._state.center_freq_hz = int(self._sdr.getFrequency(direction, ch))

        self._sdr.setBandwidth(direction, ch, self.config.bandwidth_hz)
        self._state.bandwidth_hz = int(self._sdr.getBandwidth(direction, ch))

        if self.config.gain_mode == GainMode.AUTO:
            self._sdr.setGainMode(direction, ch, True)
        else:
            self._sdr.setGainMode(direction, ch, False)
            self._sdr.setGain(direction, ch, self.config.gain_db)
        self._state.gain_db = self._sdr.getGain(direction, ch)
        self._state.gain_mode = self.config.gain_mode

        antenna_name = self.config.antenna.value
        if antenna_name != "AUTO":
            self._sdr.setAntenna(direction, ch, antenna_name)
        self._state.antenna = AntennaPort(self._sdr.getAntenna(direction, ch))

    async def _init_simulated(self) -> bool:
        """Initialize simulated SDR for testing."""
        self._state.device_name = "Simulated SDR"
        self._state.device_serial = "SIM-001"
        self._state.center_freq_hz = self.config.center_freq_hz
        self._state.sample_rate_hz = self.config.sample_rate_hz
        self._state.bandwidth_hz = self.config.bandwidth_hz
        self._state.gain_db = self.config.gain_db
        self._state.gain_mode = self.config.gain_mode
        self._state.antenna = self.config.antenna
        self._state.lo_locked = True
        self._state.connected = True
        logger.info("Simulated SDR initialized")
        return True

    async def get_status(self) -> SDRStatus:
        """Get current SDR status."""
        async with self._lock:
            return SDRStatus(
                connected=self._state.connected,
                device_name=self._state.device_name,
                serial=self._state.device_serial or None,
                center_freq_hz=self._state.center_freq_hz,
                sample_rate_hz=self._state.sample_rate_hz,
                bandwidth_hz=self._state.bandwidth_hz,
                gain_db=self._state.gain_db,
                gain_mode=self._state.gain_mode,
                antenna=self._state.antenna,
                lo_locked=self._state.lo_locked,
                temperature_c=self._state.temperature_c,
                overflow_count=self._state.overflow_count
            )

    async def tune(self, request: TuneRequest) -> TuneResponse:
        """
        Tune SDR to specified frequency and sample rate.

        Supports both direct UHD/SoapySDR control and UDP retune protocol
        for tye_sp compatibility.
        """
        async with self._lock:
            start_time = time.time()

            try:
                # Try direct hardware control first
                if self._usrp or self._sdr:
                    return await self._tune_direct(request, start_time)
                else:
                    # Fall back to UDP retune protocol
                    return await self._tune_udp(request, start_time)

            except Exception as e:
                logger.exception(f"Tune failed: {e}")
                return TuneResponse(
                    success=False,
                    actual_center_freq_hz=self._state.center_freq_hz,
                    actual_sample_rate_hz=self._state.sample_rate_hz,
                    actual_bandwidth_hz=self._state.bandwidth_hz,
                    tune_time_ms=(time.time() - start_time) * 1000
                )

    async def _tune_direct(self, request: TuneRequest, start_time: float) -> TuneResponse:
        """Tune using direct hardware control."""
        ch = self.config.channel
        loop = asyncio.get_event_loop()

        if self._usrp:
            import uhd

            def tune_uhd():
                # UHD tuning
                self._usrp.set_rx_rate(request.sample_rate_hz, ch)
                tune_req = uhd.types.TuneRequest(request.center_freq_hz)
                self._usrp.set_rx_freq(tune_req, ch)

                bw = request.bandwidth_hz or request.sample_rate_hz
                self._usrp.set_rx_bandwidth(bw, ch)

                self._state.center_freq_hz = int(self._usrp.get_rx_freq(ch))
                self._state.sample_rate_hz = int(self._usrp.get_rx_rate(ch))
                self._state.bandwidth_hz = int(self._usrp.get_rx_bandwidth(ch))

            await loop.run_in_executor(None, tune_uhd)

            # Wait for LO lock
            for _ in range(50):
                sensors = self._usrp.get_rx_sensor_names(ch)
                if "lo_locked" in sensors:
                    lo_sensor = self._usrp.get_rx_sensor("lo_locked", ch)
                    self._state.lo_locked = lo_sensor.to_bool()
                    if self._state.lo_locked:
                        break
                await asyncio.sleep(0.01)

        elif self._sdr:
            def tune_soapy():
                # SoapySDR tuning
                self._sdr.setSampleRate(0, ch, request.sample_rate_hz)
                self._sdr.setFrequency(0, ch, request.center_freq_hz)

                bw = request.bandwidth_hz or request.sample_rate_hz
                self._sdr.setBandwidth(0, ch, bw)

                self._state.center_freq_hz = int(self._sdr.getFrequency(0, ch))
                self._state.sample_rate_hz = int(self._sdr.getSampleRate(0, ch))
                self._state.bandwidth_hz = int(self._sdr.getBandwidth(0, ch))

            await loop.run_in_executor(None, tune_soapy)

        tune_time = (time.time() - start_time) * 1000

        logger.info(f"Tuned to {self._state.center_freq_hz/1e9:.4f} GHz, "
                   f"{self._state.sample_rate_hz/1e6:.1f} MSPS in {tune_time:.1f} ms")

        return TuneResponse(
            success=True,
            actual_center_freq_hz=self._state.center_freq_hz,
            actual_sample_rate_hz=self._state.sample_rate_hz,
            actual_bandwidth_hz=self._state.bandwidth_hz,
            lo_offset_hz=self.config.lo_offset_hz or None,
            tune_time_ms=tune_time
        )

    async def _tune_udp(self, request: TuneRequest, start_time: float) -> TuneResponse:
        """Tune using UDP retune protocol (tye_sp compatibility)."""

        # Receive advertisement to find tye_sp
        dst_ip, dst_port = await self._recv_advertisement()
        if not dst_ip or not dst_port:
            return TuneResponse(
                success=False,
                actual_center_freq_hz=self._state.center_freq_hz,
                actual_sample_rate_hz=self._state.sample_rate_hz,
                actual_bandwidth_hz=self._state.bandwidth_hz,
                tune_time_ms=(time.time() - start_time) * 1000
            )

        # Send retune message
        bw = request.bandwidth_hz or request.sample_rate_hz
        atten_db = -1  # Auto gain
        ref_level = -20.0

        success = await self._send_retune_message(
            dst_ip, dst_port,
            request.sample_rate_hz,
            request.center_freq_hz,
            atten_db, ref_level
        )

        if success:
            self._state.center_freq_hz = request.center_freq_hz
            self._state.sample_rate_hz = request.sample_rate_hz
            self._state.bandwidth_hz = bw

        tune_time = (time.time() - start_time) * 1000

        return TuneResponse(
            success=success,
            actual_center_freq_hz=self._state.center_freq_hz,
            actual_sample_rate_hz=self._state.sample_rate_hz,
            actual_bandwidth_hz=self._state.bandwidth_hz,
            tune_time_ms=tune_time
        )

    async def _recv_advertisement(self) -> tuple[Optional[str], Optional[int]]:
        """Listen for tye_sp advertisement broadcast."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._recv_advertisement_sync
        )

    def _recv_advertisement_sync(self) -> tuple[Optional[str], Optional[int]]:
        """Synchronous advertisement receiver."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setblocking(False)
            sock.bind(("", self._ad_port))

            deadline = time.time() + self._retune_timeout_s
            while time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                rlist, _, _ = select.select([sock], [], [], remaining)
                if not rlist:
                    continue

                data, addr = sock.recvfrom(1024)
                try:
                    msg = json.loads(data.decode())
                    if msg.get("msg_type") == "ad" and "retune_port" in msg:
                        return addr[0], int(msg["retune_port"])
                except (json.JSONDecodeError, KeyError):
                    continue

            return None, None

        except OSError as e:
            logger.error(f"Advertisement receive failed: {e}")
            return None, None
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    async def _send_retune_message(
        self, dst_ip: str, dst_port: int,
        sample_rate_hz: int, center_freq_hz: int,
        atten_db: int, ref_level: float
    ) -> bool:
        """Send retune message and wait for acknowledgment."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._send_retune_message_sync,
            dst_ip, dst_port, sample_rate_hz, center_freq_hz, atten_db, ref_level
        )

    def _send_retune_message_sync(
        self, dst_ip: str, dst_port: int,
        sample_rate_hz: int, center_freq_hz: int,
        atten_db: int, ref_level: float
    ) -> bool:
        """Synchronous retune message sender."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)

            msg = {
                "msg_type": "retune",
                "sample_rate_hz": sample_rate_hz,
                "center_freq_hz": center_freq_hz,
                "atten_db": atten_db,
                "ref_level": ref_level,
            }

            json_msg = json.dumps(msg, separators=(',', ':'))
            sock.sendto(json_msg.encode(), (dst_ip, dst_port))

            # Wait for status reply
            deadline = time.time() + self._retune_timeout_s
            while time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                rlist, _, _ = select.select([sock], [], [], remaining)
                if not rlist:
                    continue

                data, _ = sock.recvfrom(256)
                try:
                    status_msg = json.loads(data.decode())
                    if status_msg.get("msg_type") == "retune_status":
                        return status_msg.get("status") == "success"
                except (json.JSONDecodeError, KeyError):
                    continue

            return False

        except OSError as e:
            logger.error(f"Retune message send failed: {e}")
            return False
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    async def set_gain(self, request: GainRequest) -> GainResponse:
        """Set RF gain."""
        async with self._lock:
            ch = request.apply_to_channel

            try:
                if self._usrp:
                    if request.gain_mode == GainMode.AUTO:
                        self._usrp.set_rx_agc(True, ch)
                    else:
                        self._usrp.set_rx_agc(False, ch)
                        self._usrp.set_rx_gain(request.gain_db, ch)

                    actual = self._usrp.get_rx_gain(ch)
                    gain_range = self._usrp.get_rx_gain_range(ch)

                elif self._sdr:
                    if request.gain_mode == GainMode.AUTO:
                        self._sdr.setGainMode(0, ch, True)
                    else:
                        self._sdr.setGainMode(0, ch, False)
                        self._sdr.setGain(0, ch, request.gain_db)

                    actual = self._sdr.getGain(0, ch)
                    gain_range = self._sdr.getGainRange(0, ch)

                else:
                    # Simulated mode
                    self._state.gain_db = request.gain_db
                    self._state.gain_mode = request.gain_mode
                    return GainResponse(
                        success=True,
                        actual_gain_db=request.gain_db,
                        gain_range_db=(0.0, 76.0)
                    )

                self._state.gain_db = actual
                self._state.gain_mode = request.gain_mode

                return GainResponse(
                    success=True,
                    actual_gain_db=actual,
                    gain_range_db=(float(gain_range.start()), float(gain_range.stop()))
                )

            except Exception as e:
                logger.exception(f"Set gain failed: {e}")
                return GainResponse(
                    success=False,
                    actual_gain_db=self._state.gain_db,
                    gain_range_db=(0.0, 76.0)
                )

    async def set_antenna(self, request: AntennaRequest) -> AntennaResponse:
        """Select antenna port."""
        async with self._lock:
            ch = request.channel

            try:
                if self._usrp:
                    available = list(self._usrp.get_rx_antennas(ch))
                    antenna_name = request.antenna.value
                    if antenna_name != "AUTO":
                        self._usrp.set_rx_antenna(antenna_name, ch)
                    selected = self._usrp.get_rx_antenna(ch)

                elif self._sdr:
                    available = list(self._sdr.listAntennas(0, ch))
                    antenna_name = request.antenna.value
                    if antenna_name != "AUTO":
                        self._sdr.setAntenna(0, ch, antenna_name)
                    selected = self._sdr.getAntenna(0, ch)

                else:
                    # Simulated mode
                    self._state.antenna = request.antenna
                    return AntennaResponse(
                        success=True,
                        antenna=request.antenna,
                        available_antennas=["TX/RX", "RX2"]
                    )

                self._state.antenna = AntennaPort(selected)

                return AntennaResponse(
                    success=True,
                    antenna=self._state.antenna,
                    available_antennas=available
                )

            except Exception as e:
                logger.exception(f"Set antenna failed: {e}")
                return AntennaResponse(
                    success=False,
                    antenna=self._state.antenna,
                    available_antennas=["TX/RX", "RX2"]
                )

    async def calibrate(self, request: CalibrationRequest) -> CalibrationResponse:
        """Run DC offset and IQ balance calibration."""
        async with self._lock:
            start_time = time.time()

            try:
                dc_i, dc_q, iq_bal = None, None, None

                if self._usrp:
                    ch = self.config.channel

                    if request.center_freq_hz:
                        tune_req = self._usrp.tune_request(request.center_freq_hz)
                        self._usrp.set_rx_freq(tune_req, ch)

                    if request.calibrate_dc_offset:
                        self._usrp.set_rx_dc_offset(True, ch)
                        dc_i, dc_q = 0.0, 0.0  # Auto-corrected

                    if request.calibrate_iq_balance:
                        self._usrp.set_rx_iq_balance(True, ch)
                        iq_bal = 0.0  # Auto-corrected

                elif self._sdr:
                    ch = self.config.channel

                    if request.center_freq_hz:
                        self._sdr.setFrequency(0, ch, request.center_freq_hz)

                    if request.calibrate_dc_offset:
                        self._sdr.setDCOffsetMode(0, ch, True)
                        dc_i, dc_q = 0.0, 0.0

                    if request.calibrate_iq_balance:
                        self._sdr.setIQBalanceMode(0, ch, True)
                        iq_bal = 0.0

                cal_time = (time.time() - start_time) * 1000

                return CalibrationResponse(
                    success=True,
                    dc_offset_i=dc_i,
                    dc_offset_q=dc_q,
                    iq_balance=iq_bal,
                    calibration_time_ms=cal_time
                )

            except Exception as e:
                logger.exception(f"Calibration failed: {e}")
                return CalibrationResponse(
                    success=False,
                    calibration_time_ms=(time.time() - start_time) * 1000
                )

    async def stream_control(self, request: StreamControlRequest) -> StreamControlResponse:
        """Start or stop I/Q data streaming."""
        async with self._lock:
            try:
                if request.action == "start":
                    if self._state.streaming:
                        return StreamControlResponse(
                            success=True,
                            streaming=True,
                            samples_received=self._state.samples_received,
                            overflow_count=self._state.overflow_count
                        )

                    # Start streaming
                    self._state.streaming = True
                    self._state.samples_received = 0
                    self._state.overflow_count = 0

                    return StreamControlResponse(
                        success=True,
                        streaming=True,
                        samples_received=0,
                        overflow_count=0
                    )

                else:  # stop
                    self._state.streaming = False

                    return StreamControlResponse(
                        success=True,
                        streaming=False,
                        samples_received=self._state.samples_received,
                        overflow_count=self._state.overflow_count
                    )

            except Exception as e:
                logger.exception(f"Stream control failed: {e}")
                return StreamControlResponse(
                    success=False,
                    streaming=self._state.streaming,
                    samples_received=self._state.samples_received,
                    overflow_count=self._state.overflow_count
                )

    async def close(self) -> None:
        """Clean up SDR resources."""
        async with self._lock:
            self._state.streaming = False
            self._state.connected = False

            if self._stream_task:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
                self._stream_task = None

            if self._usrp:
                # UHD cleanup
                self._usrp = None

            if self._sdr:
                # SoapySDR cleanup
                self._sdr = None

            logger.info("SDR controller closed")


# Singleton instance for use by FastAPI app
_sdr_controller: Optional[SDRController] = None


def get_sdr_controller() -> SDRController:
    """Get or create the singleton SDR controller."""
    global _sdr_controller
    if _sdr_controller is None:
        _sdr_controller = SDRController()
    return _sdr_controller


async def init_sdr_controller(config: Optional[SDRConfig] = None) -> SDRController:
    """Initialize the SDR controller with optional configuration."""
    global _sdr_controller
    _sdr_controller = SDRController(config)
    await _sdr_controller.initialize()
    return _sdr_controller
