"""Depth-based RabbitMQ worker for ``alarm_inference`` messages.

Run ``python analyze_glass_and_distance.py --worker-config worker.json``.
Incoming bboxes use ``[center_x, center_y, width, height]`` in original-image
pixels. Result codes are: 0 no child detected,
1 behind glass, 2 in front of glass, (alone events only)
3 too far from kid, 4 close to kid. (hit or shake events only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sqlite3
import ssl
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from depth_anything_3.detect_person_behind_glass import classify_person_against_reference
from depth_anything_3.measure_location_distance import (
    camera_point,
    load_intrinsics_npy,
    resolve_intrinsics,
    undistort_point,
)
from depth_anything_3.utils.depth_analysis import (
    infer_depth_from_bgr, infer_depth_from_path, load_bgr_image, load_depth_map, load_depth_model,
    robust_patch_depth,
)

LOG = logging.getLogger(__name__)
RESULTS = {0: "no_kid_detected", 1: "behind_glass", 2: "in_front_of_glass",
           3: "too_far_from_kid", 4: "close_to_a_kid"}


class WorkerState:
    """Durable outgoing-result outbox plus a completed-message idempotency DB."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.outbox = self.directory / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.directory / "processed.sqlite3"), check_same_thread=False)
        self.connection.execute("CREATE TABLE IF NOT EXISTS processed (message_id TEXT PRIMARY KEY, completed_at REAL NOT NULL)")
        self.connection.commit()
        self.lock = threading.Lock()
        self.metrics: dict[str, int] = {"received": 0, "successful": 0, "errors": 0, "reconnects": 0, "outbox_replayed": 0, "dead_lettered": 0}
        self.status = "starting"
        self.last_error: str | None = None

    def increment(self, name: str) -> None:
        with self.lock:
            self.metrics[name] = self.metrics.get(name, 0) + 1

    def is_processed(self, message_id: str | None) -> bool:
        if not message_id:
            return False
        with self.lock:
            return self.connection.execute("SELECT 1 FROM processed WHERE message_id = ?", (message_id,)).fetchone() is not None

    def mark_processed(self, message_id: str | None) -> None:
        if message_id:
            with self.lock:
                self.connection.execute("INSERT OR IGNORE INTO processed(message_id, completed_at) VALUES (?, ?)", (message_id, time.time()))
                self.connection.commit()

    def add_outbox(self, result: dict[str, Any], correlation_id: str | None, source_message_id: str | None) -> Path:
        # The source ID keeps retries idempotent; the result ID is a fresh,
        # public output-message ID and must not be used for deduplication.
        identity = str(source_message_id or result.get("message_id") or hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest())
        name = hashlib.sha256(identity.encode()).hexdigest() + ".json"
        target = self.outbox / name
        envelope = {"result": result, "correlation_id": correlation_id, "source_message_id": source_message_id}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(envelope, allow_nan=False), encoding="utf-8")
        temporary.replace(target)  # Atomic on the local filesystem.
        return target

    def pending(self) -> list[Path]:
        return sorted(self.outbox.glob("*.json"))

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {"status": self.status, "last_error": self.last_error, "metrics": dict(self.metrics), "pending_outbox": len(self.pending())}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def mapping(config: dict[str, Any], config_path: Path, name: str) -> dict[str, Any]:
    """Accept either a direct camera mapping or a path to its JSON file."""
    source = config.get(name)
    if source is None:
        raise KeyError(f"Missing required config key: {name}")
    data = load_json(resolve(config_path, source)) if isinstance(source, str) else source
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    return data.get("cameras", data)


def camera_config(config: dict[str, Any], config_path: Path) -> dict[str, dict[str, Any]]:
    """Load the single camera-code mapping used by this worker.

    ``camera_config`` may be an inline object or a path to a JSON file.  Each
    camera entry contains reference depth, intrinsics, and distortion paths.
    """
    cameras = mapping(config, config_path, "camera_config")
    if not all(isinstance(value, dict) for value in cameras.values()):
        raise ValueError("Every camera_config entry must be a JSON object")
    return cameras


def bbox_center_wh(item: dict[str, Any]) -> tuple[float, float, float, float]:
    box = item.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox must be [center_x, center_y, width, height]")
    center_x, center_y, w, h = map(float, box)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox: {box}")
    return center_x, center_y, w, h


def bbox_center_image(
    box: dict[str, Any], image_shape: tuple[int, int]
) -> tuple[float, float]:
    """Return an incoming bbox centre in original-image pixels."""
    center_x, center_y, _, _ = bbox_center_wh(box)
    image_h, image_w = image_shape
    center = (center_x, center_y)
    if not (0 <= center[0] < image_w and 0 <= center[1] < image_h):
        raise ValueError(f"BBox centre {center} is outside image size {[image_w, image_h]}")
    return center


def bbox_mask(box: dict[str, Any], image_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    """Create a depth-map mask from an original-image-pixel center bbox."""
    center_x, center_y, w, h = bbox_center_wh(box)
    image_h, image_w = image_shape
    depth_h, depth_w = depth_shape
    x1 = int(np.floor((center_x - w / 2) * depth_w / image_w))
    x2 = int(np.ceil((center_x + w / 2) * depth_w / image_w))
    y1 = int(np.floor((center_y - h / 2) * depth_h / image_h))
    y2 = int(np.ceil((center_y + h / 2) * depth_h / image_h))
    x1, x2 = max(0, x1), min(depth_w, x2)
    y1, y2 = max(0, y1), min(depth_h, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"BBox outside depth image: {box['bbox']}")
    mask = np.zeros(depth_shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def sftp_download(config: dict[str, Any], config_path: Path, image_ref: str) -> Path:
    """Download an image reference (path or sftp URL) to a temporary file."""
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("Install worker dependency: pip install paramiko") from error
    sftp_config = config.get("sftp", {})
    parsed = urlparse(image_ref)
    host = parsed.hostname or sftp_config.get("host")
    if not host:
        raise ValueError("sftp.host is required")
    remote = parsed.path if parsed.scheme == "sftp" or image_ref.startswith("/") else str(Path(sftp_config.get("remote_base_path", ".")) / image_ref)
    fd, filename = tempfile.mkstemp(prefix="alarm-image-", suffix=Path(remote).suffix or ".jpg")
    os.close(fd)
    local = Path(filename)
    transport = client = None
    try:
        timeout = float(sftp_config.get("timeout_seconds", 30))
        sock = socket.create_connection((host, parsed.port or int(sftp_config.get("port", 22))), timeout=timeout)
        transport = paramiko.Transport(sock)
        args: dict[str, Any] = {"username": parsed.username or sftp_config.get("username")}
        if parsed.password or sftp_config.get("password"):
            args["password"] = parsed.password or sftp_config["password"]
        elif sftp_config.get("private_key_path"):
            args["pkey"] = paramiko.RSAKey.from_private_key_file(str(resolve(config_path, sftp_config["private_key_path"])))
        transport.connect(**args)
        client = paramiko.SFTPClient.from_transport(transport)
        client.get_channel().settimeout(timeout)
        remote_size = int(client.stat(remote).st_size)
        max_size = int(sftp_config.get("max_image_bytes", 25 * 1024 * 1024))
        if remote_size <= 0 or remote_size > max_size:
            raise ValueError(f"SFTP image size {remote_size} is outside allowed range 1..{max_size} bytes")
        client.get(remote, str(local))
        expected_hash = sftp_config.get("sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(local.read_bytes()).hexdigest()
            if actual_hash.lower() != str(expected_hash).lower():
                raise ValueError("SFTP image SHA-256 does not match sftp.sha256")
        return local
    except Exception:
        local.unlink(missing_ok=True)
        raise
    finally:
        if client: client.close()
        if transport: transport.close()


class AlarmAnalyzer:
    """Loads DA3 once, then processes one alarm message at a time."""
    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config, self.config_path = config, config_path
        self.hyper = config.get("hyperparameters", {})
        self.person_id = int(self.hyper.get("person_yolo_id", 1))
        self.patch_size = int(self.hyper.get("patch_size", 15))
        if self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("hyperparameters.patch_size must be positive and odd")
        model_cfg = config.get("depth_model", {})
        self.process_res = int(model_cfg.get("process_res", 504))
        self.model_dir = model_cfg.get("model_dir", "depth-anything/DA3METRIC-LARGE")
        self.device = model_cfg.get("device", "cuda")
        self.model = None

    def depth_model(self) -> Any:
        """Initialize DA3 only when an alarm actually needs depth inference."""
        if self.model is None:
            self.model = load_depth_model(self.model_dir, self.device)
        return self.model

    def camera(self, camera_code: str) -> dict[str, Any]:
        try:
            entry = camera_config(self.config, self.config_path)[camera_code]
        except KeyError as error:
            raise KeyError(f"No camera configuration for camera_code '{camera_code}'") from error
        for key in ("reference_depth_path", "intrinsics_path", "distortion_path", "image_size"):
            if key not in entry:
                raise KeyError(f"Camera '{camera_code}' is missing '{key}'")
        return entry

    def validate_cameras(self) -> None:
        """Fail fast for bad calibration/reference files instead of on an alarm."""
        for camera_code, entry in camera_config(self.config, self.config_path).items():
            if not isinstance(entry.get("image_size"), list) or len(entry["image_size"]) != 2:
                raise ValueError(f"Camera '{camera_code}' image_size must be [width, height]")
            if any(int(value) <= 0 for value in entry["image_size"]):
                raise ValueError(f"Camera '{camera_code}' has invalid image_size")
            self.calibration(camera_code)
            reference = load_depth_map(resolve(self.config_path, entry["reference_depth_path"]))
            if not np.all(np.isfinite(reference)) or not np.any(reference > 0):
                raise ValueError(f"Camera '{camera_code}' reference depth has no valid positive values")

    def calibration(self, camera_code: str) -> tuple[list[float], np.ndarray]:
        entry = self.camera(camera_code)
        intrinsics = load_intrinsics_npy(resolve(self.config_path, entry["intrinsics_path"]))
        distortion = np.asarray(
            np.load(resolve(self.config_path, entry["distortion_path"]), allow_pickle=False),
            dtype=np.float64,
        )
        if distortion.size < 4 or not np.all(np.isfinite(distortion)):
            raise ValueError(f"Invalid distortion coefficients for camera '{camera_code}'")
        return intrinsics, distortion

    def analyze(self, message: dict[str, Any], local_image: Path | None = None) -> dict[str, Any]:
        """Process one message; ``local_image`` bypasses SFTP for local tests."""
        boxes = message.get("bboxes", [])
        if not isinstance(boxes, list): raise ValueError("bboxes must be a list")
        people = [box for box in boxes if box.get("yolo_id") == self.person_id]
        # This is the public RabbitMQ output contract. Camera and event are
        # intentionally used only internally and are never published.
        common = {
            "message_id": str(uuid.uuid4()),
            "alarm_message_id": message.get("alarm_message_id"),
        }
        camera_code = message.get("camera", {}).get("camera_code")
        if not people:
            return {**common, "result_code": 0, "result": RESULTS[0], "details": {"person_yolo_id": self.person_id}}
        if not camera_code: raise ValueError("camera.camera_code is required")
        if local_image is None:
            if not isinstance(message.get("original_image"), str): raise ValueError("original_image is required")
            image = sftp_download(self.config, self.config_path, message["original_image"])
            delete_image = True
        else:
            image = local_image.resolve()
            if not image.is_file():
                raise FileNotFoundError(f"Local test image does not exist: {image}")
            delete_image = False
        try:
            frame = load_bgr_image(image)
            expected_size = self.camera(camera_code)["image_size"]
            actual_size = [int(frame.shape[1]), int(frame.shape[0])]
            if actual_size != [int(expected_size[0]), int(expected_size[1])]:
                raise ValueError(
                    f"Camera {camera_code} image size {actual_size} does not match "
                    f"calibration image_size {expected_size}"
                )
            event = str(message.get("event", "")).lower()
            if event in {"alone", "single", "single_person"}:
                # The reference depth is expected to have been generated from
                # the original (not undistorted) camera frame.
                depth, _ = infer_depth_from_path(self.depth_model(), image, self.process_res)
                return self.classify_glass(common, camera_code, people, frame.shape[:2], depth)
            if event in {"hit", "hitting", "shake", "shaking"}:
                intrinsics, distortion = self.calibration(camera_code)
                fx, fy, cx, cy = intrinsics
                camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
                undistorted = cv2.undistort(frame, camera_matrix, distortion, None, camera_matrix)
                depth = infer_depth_from_bgr(self.depth_model(), undistorted, self.process_res)
                return self.measure_distance(
                    common, people, boxes, frame.shape[:2], depth, intrinsics, distortion
                )
            raise ValueError("event must be alone, hitting, or shaking")
        finally:
            if delete_image:
                image.unlink(missing_ok=True)

    def classify_glass(self, common: dict[str, Any], camera_code: str, people: list[dict[str, Any]], image_shape: tuple[int, int], depth: np.ndarray) -> dict[str, Any]:
        reference = load_depth_map(resolve(self.config_path, self.camera(camera_code)["reference_depth_path"]))
        if reference.shape != depth.shape:
            reference = cv2.resize(reference, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        details = [classify_person_against_reference(reference, depth, bbox_mask(person, image_shape, depth.shape), float(self.hyper.get("depth_threshold_m", .2)), float(self.hyper.get("min_front_fraction", .6)), int(self.hyper.get("erode_pixels", 5)), int(self.hyper.get("min_valid_pixels", 100))) for person in people]
        code = 1 if any(item["classification"] == "behind_glass" for item in details) else 2
        return {**common, "result_code": code, "result": RESULTS[code], "details": {"people": details}}

    def measure_distance(self, common: dict[str, Any], people: list[dict[str, Any]], boxes: list[dict[str, Any]], image_shape: tuple[int, int], depth: np.ndarray, intrinsics: list[float], distortion: np.ndarray) -> dict[str, Any]:
        alarms = [box for box in boxes if box.get("is_alarm") is True]
        if not alarms: raise ValueError("hitting/shaking message needs an is_alarm=true bbox")
        matrix, _ = resolve_intrinsics(None, intrinsics, image_shape[1], image_shape[0], depth.shape[1], depth.shape[0])
        if matrix is None: raise ValueError("Camera intrinsics could not be resolved")
        def point(box: dict[str, Any]) -> np.ndarray:
            source_center = bbox_center_image(box, image_shape)
            corrected = undistort_point(source_center, intrinsics, distortion)
            center = (corrected[0] * depth.shape[1] / image_shape[1], corrected[1] * depth.shape[0] / image_shape[0])
            return camera_point(center, robust_patch_depth(depth, center, self.patch_size), matrix)
        points = point
        distance = min(float(np.linalg.norm(points(person) - points(alarm))) for person in people for alarm in alarms)
        threshold = float(self.hyper.get("distance_threshold_m", 1.5))
        code = 3 if distance > threshold else 4
        return {**common, "result_code": code, "result": RESULTS[code], "details": {"3d_distance_m": distance, "distance_threshold_m": threshold, "intrinsics_source": "camera_config", "distortion_applied": True}}


def start_health_server(state: WorkerState, port: int | None, host: str = "127.0.0.1") -> None:
    """Expose JSON health and counters without adding a web-framework dependency."""
    if port is None:
        return

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required stdlib handler name
            if self.path not in {"/health", "/metrics"}:
                self.send_error(404)
                return
            body = json.dumps(state.health()).encode("utf-8")
            self.send_response(200 if state.status == "connected" else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="alarm-health").start()
    LOG.info("health and metrics endpoint listening on port %s", port)


def rabbit_parameters(pika: Any, rabbit: dict[str, Any]) -> Any:
    username = os.getenv(str(rabbit.get("username_env", ""))) or rabbit.get("username")
    password = os.getenv(str(rabbit.get("password_env", ""))) or rabbit.get("password")
    if not username or not password:
        raise ValueError("RabbitMQ credentials must be configured with username_env/password_env (preferred) or username/password")
    tls = rabbit.get("tls", {})
    ssl_options = None
    if tls.get("enabled", False):
        context = ssl.create_default_context(cafile=tls.get("ca_certificate"))
        if tls.get("client_certificate"):
            context.load_cert_chain(tls["client_certificate"], tls.get("client_key"))
        if tls.get("verify_server", True) is False:
            raise ValueError("TLS server verification must not be disabled")
        ssl_options = pika.SSLOptions(context, tls.get("server_hostname") or rabbit["host"])
    return pika.ConnectionParameters(
        host=rabbit["host"], port=int(rabbit.get("port", 5671 if ssl_options else 5672)),
        virtual_host=rabbit.get("virtual_host", "/"),
        credentials=pika.PlainCredentials(username, password),
        heartbeat=int(rabbit.get("heartbeat", 300)),
        blocked_connection_timeout=float(rabbit.get("blocked_connection_timeout_seconds", 60)),
        ssl_options=ssl_options,
    )


def error_result(body: bytes, error: Exception) -> dict[str, Any]:
    """A structured result for terminal failures that are sent to the DLQ."""
    try:
        message = json.loads(body.decode("utf-8"))
    except Exception:
        message = {}
    return {
        "message_id": str(uuid.uuid4()), "alarm_message_id": message.get("alarm_message_id"),
        "result_code": "error", "result": "analysis_failed",
        "details": {"error_type": type(error).__name__, "error": str(error)},
    }


def run_worker(config_path: Path) -> None:
    try:
        import pika
    except ImportError as error:
        raise RuntimeError("Install worker dependency: pip install pika") from error
    config_path, config = config_path.resolve(), load_json(config_path.resolve())
    rabbit = config.get("rabbitmq", {})
    required = ("host", "input_queue", "output_queue")
    if missing := [key for key in required if key not in rabbit]:
        raise KeyError(f"Missing rabbitmq keys: {', '.join(missing)}")
    state = WorkerState(resolve(config_path, config.get("local_state_dir", "alarm_worker_state")))
    start_health_server(state, config.get("health_port"), str(config.get("health_host", "127.0.0.1")))
    analyzer = AlarmAnalyzer(config, config_path)
    analyzer.validate_cameras()
    parameters = rabbit_parameters(pika, rabbit)
    maximum_retries = int(rabbit.get("max_retries", 5))
    reconnect_delay = float(rabbit.get("reconnect_initial_seconds", 1))
    reconnect_max = float(rabbit.get("reconnect_max_seconds", 60))
    dlx = rabbit.get("dead_letter_exchange", "alarm.dlx")
    dlq = rabbit.get("dead_letter_queue", f"{rabbit['input_queue']}.dead")

    def connect() -> tuple[Any, Any]:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.exchange_declare(exchange=dlx, exchange_type="direct", durable=True)
        channel.queue_declare(queue=dlq, durable=True)
        channel.queue_bind(queue=dlq, exchange=dlx, routing_key=dlq)
        channel.queue_declare(
            queue=rabbit["input_queue"], durable=True,
            arguments={"x-dead-letter-exchange": dlx, "x-dead-letter-routing-key": dlq},
        )
        channel.queue_declare(queue=rabbit["output_queue"], durable=True)
        channel.basic_qos(prefetch_count=int(rabbit.get("prefetch_count", 1)))
        channel.confirm_delivery()
        return connection, channel

    def publish_result(channel: Any, path: Path) -> None:
        envelope = load_json(path)
        result = envelope["result"]
        confirmed = channel.basic_publish(
            exchange=rabbit.get("output_exchange", ""),
            routing_key=rabbit.get("output_routing_key", rabbit["output_queue"]),
            body=json.dumps(result, allow_nan=False).encode("utf-8"), mandatory=True,
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json", correlation_id=envelope.get("correlation_id")),
        )
        if confirmed is False:
            raise RuntimeError("RabbitMQ negatively acknowledged output publish")
        state.mark_processed(envelope.get("source_message_id", result.get("message_id")))
        path.unlink(missing_ok=True)

    while True:
        connection = None
        try:
            state.status, state.last_error = "connecting", None
            connection, channel = connect()
            for path in state.pending():
                publish_result(channel, path)
                state.increment("outbox_replayed")
            reconnect_delay = float(rabbit.get("reconnect_initial_seconds", 1))
            state.status = "connected"

            def retry_or_dead_letter(ch: Any, method: Any, properties: Any, body: bytes, error: Exception) -> None:
                headers = dict(properties.headers or {})
                retry_count = int(headers.get("x-worker-retry-count", 0))
                if retry_count >= maximum_retries:
                    try:
                        source_message_id = json.loads(body.decode("utf-8")).get("message_id")
                    except Exception:
                        source_message_id = None
                    terminal = state.add_outbox(error_result(body, error), properties.correlation_id, source_message_id)
                    publish_result(ch, terminal)
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    state.increment("dead_lettered")
                    return
                headers["x-worker-retry-count"] = retry_count + 1
                confirmed = ch.basic_publish(
                    exchange="", routing_key=rabbit["input_queue"], body=body, mandatory=True,
                    properties=pika.BasicProperties(delivery_mode=2, content_type="application/json", correlation_id=properties.correlation_id, headers=headers),
                )
                if confirmed is False:
                    raise RuntimeError("RabbitMQ negatively acknowledged retry publish")
                ch.basic_ack(delivery_tag=method.delivery_tag)

            def callback(ch: Any, method: Any, properties: Any, body: bytes) -> None:
                state.increment("received")
                try:
                    message = json.loads(body.decode("utf-8"))
                    message_id = message.get("message_id")
                    if state.is_processed(message_id):
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                    result = analyzer.analyze(message)
                    pending = state.add_outbox(result, properties.correlation_id, message_id)
                    publish_result(ch, pending)  # Publisher confirm arrives before input ack.
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    state.increment("successful")
                    LOG.info("published %s for %s", result["result"], result.get("message_id"))
                except Exception as error:
                    state.increment("errors")
                    LOG.exception("alarm processing failed")
                    retry_or_dead_letter(ch, method, properties, body, error)

            LOG.info("waiting for RabbitMQ queue %s", rabbit["input_queue"])
            channel.basic_consume(queue=rabbit["input_queue"], on_message_callback=callback, auto_ack=False)
            channel.start_consuming()
        except KeyboardInterrupt:
            LOG.info("worker stopped")
            return
        except Exception as error:
            state.status, state.last_error = "disconnected", f"{type(error).__name__}: {error}"
            state.increment("reconnects")
            LOG.exception("RabbitMQ connection failed; reconnecting in %.1f seconds", reconnect_delay)
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, reconnect_max)
        finally:
            if connection is not None and connection.is_open:
                connection.close()


def run_local_test(config_path: Path, message_path: Path, image_path: Path, output_path: Path | None) -> None:
    """Run the same analysis logic locally, without RabbitMQ or SFTP."""
    config_path = config_path.resolve()
    config = load_json(config_path)
    message = load_json(message_path.resolve())
    if isinstance(message, list):
        if len(message) != 1 or not isinstance(message[0], dict):
            raise ValueError("--test-message JSON must be one object or an array containing exactly one object")
        message = message[0]
    if not isinstance(message, dict):
        raise ValueError("--test-message JSON must be an object")
    analyzer = AlarmAnalyzer(config, config_path)
    analyzer.validate_cameras()
    result = analyzer.analyze(message, local_image=image_path)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        LOG.info("Saved local test result: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, required=True, help="Worker configuration JSON")
    parser.add_argument("--test-message", type=Path, help="Local alarm JSON; bypasses RabbitMQ")
    parser.add_argument("--local-image", type=Path, help="Local image for --test-message; bypasses SFTP")
    parser.add_argument("--test-output", type=Path, help="Optional output JSON path for --test-message")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    if args.test_message is None:
        if args.local_image is not None or args.test_output is not None:
            parser.error("--local-image and --test-output require --test-message")
        run_worker(args.worker_config)
    else:
        if args.local_image is None:
            parser.error("--local-image is required with --test-message")
        run_local_test(args.worker_config, args.test_message, args.local_image, args.test_output)


if __name__ == "__main__": main()
