from __future__ import annotations

import dataclasses
import io
import uuid
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Protocol
from urllib import request as urllib_request

import numpy as np


@dataclasses.dataclass(frozen=True)
class PerceptionResult:
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray
    used_fallback: bool = False
    error: str | None = None


class PerceptionClient(Protocol):
    def process(
        self,
        *,
        select_objects: str,
        exclude_objects: str,
        is_base_init: bool,
        is_wrist_init: bool,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> PerceptionResult:
        ...


def _validate_rgb(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {arr.shape}.")
    return arr


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _read_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _restore_dimmed_context(
    *,
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    dim_factor: float = 0.35,
    black_threshold: int = 6,
) -> np.ndarray:
    original_rgb = _validate_rgb("original_rgb", original_rgb)
    overlay_rgb = _validate_rgb("overlay_rgb", overlay_rgb)
    if original_rgb.shape != overlay_rgb.shape:
        return overlay_rgb
    black_background = np.all(overlay_rgb <= int(black_threshold), axis=-1)
    dimmed = np.clip(original_rgb.astype(np.float32) * float(dim_factor), 0, 255).astype(np.uint8)
    return np.where(black_background[..., None], dimmed, overlay_rgb)


def _multipart_form_data(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
) -> tuple[bytes, str]:
    boundary = f"piper-obeyed-vla-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    for name, (filename, payload) in files.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
            + payload
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _parse_multipart_npy(body: bytes, content_type: str) -> dict[str, np.ndarray]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=email_policy).parsebytes(header + body)
    arrays: dict[str, np.ndarray] = {}
    if not message.is_multipart():
        raise ValueError("Perception response is not multipart.")
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue
        arrays[filename] = _read_npy(part.get_payload(decode=True))
    return arrays


class PassthroughPerceptionClient:
    def process(
        self,
        *,
        select_objects: str,
        exclude_objects: str,
        is_base_init: bool,
        is_wrist_init: bool,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> PerceptionResult:
        del select_objects, exclude_objects, is_base_init, is_wrist_init
        return PerceptionResult(
            base_rgb=_validate_rgb("base_rgb", base_rgb),
            wrist_rgb=_validate_rgb("wrist_rgb", wrist_rgb),
            used_fallback=True,
        )


class HttpPerceptionClient:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_s: float = 10.0,
        opener: Callable | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self._opener = opener or urllib_request.urlopen

    def process(
        self,
        *,
        select_objects: str,
        exclude_objects: str,
        is_base_init: bool,
        is_wrist_init: bool,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> PerceptionResult:
        base_rgb = _validate_rgb("base_rgb", base_rgb)
        wrist_rgb = _validate_rgb("wrist_rgb", wrist_rgb)
        body, content_type = _multipart_form_data(
            fields={
                "select_objects": select_objects,
                "exclude_objects": exclude_objects,
                "is_base_init": str(bool(is_base_init)).lower(),
                "is_wrist_init": str(bool(is_wrist_init)).lower(),
            },
            files={
                "base_npy": ("base.npy", _npy_bytes(base_rgb)),
                "wrist_npy": ("wrist.npy", _npy_bytes(wrist_rgb)),
            },
        )
        req = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with self._opener(req, timeout=self.timeout_s) as resp:
            resp_body = resp.read()
            resp_content_type = resp.headers.get("Content-Type", "")
        arrays = _parse_multipart_npy(resp_body, resp_content_type)
        try:
            base_out = arrays["overlay_base.npy"]
            wrist_out = arrays["overlay_wrist.npy"]
        except KeyError as exc:
            raise ValueError(f"Perception response missing {exc.args[0]}.") from exc
        return PerceptionResult(
            base_rgb=_restore_dimmed_context(original_rgb=base_rgb, overlay_rgb=base_out),
            wrist_rgb=_restore_dimmed_context(original_rgb=wrist_rgb, overlay_rgb=wrist_out),
            used_fallback=False,
        )


class FallbackPerceptionClient:
    def __init__(self, *, primary: PerceptionClient, fallback: PerceptionClient) -> None:
        self.primary = primary
        self.fallback = fallback

    def process(
        self,
        *,
        select_objects: str,
        exclude_objects: str,
        is_base_init: bool,
        is_wrist_init: bool,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> PerceptionResult:
        try:
            return self.primary.process(
                select_objects=select_objects,
                exclude_objects=exclude_objects,
                is_base_init=is_base_init,
                is_wrist_init=is_wrist_init,
                base_rgb=base_rgb,
                wrist_rgb=wrist_rgb,
            )
        except Exception as exc:
            result = self.fallback.process(
                select_objects=select_objects,
                exclude_objects=exclude_objects,
                is_base_init=is_base_init,
                is_wrist_init=is_wrist_init,
                base_rgb=base_rgb,
                wrist_rgb=wrist_rgb,
            )
            return dataclasses.replace(result, used_fallback=True, error=str(exc))


def make_perception_client(
    *,
    mode: str,
    endpoint: str,
    timeout_s: float = 10.0,
    fallback_to_passthrough: bool = True,
) -> PerceptionClient:
    if mode == "passthrough":
        return PassthroughPerceptionClient()
    if mode != "http":
        raise ValueError(f"Unsupported perception mode {mode!r}; use 'http' or 'passthrough'.")
    http_client = HttpPerceptionClient(endpoint=endpoint, timeout_s=timeout_s)
    if fallback_to_passthrough:
        return FallbackPerceptionClient(
            primary=http_client,
            fallback=PassthroughPerceptionClient(),
        )
    return http_client
