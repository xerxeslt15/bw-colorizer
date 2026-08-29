"""
Kernpipeline des BW Colorizers:

  Video --(ffmpeg)--> Einzelbilder --(KI-Backend)--> eingefärbte Bilder --(ffmpeg)--> Video + Original-Audio

Läuft in einem eigenen Thread (siehe main.py), meldet Fortschritt über
Callbacks, damit die GUI nicht einfriert und ein Stop jederzeit möglich ist.
"""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool


class CancelledError(Exception):
    pass


def probe_video(path: str) -> VideoInfo:
    """Liest Video-Metadaten per ffprobe aus."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)

    v_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    a_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    num, den = v_stream.get("r_frame_rate", "25/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 25.0

    duration = float(data["format"].get("duration", 0))
    frame_count = int(v_stream.get("nb_frames", 0)) or int(duration * fps)

    return VideoInfo(
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=a_stream is not None,
    )


def colorize_video(
    input_path: str,
    output_path: str,
    backend,
    progress_cb: Callable[[int, int, str], None] = lambda done, total, msg: None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    jpeg_quality: int = 2,
    log_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Färbt ein einzelnes Video vollständig ein.

    progress_cb(done, total, message) wird laufend aufgerufen (Fortschritt,
    ueberschreibt sich staendig).
    log_cb(message) wird für dauerhafte Log-Zeilen aufgerufen (z.B. Backend-
    Diagnose-Ausgaben) - faellt auf progress_cb zurueck, falls nicht gesetzt.
    cancel_flag() sollte True zurückgeben, sobald der Nutzer abbrechen will.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    if log_cb is None:
        log_cb = lambda m: progress_cb(0, 0, m)

    info = probe_video(input_path)
    progress_cb(0, info.frame_count, f"Analysiere Video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} Frames")

    with tempfile.TemporaryDirectory(prefix="bwcolor_") as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        colored_dir = Path(tmpdir) / "colored"
        frames_dir.mkdir()
        colored_dir.mkdir()

        # --- 1. Frames extrahieren ---
        progress_cb(0, info.frame_count, "Extrahiere Einzelbilder...")
        extract_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-qscale:v", str(jpeg_quality),
            str(frames_dir / "frame_%08d.jpg"),
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)

        # --- 2. Frames einfärben ---
        backend.load(log=log_cb)
        if hasattr(backend, "reset_temporal"):
            backend.reset_temporal()

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        total = len(frame_files) or info.frame_count

        prev_hist = None
        cut_count = 0

        for i, fpath in enumerate(frame_files, start=1):
            if cancel_flag and cancel_flag():
                raise CancelledError("Vom Nutzer abgebrochen")

            frame_bgr = cv2.imread(str(fpath))

            # Szenenschnitt erkennen: bei starkem Bildwechsel die zeitliche
            # Glaettung zuruecksetzen, damit keine Farbwerte der voran-
            # gegangenen (unzusammenhaengenden) Szene eingemischt werden.
            # Sonst "haengt" nach jedem Schnitt fuer ~15-20 Frames eine
            # falsche Uebergangsfarbe im Bild (Hauptursache fuer Flackern
            # ueber den ganzen Film verteilt).
            small = cv2.resize(frame_bgr, (64, 64))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if similarity < 0.85 and hasattr(backend, "reset_temporal"):
                    backend.reset_temporal()
                    cut_count += 1
            prev_hist = hist

            colored = backend.colorize_frame(frame_bgr)

            # Ausgabegröße an Originalauflösung angleichen, falls das Modell
            # intern skaliert
            if colored.shape[:2] != frame_bgr.shape[:2]:
                colored = cv2.resize(colored, (frame_bgr.shape[1], frame_bgr.shape[0]))

            out_path = colored_dir / fpath.name
            cv2.imwrite(str(out_path), colored, [cv2.IMWRITE_JPEG_QUALITY, 97])

            if i % 5 == 0 or i == total:
                progress_cb(i, total, f"Färbe Frame {i}/{total} ein...")

        log_cb(f"Szenenschnitte erkannt und Glaettung zurueckgesetzt: {cut_count}x")

        # --- 4. Video wieder zusammensetzen (Ton direkt aus dem Original) ---
        progress_cb(total, total, "Setze Video zusammen (ffmpeg)...")
        assemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(info.fps),
            "-i", str(colored_dir / "frame_%08d.jpg"),
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        ]
        if info.has_audio:
            assemble_cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        assemble_cmd += [output_path]

        result = subprocess.run(assemble_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg-Fehler beim Zusammensetzen: {result.stderr.decode(errors='ignore')[-800:]}")

    progress_cb(total, total, f"Fertig: {output_path}")


def make_output_path(input_path: str, suffix: str = "_farbe") -> str:
    p = Path(input_path)
    return str(p.with_name(p.stem + suffix + ".mp4"))
