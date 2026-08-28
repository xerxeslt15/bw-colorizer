"""
Colorization-Backend auf Basis des offiziellen DDColor-Codes
(https://github.com/piddnad/DDColor, ICCV 2023).

Implementiert die Lab-Pipeline selbst (statt die fertige
ColorizationPipeline aus dem Repo zu nutzen), damit wir zwei Dinge
nachbessern können, die bei roher Frame-für-Frame-Colorization von
Videos typischerweise stören:

  1. Saettigung: DDColor faerbt von Haus aus recht kraeftig/grell ein.
     `saturation` (0-1) daempft die Farbintensitaet.
  2. Flackern: Jedes Frame wird eigentlich unabhaengig eingefaerbt, was
     bei Videos zu leicht wechselnden Farbtoenen von Frame zu Frame
     fuehrt. Dagegen wirken zwei Mechanismen zusammen:
     a) `stats_smoothing`: der globale Farbton (Mittelwert/Streuung der
        a/b-Kanaele) wird per gleitendem Durchschnitt ueber die Zeit
        stabilisiert - das faengt den Hauptteil des Flackerns ab
        (frame-weise schwankende Gesamtfaerbung/Saettigung).
     b) `temporal_smoothing`: zusaetzlich werden die a/b-Kanaele mit dem
        (per Optical Flow bewegungsausgeglichenen) Vorgaenger-Frame
        gemischt - das daempft noch lokale, ortsgebundene Unterschiede.
     Beide 0-1, hoehere Werte = ruhiger, aber traeger bei echten
     Farbwechseln (z.B. Schnitt auf eine andere Szene).

Modellgewichte kommen automatisch von Hugging Face Hub beim ersten
Start (Cache: ~/.cache/huggingface).
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

_DDCOLOR_REPO = Path(__file__).parent / "DDColor"
if getattr(sys, "frozen", False):
    # In der gebauten exe (PyInstaller) liegt der Code stattdessen relativ
    # zum Bundle-Verzeichnis.
    _DDCOLOR_REPO = Path(sys._MEIPASS) / "backends" / "DDColor"


class DDColorBackend:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "ddcolor_modelscope",
        input_size: int = 512,
        saturation: float = 0.75,
        temporal_smoothing: float = 0.55,
        stats_smoothing: float = 0.9,
    ):
        self.device = device
        self.model_name = model_name
        self.input_size = input_size
        self.saturation = saturation
        self.temporal_smoothing = temporal_smoothing
        self.stats_smoothing = stats_smoothing

        self._model = None
        self._torch_device = None
        self._prev_ab = None  # fuer zeitliche Glaettung zwischen Frames
        self._prev_gray = None  # fuer Optical-Flow-Berechnung
        self._running_mean = None  # fuer globale Farbton-Stabilisierung
        self._running_std = None

    def load(self, log=print):
        if self._model is not None:
            return

        if not _DDCOLOR_REPO.exists():
            raise RuntimeError(
                f"DDColor-Code nicht gefunden unter {_DDCOLOR_REPO}. "
                "Bitte setup.bat erneut ausführen (lädt den Code automatisch herunter)."
            )

        repo_str = str(_DDCOLOR_REPO)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        log(f"Lade DDColor-Modell '{self.model_name}' (beim ersten Mal Download von Hugging Face)...")

        import torch
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        self._torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        model = DDColorHF.from_pretrained(f"piddnad/{self.model_name}")
        model = model.to(self._torch_device)
        model.eval()

        self._model = model
        log("Modell geladen.")

    def reset_temporal(self):
        """Vor jedem neuen Video aufrufen, damit die Glaettung nicht mit
        Farbwerten aus dem vorherigen Video startet."""
        self._prev_ab = None
        self._prev_gray = None
        self._running_mean = None
        self._running_std = None

    def _stabilize_color_stats(self, ab: np.ndarray) -> np.ndarray:
        """Gleicht Mittelwert/Streuung der a/b-Kanaele an einen ueber die
        Zeit gleitenden Durchschnitt an - daempft globales Flackern des
        Farbtons (z.B. mal insgesamt waermer, mal insgesamt blasser)."""
        eps = 1e-6
        flat = ab.reshape(-1, 2)
        cur_mean = flat.mean(axis=0)
        cur_std = flat.std(axis=0) + eps

        if self._running_mean is None:
            self._running_mean = cur_mean
            self._running_std = cur_std
            return ab

        self._running_mean = self.stats_smoothing * self._running_mean + (1 - self.stats_smoothing) * cur_mean
        self._running_std = self.stats_smoothing * self._running_std + (1 - self.stats_smoothing) * cur_std

        return (ab - cur_mean) / cur_std * self._running_std + self._running_mean

    def _warp_prev_ab(self, gray: np.ndarray) -> np.ndarray | None:
        """Verschiebt die a/b-Kanaele des vorherigen Frames per Optical
        Flow, damit sie zur Bewegung im aktuellen Frame passen. Gibt None
        zurueck, wenn kein Vorgaenger vorhanden ist oder die Berechnung
        fehlschlaegt (dann wird ohne Glaettung weitergemacht)."""
        if self._prev_gray is None or self._prev_ab is None:
            return None
        if self._prev_gray.shape != gray.shape:
            return None

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            h, w = gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(
                self._prev_ab, map_x, map_y,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            return warped
        except cv2.error:
            return None

    def colorize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()

        import torch
        import torch.nn.functional as F

        height, width = frame_bgr.shape[:2]
        img = (frame_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float()
            .unsqueeze(0)
            .to(self._torch_device)
        )

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            output_ab = self._model(tensor_gray_rgb).cpu()

        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width))[0]
            .float()
            .numpy()
            .transpose(1, 2, 0)
        )

        # 1) Saettigung daempfen
        output_ab_resized = output_ab_resized * self.saturation

        # 2) Globalen Farbton stabilisieren (Hauptursache fuer Flackern)
        if self.stats_smoothing > 0:
            output_ab_resized = self._stabilize_color_stats(output_ab_resized)

        # 3) Zusaetzlich lokale, bewegungsausgeglichene Glaettung
        if self.temporal_smoothing > 0:
            warped_prev_ab = self._warp_prev_ab(gray)
            if warped_prev_ab is not None and warped_prev_ab.shape == output_ab_resized.shape:
                output_ab_resized = (
                    self.temporal_smoothing * warped_prev_ab
                    + (1 - self.temporal_smoothing) * output_ab_resized
                )

        self._prev_ab = output_ab_resized
        self._prev_gray = gray

        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
        return output_img
