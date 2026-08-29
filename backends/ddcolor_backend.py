"""
Colorization-Backend auf Basis des offiziellen DDColor-Codes
(https://github.com/piddnad/DDColor, ICCV 2023).

Implementiert die Lab-Pipeline selbst (statt die fertige
ColorizationPipeline aus dem Repo zu nutzen), damit wir Dinge
nachbessern koennen, die bei roher Frame-fuer-Frame-Colorization von
Videos typischerweise stoeren:

  1. Saettigung: DDColor faerbt von Haus aus recht kraeftig/grell ein.
     `saturation` (0-1) daempft die Farbintensitaet.

  2. Globales Flackern (Farbton schwankt insgesamt von Frame zu Frame):
     `stats_smoothing` zieht den mittleren Farbton sanft in Richtung
     eines gleitenden Durchschnitts. Bewusst NUR der Mittelwert, keine
     Streuungs-Angleichung mehr (eine fruehere Version hat versucht,
     auch die Streuung anzugleichen - das hat bei Frames mit wenig
     vorhergesagter Farbe [kleine Streuung] das Rauschen hochskaliert
     und dadurch Aussetzer noch verstaerkt statt sie zu daempfen).

  3. "Aussetzer" (Modell faerbt einen Frame kaum ein -> wirkt fast S/W):
     Die durchschnittliche Farbintensitaet jedes Frames wird mit einem
     laufenden Referenzwert verglichen. Faellt sie deutlich ab, wird
     automatisch staerker auf den (bewegungsausgeglichenen)
     Vorgaenger-Frame zurueckgegriffen, statt den Farbverlust zu
     uebernehmen. Das verhindert das Hin-und-Herspringen zwischen
     "farbig" und "fast S/W".

  4. Lokales Flackern: die a/b-Kanaele werden zusaetzlich mit dem per
     Optical Flow (Farneback) bewegungsausgeglichenen Vorgaenger-Frame
     gemischt (`temporal_smoothing`), um kleinraeumige Unterschiede zu
     daempfen, ohne bei Bewegung zu verschmieren.

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
        saturation: float = 0.45,
        temporal_smoothing: float = 0.85,
        stats_smoothing: float = 0.9,
        mean_shift_strength: float = 0.0,  # standardmaessig AUS (siehe Hinweis unten)
        dropout_ratio_threshold: float = 0.5,
        dropout_smoothing: float = 0.9,
        spatial_blur_ksize: int = 0,  # 0 = automatisch aus Aufloesung berechnen
        anchor_smoothing: float = 0.97,  # Glaettung fuer voellig ruhige Bereiche
        motion_scale: float = 8.0,  # Pixel-Bewegung, ab der volle "Normal"-Glaettung gilt
    ):
        self.device = device
        self.model_name = model_name
        self.input_size = input_size
        self.saturation = saturation
        self.temporal_smoothing = temporal_smoothing
        self.stats_smoothing = stats_smoothing
        self.mean_shift_strength = mean_shift_strength
        # Schwelle: faellt die Farbintensitaet eines Frames unter diesen
        # Anteil des laufenden Durchschnitts, gilt er als "Aussetzer".
        self.dropout_ratio_threshold = dropout_ratio_threshold
        # Wie stark bei einem erkannten Aussetzer auf den Vorgaenger-Frame
        # zurueckgegriffen wird (0-1, hoeher = staerker).
        self.dropout_smoothing = dropout_smoothing
        self.spatial_blur_ksize = spatial_blur_ksize
        self.anchor_smoothing = anchor_smoothing
        self.motion_scale = motion_scale

        self._model = None
        self._torch_device = None
        self._prev_ab = None  # fuer zeitliche Glaettung zwischen Frames
        self._prev_gray = None  # fuer Optical-Flow-Berechnung
        self._running_mean = None  # fuer globale Farbton-Stabilisierung
        self._running_mag = None  # fuer Aussetzer-Erkennung

    def load(self, log=print):
        log(
            f"[Backend-Version-Check] saturation={self.saturation}  "
            f"temporal_smoothing={self.temporal_smoothing}  "
            f"stats_smoothing={self.stats_smoothing}  "
            f"spatial_blur_ksize={self.spatial_blur_ksize}  "
            f"shadow/highlight-Taper aktiv=ja  mean_shift_strength={self.mean_shift_strength}  "
            f"anchor_smoothing={self.anchor_smoothing}  motion_scale={self.motion_scale}  "
            f"(Datei-Stand: 2026-08-29c-anker-v2)"
        )
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
        Werten aus dem vorherigen Video startet."""
        self._prev_ab = None
        self._prev_gray = None
        self._running_mean = None
        self._running_mag = None

    def _shift_color_mean(self, ab: np.ndarray) -> np.ndarray:
        """Zieht den mittleren Farbton sanft Richtung gleitendem
        Durchschnitt (nur Verschiebung, keine Streuungs-Skalierung -
        das bleibt numerisch stabil, auch bei fast farblosen Frames)."""
        flat = ab.reshape(-1, 2)
        cur_mean = flat.mean(axis=0)

        if self._running_mean is None:
            self._running_mean = cur_mean
            return ab

        self._running_mean = self.stats_smoothing * self._running_mean + (1 - self.stats_smoothing) * cur_mean
        shift = (self._running_mean - cur_mean) * self.mean_shift_strength
        return ab + shift

    def _detect_dropout_ratio(self, ab: np.ndarray) -> float:
        """Vergleicht die Farbintensitaet des aktuellen Frames mit dem
        laufenden Durchschnitt. Gibt einen Faktor 0-1 zurueck: 1 = normal,
        0 = kompletter Aussetzer (praktisch keine Farbe vorhergesagt)."""
        cur_mag = float(np.sqrt((ab ** 2).sum(axis=-1)).mean())

        if self._running_mag is None:
            self._running_mag = cur_mag
            return 1.0

        ratio = cur_mag / (self._running_mag + 1e-6)
        # Referenzwert langsam nachfuehren (aber nicht von Aussetzern
        # herunterziehen lassen, damit die Erkennung stabil bleibt)
        if ratio > 0.7:
            self._running_mag = self.dropout_smoothing * self._running_mag + (1 - self.dropout_smoothing) * cur_mag

        return min(ratio, 1.0)

    def _warp_prev_ab_with_motion(self, gray: np.ndarray):
        """Verschiebt die a/b-Kanaele des vorherigen Frames per Optical
        Flow, damit sie zur Bewegung im aktuellen Frame passen, und gibt
        zusaetzlich eine Bewegungsstaerke pro Pixel zurueck (0 = komplett
        ruhig/statisch, hoeher = staerkere Bewegung). Gibt (None, None)
        zurueck, wenn kein Vorgaenger vorhanden ist oder die Berechnung
        fehlschlaegt."""
        if self._prev_gray is None or self._prev_ab is None:
            return None, None
        if self._prev_gray.shape != gray.shape:
            return None, None

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
            motion_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            return warped, motion_mag
        except cv2.error:
            return None, None

    def colorize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()

        import torch
        import torch.nn.functional as F

        height, width = frame_bgr.shape[:2]
        img = (frame_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Blur-Kernelgroesse einmalig aus der Aufloesung ableiten (falls
        # nicht explizit gesetzt) - ca. 1% der kleineren Bildkante, ungerade
        if self.spatial_blur_ksize == 0:
            k = max(3, round(min(height, width) * 0.01))
            if k % 2 == 0:
                k += 1
            self.spatial_blur_ksize = k

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

        # 1a) Farbe in sehr dunklen/hellen Bereichen zurueckdraengen -
        #     aber NIE komplett auf 0 (das wuerde bei natuerlichem
        #     Helligkeitsflackern alter Filmabtastungen selbst zu Farb-
        #     Flackern fuehren, weil Pixel dann staendig die Schwelle
        #     ueber-/unterschreiten wuerden). Weicher Uebergang
        #     (smoothstep) statt harter linearer Kante, mit Mindestwert.
        l_channel = orig_l[:, :, 0]  # bei float32-Eingabe liefert cv2 bereits 0..100
        shadow_th, highlight_th = 14.0, 92.0
        taper_floor = 0.4

        def _smoothstep(t):
            t = np.clip(t, 0, 1)
            return t * t * (3 - 2 * t)

        taper = np.ones_like(l_channel)
        shadow_t = _smoothstep(l_channel / shadow_th)
        taper = np.where(l_channel < shadow_th, taper_floor + (1 - taper_floor) * shadow_t, taper)
        highlight_t = _smoothstep((100.0 - l_channel) / (100.0 - highlight_th))
        taper = np.where(l_channel > highlight_th, taper_floor + (1 - taper_floor) * highlight_t, taper)
        output_ab_resized = output_ab_resized * taper[:, :, None]

        # 1b) Raeumlich leicht weichzeichnen (nur Farbe, nicht Helligkeit!)
        #     Das entfernt feines, pixelnahes Farbrauschen, das von Frame zu
        #     Frame unabhaengig neu entsteht und als Flackern auffaellt -
        #     das Auge ist fuer Farbdetails ohnehin viel unempfindlicher
        #     als fuer Helligkeit (genau wie bei der Chroma-Subsampling-
        #     Praxis in Videocodecs).
        if self.spatial_blur_ksize > 1:
            output_ab_resized = cv2.GaussianBlur(
                output_ab_resized, (self.spatial_blur_ksize, self.spatial_blur_ksize), 0
            )

        # 2) Globalen Farbton sanft angleichen
        if self.stats_smoothing > 0 and self.mean_shift_strength > 0:
            output_ab_resized = self._shift_color_mean(output_ab_resized)

        # 3) Aussetzer erkennen (Frame hat viel weniger Farbe als ueblich)
        dropout_ratio = self._detect_dropout_ratio(output_ab_resized)

        # 4) Lokale Glaettung, ortsabhaengig nach Bewegungsstaerke:
        #    Ruhige/statische Bildbereiche (z.B. ein stillstehender Hut)
        #    bekommen eine sehr starke Farb-Fixierung (nahe eingefroren -
        #    "Anker"), waehrend sich bewegende Bereiche normal auf neue
        #    Vorhersagen reagieren koennen. Das begegnet direkt dem
        #    beobachteten Muster: gerade unbewegte Flaechen zeigten trotz
        #    hoher globaler Glaettung weiterhin langsames Farbdriften.
        base_smoothing = self.temporal_smoothing
        if dropout_ratio < self.dropout_ratio_threshold:
            extra = (self.dropout_ratio_threshold - dropout_ratio) / self.dropout_ratio_threshold
            max_extra_smoothing = 0.9
            base_smoothing = self.temporal_smoothing + (max_extra_smoothing - self.temporal_smoothing) * extra
            base_smoothing = min(base_smoothing, max_extra_smoothing)

        warped_prev_ab, motion_mag = self._warp_prev_ab_with_motion(gray)
        if warped_prev_ab is not None and warped_prev_ab.shape == output_ab_resized.shape:
            # Bewegungsstaerke 0..1 normalisieren (motion_scale Pixel
            # Verschiebung = volle Bewegung, darueber hinaus wird gekappt)
            motion_norm = np.clip(motion_mag / self.motion_scale, 0, 1)
            # Bei Bewegung 0 -> anchor_smoothing (sehr stark), bei viel
            # Bewegung -> base_smoothing (normal)
            local_smoothing = self.anchor_smoothing - (self.anchor_smoothing - base_smoothing) * motion_norm
            local_smoothing = local_smoothing[:, :, None]

            output_ab_resized = (
                local_smoothing * warped_prev_ab + (1 - local_smoothing) * output_ab_resized
            )

        self._prev_ab = output_ab_resized
        self._prev_gray = gray

        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
        return output_img
