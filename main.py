"""
BW Colorizer by DSK
--------------------
GUI zum automatischen Einfärben von Schwarz-Weiß-Videos mittels KI (DDColor).

Start: python main.py
"""

from __future__ import annotations
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from backends.ddcolor_backend import DDColorBackend
from colorize_engine import CancelledError, colorize_video, make_output_path
from gpu_utils import detect_gpu

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv")


class BWColorizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BW Colorizer by DSK")
        self.geometry("780x600")
        self.minsize(680, 520)

        self.files: list[str] = []
        self.output_dir: str | None = None
        self.is_running = False
        self.cancel_requested = False
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()

        self.gpu_info = detect_gpu()
        self.backend = DDColorBackend(device=self.gpu_info["device"])

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 6))

        title = ctk.CTkLabel(
            header, text="BW Colorizer", font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(side="left")

        signature = ctk.CTkLabel(
            header, text="by DSK", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#4da3ff",
        )
        signature.pack(side="left", padx=(10, 0), pady=(6, 0))

        gpu_text = f"GPU: {self.gpu_info['name']}"
        if self.gpu_info["vram_gb"]:
            gpu_text += f"  ({self.gpu_info['vram_gb']} GB VRAM)"
        gpu_color = "#2ecc71" if self.gpu_info["available"] else "#e74c3c"
        self.gpu_label = ctk.CTkLabel(self, text=gpu_text, text_color=gpu_color)
        self.gpu_label.pack(anchor="w", padx=20)

        # --- Dateiauswahl ---
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="both", expand=False, padx=20, pady=10)

        btn_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkButton(btn_row, text="Videos hinzufügen...", command=self._add_files).pack(side="left")
        ctk.CTkButton(btn_row, text="Liste leeren", fg_color="#555", command=self._clear_files).pack(side="left", padx=8)

        self.file_listbox = ctk.CTkTextbox(file_frame, height=140)
        self.file_listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.file_listbox.configure(state="disabled")

        # --- Ausgabeordner ---
        out_frame = ctk.CTkFrame(self, fg_color="transparent")
        out_frame.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkButton(out_frame, text="Ausgabeordner wählen...", command=self._choose_output_dir).pack(side="left")
        self.out_label = ctk.CTkLabel(out_frame, text="Standard: gleicher Ordner wie Quelldatei (Suffix _farbe)")
        self.out_label.pack(side="left", padx=10)

        # --- Start/Stop ---
        action_row = ctk.CTkFrame(self, fg_color="transparent")
        action_row.pack(fill="x", padx=20, pady=(0, 6))

        self.start_btn = ctk.CTkButton(action_row, text="Einfärben starten", command=self._start, height=38,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.start_btn.pack(side="left")

        self.stop_btn = ctk.CTkButton(action_row, text="Abbrechen", command=self._stop, height=38,
                                       fg_color="#c0392b", hover_color="#922b21", state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        # --- Fortschritt ---
        self.overall_label = ctk.CTkLabel(self, text="Bereit.")
        self.overall_label.pack(anchor="w", padx=20)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=20, pady=(4, 10))

        # --- Log ---
        self.log_box = ctk.CTkTextbox(self, height=160)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.log_box.configure(state="disabled")

    # -------------------------------------------------------------- Events
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Videos auswählen",
            filetypes=[("Videos", " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)), ("Alle Dateien", "*.*")],
        )
        for p in paths:
            if p not in self.files:
                self.files.append(p)
        self._refresh_file_list()

    def _clear_files(self):
        self.files.clear()
        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", "end")
        for f in self.files:
            self.file_listbox.insert("end", f + "\n")
        self.file_listbox.configure(state="disabled")

    def _choose_output_dir(self):
        d = filedialog.askdirectory(title="Ausgabeordner wählen")
        if d:
            self.output_dir = d
            self.out_label.configure(text=f"Ausgabe: {d}")

    def _log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # --------------------------------------------------------- Ablaufsteuerung
    def _start(self):
        if not self.files:
            messagebox.showwarning("Keine Dateien", "Bitte zuerst mindestens ein Video hinzufügen.")
            return
        if self.is_running:
            return

        self.is_running = True
        self.cancel_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._run_batch, daemon=True)
        thread.start()

    def _stop(self):
        self.cancel_requested = True
        self._log("Abbruch angefordert - wird nach dem aktuellen Frame gestoppt...")

    def _run_batch(self):
        total_files = len(self.files)
        for idx, src in enumerate(self.files, start=1):
            if self.cancel_requested:
                break

            if self.output_dir:
                out_path = str(Path(self.output_dir) / (Path(src).stem + "_farbe.mp4"))
            else:
                out_path = make_output_path(src)

            self.msg_queue.put(("status", f"[{idx}/{total_files}] Starte: {Path(src).name}"))

            def progress_cb(done, total, msg, idx=idx, total_files=total_files):
                self.msg_queue.put(("progress", idx, total_files, done, total, msg))

            try:
                colorize_video(
                    src, out_path, self.backend,
                    progress_cb=progress_cb,
                    cancel_flag=lambda: self.cancel_requested,
                )
                self.msg_queue.put(("status", f"[{idx}/{total_files}] Fertig -> {out_path}"))
            except CancelledError:
                self.msg_queue.put(("status", "Abgebrochen."))
                break
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self.msg_queue.put(("status", f"FEHLER bei {Path(src).name}: {exc}"))
                self.msg_queue.put(("status", tb))

        self.msg_queue.put(("done",))

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]

                if kind == "status":
                    self._log(item[1])
                    self.overall_label.configure(text=item[1])
                elif kind == "progress":
                    _, idx, total_files, done, total, msg = item
                    frac = (done / total) if total else 0
                    self.overall_label.configure(text=f"Datei {idx}/{total_files}: {msg}")
                    self.progress_bar.set(frac)
                elif kind == "done":
                    self.is_running = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.overall_label.configure(text="Alle Videos verarbeitet." if not self.cancel_requested else "Abgebrochen.")
        except queue.Empty:
            pass

        self.after(150, self._poll_queue)


if __name__ == "__main__":
    app = BWColorizerApp()
    app.mainloop()
