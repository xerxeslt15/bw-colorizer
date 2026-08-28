# BW Colorizer by DSK

Ein erstes lauffähiges Grundgerüst für eine GUI, die Schwarz-Weiß-Videos
automatisch per KI einfärbt. Nutzt den offiziellen
[DDColor](https://github.com/piddnad/DDColor)-Code (ICCV 2023, Autor: piddnad)
direkt (kein ModelScope, keine Kompilierung nötig), läuft lokal auf deiner GPU.

## Voraussetzungen

1. **Python 3.10, 3.11 oder 3.12** (64-bit, NICHT 3.13+) — https://www.python.org/downloads/
2. **FFmpeg** muss installiert und im PATH verfügbar sein
   (`ffmpeg -version` sollte im CMD/PowerShell funktionieren).
   Download: https://www.gyan.dev/ffmpeg/builds/ (Variante "essentials", zip
   entpacken, `bin`-Ordner zur PATH-Umgebungsvariable hinzufügen)
3. **NVIDIA-Treiber aktuell** (für RTX 5070 Ti / RTX 5080)
4. Windows 10/11 mit `curl` und `tar` (beide seit Windows 10 1803 vorinstalliert) —
   werden benutzt, um den DDColor-Code herunterzuladen (kein `git` nötig)

## ⚠️ Wichtiger Hinweis zu RTX 5070 Ti / RTX 5080 (Blackwell, sm_120)

Deine Karten laufen auf der neuen Blackwell-Architektur (`sm_120`). Dafür
wird **CUDA 12.8** benötigt - `setup.bat` installiert deshalb PyTorch über
den `cu128`-Index. Falls der stabile Build nicht klappt, versucht das
Skript automatisch den Nightly-Build mit `cu128`. Am Ende von `setup.bat`
steht ein GPU-Check: dort sollte `CUDA verfuegbar: True` und dein Karten-
name erscheinen. Steht dort `False`, meld dich - dann schauen wir uns die
genaue Fehlermeldung an.

## Python-Version

Verwende **Python 3.10, 3.11 oder 3.12** (64-bit). Mit Python 3.13+ gibt es
aktuell noch Probleme, weil einige Pakete dafür keine vorgefertigten Wheels
haben und dann lokal kompiliert werden müssten (braucht Visual C++ Build
Tools, die üblicherweise fehlen). Falls `python --version` bei dir 3.13
oder höher zeigt: installiere zusätzlich Python 3.11 von
https://www.python.org/downloads/release/python-3119/ und lege die
virtuelle Umgebung damit an:
```
py -3.11 -m venv venv
```
(Diese Zeile ersetzt im Bedarfsfall den `python -m venv venv`-Schritt in
`setup.bat`.)

## Installation

```
setup.bat
```

Das Skript legt eine virtuelle Umgebung an, installiert PyTorch (CUDA),
lädt den DDColor-Code (reines Python, ca. wenige MB) nach `backends/DDColor`
herunter und installiert die restlichen Abhängigkeiten aus `requirements.txt`.

## Start

```
venv\Scripts\python.exe main.py
```

Beim allerersten Einfärben eines Videos lädt das Tool automatisch die
DDColor-Modellgewichte von Hugging Face herunter (~100-400 MB, einmalig,
danach lokal gecacht unter `~/.cache/huggingface`).

## Funktionsumfang (aktueller Stand)

- Mehrere Videos gleichzeitig auswählen (Batch)
- Automatische GPU-Erkennung (Name + VRAM-Anzeige oben in der GUI)
- Fortschrittsanzeige pro Frame und pro Datei
- Abbrechen-Funktion
- Original-Audiospur wird übernommen
- Ausgabe als MP4 (H.264), Ausgabeordner frei wählbar (Standard: gleicher
  Ordner wie Quelldatei, Suffix `_farbe`)

## Bekannte Baustellen / nächste Schritte

- **Geschwindigkeit**: Aktuell wird jedes Frame einzeln auf die Platte
  geschrieben/gelesen. Für lange Filme kann man das direkt über Pipes lösen
  (schneller, weniger Festplattenzugriffe) — sag Bescheid, wenn du das als
  nächstes willst.
- **Downloadgröße bei Verteilung als exe**: PyTorch + Modell sind groß.
  Für eine reine "Doppelklick-exe" (wie beim DSK Converter) müsste das
  Modell separat nachgeladen werden statt es einzubetten.
- **Dark-Mode-Signatur/Branding**: aktuell simpel gehalten ("by DSK" oben),
  kann analog zum DSK Converter weiter gestylt werden.
- **Modellvarianten**: DDColor bietet neben `ddcolor_modelscope` (Standard)
  auch `ddcolor_paper`, `ddcolor_artistic` und das kleinere/schnellere
  `ddcolor_paper_tiny` — umstellbar über `model_name` in
  `backends/ddcolor_backend.py`.

## Projektstruktur

```
bw_colorizer/
  main.py                    GUI (Einstiegspunkt)
  colorize_engine.py         Video-Pipeline (Frames extrahieren/einfärben/zusammensetzen)
  gpu_utils.py                GPU-Erkennung
  backends/
    ddcolor_backend.py        Colorization-Modell-Wrapper
    DDColor/                  (wird von setup.bat heruntergeladen, nicht im ZIP enthalten)
  requirements.txt
  setup.bat
```

## Als EXE bauen (über GitHub Actions, wie beim DSK Converter)

Lokal eine exe zu bauen ist bei diesem Projekt unpraktisch (PyTorch +
Modellgewichte sind mehrere GB groß, plus deine lokale Python-Version macht
oft Ärger). Deshalb baut - genau wie beim DSK Converter - GitHub Actions die
exe für dich in der Cloud:

1. Neues (am besten **öffentliches**) GitHub-Repository anlegen, z. B.
   `bw-colorizer`. Öffentlich deshalb, weil der Artefakt-Speicherplatz bei
   öffentlichen Repos unlimitiert ist - bei privaten Repos ist er begrenzt
   (500 MB im kostenlosen Plan) und PyTorch allein ist schon ca. 2-3 GB groß.
2. Den kompletten Inhalt dieses `bw_colorizer`-Ordners in das Repo pushen
   (der `venv`-Ordner und `backends/DDColor` müssen NICHT mit hochgeladen
   werden - die baut sich der Workflow selbst; am besten eine `.gitignore`
   mit den Zeilen `venv/` und `backends/DDColor/` anlegen).
3. Im Tab "Actions" auf GitHub sollte der Workflow "Build BW Colorizer EXE"
   automatisch nach dem Push starten (oder manuell über "Run workflow"
   anstoßen).
4. Nach ca. 15-25 Minuten (PyTorch-Download + Bundling brauchen ihre Zeit)
   steht unten im abgeschlossenen Workflow-Lauf ein Artefakt
   **BWColorizer-windows** zum Download bereit - das ist ein ZIP mit der
   fertigen `BWColorizer.exe` plus allen nötigen DLLs.

**Wichtig:** Die exe ist eine "onedir"-Build, d. h. du bekommst einen ganzen
Ordner (nicht nur eine einzelne Datei) - die exe braucht die Dateien
daneben, also den kompletten entpackten Ordner zusammenhalten (z. B. als
"BWColorizer" auf dem Desktop). FFmpeg muss weiterhin separat installiert
und im PATH sein (wird nicht mitgebündelt).

Die exe selbst enthält KEINE Modellgewichte - die lädt sie beim ersten
Einfärben automatisch von Hugging Face herunter, genau wie beim
Python-Start.

## Lizenzhinweis

Der DDColor-Code und die Modellgewichte stammen von piddnad
(https://github.com/piddnad/DDColor) und unterliegen dessen Lizenz. Bei
kommerzieller Nutzung dort die genauen Lizenzbedingungen prüfen.

