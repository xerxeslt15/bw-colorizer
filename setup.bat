@echo off
echo === BW Colorizer by DSK - Setup ===
echo.

python --version
echo Hinweis: Python 3.10, 3.11 oder 3.12 wird empfohlen (nicht 3.13+).
echo.

if not exist venv (
    echo Erstelle virtuelle Umgebung...
    python -m venv venv
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip

echo.
echo Installiere PyTorch mit CUDA 12.8 Support (Blackwell / RTX 5070 Ti / RTX 5080 - sm_120)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo Pruefe, ob die GPU von diesem PyTorch-Build erkannt wird...
python -c "import torch; ok = torch.cuda.is_available(); print('CUDA verfuegbar:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else '-')" 2>nul
if errorlevel 1 (
    echo.
    echo Der stabile cu128-Build hat nicht funktioniert. Versuche Nightly-Build mit cu128...
    pip uninstall -y torch torchvision
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
)

echo.
echo Installiere restliche Abhaengigkeiten...
pip install -r requirements.txt

echo.
if not exist backends\DDColor (
    echo Lade DDColor-Code herunter ^(einmalig, kein pip-Paket^)...
    curl -L -o ddcolor_src.zip https://github.com/piddnad/DDColor/archive/refs/heads/master.zip
    if not exist ddcolor_src.zip (
        echo FEHLER: Download fehlgeschlagen. Bitte Internetverbindung pruefen und setup.bat erneut starten.
        pause
        exit /b 1
    )
    tar -xf ddcolor_src.zip
    move /Y DDColor-master backends\DDColor >nul
    del ddcolor_src.zip
    echo DDColor-Code bereit unter backends\DDColor
) else (
    echo DDColor-Code bereits vorhanden ^(backends\DDColor^), ueberspringe Download.
)

echo.
echo === Fertig! ===
echo Kurzer GPU-Check:
venv\Scripts\python.exe -c "import torch; ok=torch.cuda.is_available(); print('CUDA verfuegbar:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'keine')"
echo.
echo Hinweis: Beim allerersten Einfaerben laedt das Tool zusaetzlich die Modell-
echo gewichte von Hugging Face herunter (ca. 100-400 MB, einmalig).
echo.
echo Starte das Tool mit: venv\Scripts\python.exe main.py
pause
