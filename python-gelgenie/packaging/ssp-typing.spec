# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the standalone ssp-typing app (one-dir bundle).
#
# Build from python-gelgenie/:
#   .venv/bin/pyinstaller packaging/ssp-typing.spec --noconfirm \
#       --distpath dist --workpath build/pyi
#
# Result: dist/ssp-typing/ssp-typing  (+ bundled webapp static UI).
# The bundle is torch-free and gelgenie-free by design: only the ssp package
# and its runtime deps (numpy, scikit-image, imageio, onnxruntime, rich-click,
# openpyxl, fastapi, uvicorn) are collected.

import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(os.path.abspath(SPECPATH))  # python-gelgenie/

# uvicorn resolves its loops/protocols/lifespan implementations dynamically
hiddenimports = collect_submodules("uvicorn")

a = Analysis(
    [os.path.join(ROOT, "ssp", "cli.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "ssp", "webapp", "static"), "ssp/webapp/static"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio", "gelgenie", "matplotlib",
        "pandas", "wandb", "albumentations", "monai", "segmentation_models_pytorch",
        "cv2", "PyQt5", "PySide2", "tkinter", "notebook", "IPython",
        "pytest", "sympy", "numba",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="ssp-typing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ssp-typing",
)
