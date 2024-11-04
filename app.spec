# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/home/hquanhh/data_plate/ai-engine-main/src/app.py'],
    pathex=['/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/cv2', '/home/hquanhh/data_plate/ai-engine-main'],
    binaries=[],
    datas=[('/home/hquanhh/data_plate/ai-engine-main/pretrained', 'pretrained/'), ('/home/hquanhh/data_plate/ai-engine-main/src/static', 'static/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/openvino', 'openvino/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/uvicorn', 'uvicorn/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/fastapi', 'fastapi/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/starlette', 'starlette/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/pydantic', 'pydantic/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/ecdsa', 'ecdsa/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/jose', 'jose/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/passlib', 'passlib/'), ('/home/hquanhh/anaconda3/envs/vtv/lib/python3.9/site-packages/dotenv', 'dotenv/'), ('/home/hquanhh/data_plate/ai-engine-main/.env', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
