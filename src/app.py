from typing import Union, Dict, TypeVar
import uvicorn
import time
from fastapi import FastAPI, Request, Header, File, UploadFile
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
# from src.core.model import InferRequest

from core.internal_auth import create_token_generator, add_internal_auth_middleware
from core.prj_config import prj_config, BASE_DIR, STATIC_DIR
from core.ai_executor import AIExecutor

import argparse

T = TypeVar("T")

ai_executor = AIExecutor()

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=STATIC_DIR + r"/static"), name="static")
# app.mount("/static", StaticFiles(directory="static/swagger"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@app.on_event("startup")
def startup():
    create_token_generator()
    time.sleep(2)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.post("/plate-recognition")
async def plate_recogniton(file: UploadFile = File(...)):
    byte_content = file.file.read()
    res = await ai_executor.infer_from_file_upload(byte_content)
    return res

@app.post("/{img_type}")
async def infer(img_type, cam_id, request: Request):
    payload: bytes = await request.body()
    start = time.time()
    res = await ai_executor.infer_from_binary(cam_id, payload, img_type)
    print(f"cam_id {cam_id}  {time.time() - start}")
    return res

def get_dict(object: T, allow_none=False):
    res = {}
    data = object if type(object) is dict else object.__dict__
    for key, value in data.items():
        if value == None and allow_none == False:
            continue
        if isinstance(value, BaseModel):
            res[key] = get_dict(value)
        elif type(value) is not dict:
            res[key] = value
        elif len(value.keys()) == 0:
            continue
        else:
            res[key] = get_dict(value, allow_none)
    return res

@app.get("/backup/env")
def get_env():
    print(prj_config)
    res = get_dict(prj_config)
    print(res)
    return res

@app.post("/backup/env")
def post_env(env: Dict):
    try:
        payload = []
        for key, value in env.items():
            payload.append(str(key) + "=" + str(value).replace("'", '"'))
        with open(BASE_DIR + r'/.env', 'w', encoding='utf8') as fh:
            fh.write("\n".join(payload))
            fh.flush()
        return {"status_code": 200, "msg": "Success", "data": None}
    except:
        return {"status_code": 400, "msg": "Failure", "data": None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(prj_config.AI_SERVICE_PORT))
