import os
from os import getenv
from dotenv import load_dotenv
from pydantic import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../"))
STATIC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class PrjConfig(BaseSettings):
    SERVICE_NAME = getenv("SERVICE_NAME", "BACKEND")
    SECRET_KEY = getenv("SECRET_KEY", "123456")
    AI_SERVICE_PORT = getenv("AI_SERVICE_PORT", "8002")
    SECURITY_ALGORITHM = getenv("SECURITY_ALGORITHM", "HS256")
    MAX_REQUEST_PER_SEC = 3
    MIN_REQUEST_PER_SEC = 1
    ACCELERATE_MODE_DURATION = 2
    TOKEN = b"CA-s7ielaGkTr0vb_7Z8yBtWPAVTpZLcBTrj_Tt4cz4="
    SCORE_THRESHOLD = float(getenv("SCORE_THRESHOLD", 0.5))
    PADDING = int(getenv("PADDING", 10))
    SAVE_PATH = getenv("SAVE_PATH", os.path.join(BASE_DIR, "capture-img"))


prj_config = PrjConfig()
print(prj_config)
# print("--- AI Service Config: \n", json.dumps(prj_config.dict(), indent=4))
