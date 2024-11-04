import pickle
from cryptography.fernet import Fernet

from src.core.prj_config import prj_config

SECRET_KEY = "c15d4dc89135c712a135c40d0dda56dab952e8304c75d9eef2bbcb9df877f50169b4c7dc2a69b6c66a1c78f165c674a91da3"
ALGORITHM = "HS256"


def load_content_decode(path):
    fernet = Fernet(prj_config.TOKEN)
    with open(path, "rb") as f:
        content = pickle.load(f)
    content = fernet.decrypt(content)
    return content
