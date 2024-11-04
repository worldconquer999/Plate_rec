from pydantic import BaseModel


class Response(BaseModel):
    status: int = 200
    message: str = "success"
    result: object

    class Config:
        arbitrary_types_allowed = True


class InferRequestImgType:
    PICKLE = "PICKLE"
    BASE64 = "BASE64"
    BYTE = "BYTE"


class InferRequest(BaseModel):
    cam_id: str
    data: bytes
    img_type: str = InferRequestImgType.PICKLE
