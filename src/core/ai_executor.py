from fastapi import HTTPException
from ai_engine.plate_recognition import PlateRecognition
from core.image_utils import get_numpy_frame, save_image
from core.model import Response
from core.prj_config import prj_config
import time


class AIExecutor:
    MIN_INTERVAL = 1.0 / prj_config.MAX_REQUEST_PER_SEC
    MAX_INTERVAL = 1.0 / prj_config.MIN_REQUEST_PER_SEC
    ACCELERATE_MODE_DURATION = prj_config.ACCELERATE_MODE_DURATION

    def __init__(self) -> None:
        self.ai_executor = PlateRecognition()
        self.cam_requests = {}
        self.expired_infer_from_file = 0

    @staticmethod
    def is_valid_time_interval(previous_request_timestamp, current_timestamp, acceleration_mode=False) -> bool:
        interval = AIExecutor.MIN_INTERVAL if acceleration_mode else AIExecutor.MAX_INTERVAL
        return False if previous_request_timestamp + interval > current_timestamp else True

    async def infer_from_binary(self, cam_id, binary_frame, img_type="PICKLE"):
        current_timestamp = time.time()
        if cam_id not in self.cam_requests:
            self.cam_requests[cam_id] = {
                "prev": current_timestamp,
                "acceleration_mode_expired_at": 0,
                "acceleration_mode": False
            }
        if not AIExecutor.is_valid_time_interval(self.cam_requests[cam_id]['prev'], current_timestamp,
                                                 self.cam_requests[cam_id]['acceleration_mode']):
            print(current_timestamp)
            raise HTTPException(
                detail="Time interval invalid, slowdown your request rate",
                status_code=400
            )

        if self.cam_requests[cam_id]["acceleration_mode_expired_at"] < current_timestamp:
            self.cam_requests[cam_id]["acceleration_mode"] = False

        self.cam_requests[cam_id]["prev"] = current_timestamp
        try:
            numpy_frame = get_numpy_frame(binary_frame, img_type)
        except Exception as e:
            raise HTTPException(
                detail=str(e),
                status_code=400
            )
        
        try:
            save_image(numpy_frame)
        except Exception as e:
            print("Save image error ", e)

        infer_result = self.ai_executor.get_result(numpy_frame)
        if infer_result is None:
            return Response(result={"text_plate": "", "bbox": [], "score": 0})

        self.cam_requests[cam_id]["acceleration_mode"] = True
        self.cam_requests[cam_id]["acceleration_mode_expired_at"] = self.cam_requests[cam_id]["prev"] + AIExecutor.ACCELERATE_MODE_DURATION

        return Response(result={"text_plate": infer_result.get("recognizer_result"), "bbox": infer_result.get("bbox"),
                                "score": infer_result.get("score")})

    async def infer_from_file_upload(self, byte_frame):
        current_timestamp = time.time()
        if current_timestamp - self.expired_infer_from_file < 2:
            raise HTTPException(
                detail="Time interval invalid, slowdown your request rate",
                status_code=400
            )
        self.expired_infer_from_file = current_timestamp
        try:
            numpy_frame = get_numpy_frame(byte_frame, img_type="BYTE")
        except Exception as e:
            raise HTTPException(
                detail=str(e),
                status_code=400
            )

        infer_result = self.ai_executor.get_result(numpy_frame)
        if infer_result is None:
            return Response(result={"text_plate": "", "bbox": [], "score": 0})

        return Response(result={"text_plate": infer_result.get("recognizer_result"), "bbox": infer_result.get("bbox"),
                                "score": infer_result.get("score")})
