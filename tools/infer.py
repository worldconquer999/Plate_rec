import cv2
import os
import glob
import json
import shutil

from src.ai_engine.plate_recognition import PlateRecognition
from src.ai_engine.model.ocr_decode import OCRDecoder


# PATH_IMG =  r"E:\DataVTV\ALL_IMAGE\plate_in"
# PATH_OUT =  r"E:\DataVTV\ALL_INFER_RESULT\plate_in"

# PATH_IMG =  r"E:\DataVTV\ALL_IMAGE\plate_out"
# PATH_OUT =  r"E:\DataVTV\ALL_INFER_RESULT\plate_out"

# PATH_IMG =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\congvom\2024-05-29"
# PATH_OUT =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\congvom\infer-2024-05-29"

# files = os.listdir(PATH_IMG)
# model = PlateRecognition()
# res = {}
# for file_name in files:
#     path_file = os.path.join(PATH_IMG, file_name)
#     img = cv2.imread(path_file)
#     result = model.get_result(img)
#     if result is not None:
#         bbox = result["bbox"]
#         recognizer_result = result["recognizer_result"]
#         res[file_name] = recognizer_result
#         img = cv2.putText(img, recognizer_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
#                    1, (255, 0, 0) , 2, cv2.LINE_AA) 
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=2)
#         cv2.imwrite(os.path.join(PATH_OUT, file_name), img)
#     else:
#         res[file_name] = ""
    
#     print(file_name, res[file_name])

# with open("result_window.json", "w") as f:
#     json.dump(res, f)

# PATH_IMG = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\result"
# PATH_OUT = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\result_ocr"
# model = OCRDecoder(None)
# files = os.listdir(PATH_IMG)
# for file_name in files:
#     path_file = os.path.join(PATH_IMG, file_name)
#     img = cv2.imread(path_file, 0)
#     print(img.shape)
#     out1 = model.predict(img)
#     out1 = model.decode_batch(out1)[0]
#     path_file = os.path.join(PATH_OUT, out1 + "----" + file_name)
#     cv2.imwrite(path_file, img)


# PATH_IMG = r"E:\Image_Picture_VTV_to_train\VTV_IMAGE_ERROR_TO_LABEL\Image"
# label = r"E:\Image_Picture_VTV_to_train\VTV_IMAGE_ERROR_TO_LABEL\Label\*.json"
PATH_IMG = r"E:\DataVTV\Image"
label = r"E:\DataVTV\Label\*.json"
labels = glob.glob(label)

model = PlateRecognition()

def get_json_label(path: str) -> dict:
    json_data = json.load(open(path,"r"))
    data = json_data["_via_img_metadata"]
    data = list(data.values())[0]

    return data

def get_label(regions: list) -> str:
    len_regions = len(regions)
    if len_regions == 1:
        return regions[0]["region_attributes"]["label"]
    elif len_regions == 3:
        str_1 = regions[0]["region_attributes"]["label"]
        str_2 = regions[1]["region_attributes"]["label"]
        return str_1 + " " + str_2
    else: return None


num_sample = 0
num_true = 0

for path in labels:
    data = get_json_label(path)

    name = data["filename"]
    image = cv2.imread(PATH_IMG + "/" +name)

    regions = data["regions"]
    str_label = get_label(regions)
    
    if str_label is None: continue

    result = model.get_result(image)
    if result is None: continue
    
    num_sample += 1
    predict_label = result["recognizer_result"]
    if predict_label == str_label: num_true += 1

print(num_sample, num_true, round(num_true / num_sample, 2))

# PATH_HISTORY = r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\save_history"
# list_json_file = os.listdir(PATH_HISTORY)
# out_json = {}
# for json_file in list_json_file:
#     json_path = os.path.join(PATH_HISTORY, json_file)
#     with open(json_path, "r") as f:
#         data = json.load(f)
#         for r in data["data"]["data"]:
#             image_out_path = r["history"]["image_out_path"]
#             if image_out_path:
#                 print(image_out_path)
#                 image_out_path = image_out_path.split("/")[1]
#                 out_json[image_out_path] = "image_out_path"
#             image_in_path = r["history"]["image_in_path"]
#             if image_in_path:
#                 print(image_in_path)
#                 image_in_path = image_in_path.split("/")[1]
#                 out_json[image_in_path] = "image_in_path"

# PATH_HISTORY = r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\save_abnormal"
# list_json_file = os.listdir(PATH_HISTORY)
# for json_file in list_json_file:
#     json_path = os.path.join(PATH_HISTORY, json_file)
#     with open(json_path, "r") as f:
#         data = json.load(f)
#         for r in data["data"]["data"]:
#             image_path = r["abnormal_action"]["image_path"]
#             if image_path:
#                 print(image_path)
#                 image_path = image_path.split("/")[1]
#                 out_json[image_path] = "image_path"

# with open(r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\save_history.json", "w") as f:
#     json.dump(out_json, f)

# PATH_IMG =  r"E:\Image_Picture_VTV_to_train\IMAGE_NO_HISTORY"
# PATH_OUT =  r"E:\Image_Picture_VTV_to_train\RESULT_NO_HISTORY"

# with open("log_infer.txt", "w") as f:
#     f.write("start infer")
# with open("log_infer_error.txt", "w") as f:
#     f.write("start infer error")
# # with open("E:\Image_Picture_VTV_to_train\save_history.json", "r") as f:
# #     image_map = json.load(f)

# model = PlateRecognition()

# list_cong_name = os.listdir(PATH_IMG)
# for cong_name in list_cong_name:
#     cong_path = os.path.join(PATH_IMG, cong_name)
#     out_cong_path = os.path.join(PATH_OUT, cong_name)
#     if not os.path.exists(out_cong_path):
#         os.mkdir(out_cong_path)
#     list_date = os.listdir(cong_path)
#     for date in list_date:
#         date_path = os.path.join(cong_path, date)
#         out_date_path = os.path.join(out_cong_path, date)
#         if not os.path.exists(out_date_path):
#             os.mkdir(out_date_path)
        
#         if not os.path.exists(out_date_path + '/image_in_path'):
#             os.mkdir(out_date_path + '/image_in_path')
#         if not os.path.exists(out_date_path + '/image_out_path'):
#             os.mkdir(out_date_path + '/image_out_path')

#         list_files = os.listdir(date_path)
#         for filename in list_files:
#             file_path = os.path.join(date_path, filename)
#             # if not image_map.get(filename, ""):
#             #     continue
#             # out_path = os.path.join(out_date_path, image_map[filename], filename)

#             out_path = os.path.join(out_date_path, filename)
#             try:
#                 img = cv2.imread(file_path)
#                 result = model.get_result(img)
#                 if result is not None:
#                     bbox = result["bbox"]
#                     recognizer_result = result["recognizer_result"]
#                     img = cv2.putText(img, recognizer_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
#                             1, (255, 0, 0) , 2, cv2.LINE_AA) 
#                     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=2)
#                     cv2.imwrite(out_path, img)
#                     print(f"{file_path}\t{out_path}")
#                     with open('log_infer.txt', 'a') as f:
#                         f.write(f"{file_path}\t{out_path}\n")
#             except Exception as e:
#                 print(f"ERROR\t{file_path}\t{out_path}\t{e}")
#                 with open('log_infer_error.txt', 'a') as f:
#                     f.write(f"ERROR\t{file_path}\t{out_path}\t{e}\n")


# PATH_HISTORY = r"E:\Image_Picture_VTV_to_train\save_history"
# list_json_file = os.listdir(PATH_HISTORY)
# img_file = []
# for json_file in list_json_file:
#     json_path = os.path.join(PATH_HISTORY, json_file)
#     with open(json_path, "r") as f:
#         data = json.load(f)
#         for r in data["data"]["data"]:
#             image_out_path = r["history"]["image_out_path"]
#             if image_out_path:
#                 image_out_path = image_out_path.split("/")[1]
#                 img_file.append(image_out_path)
#             image_in_path = r["history"]["image_in_path"]
#             if image_in_path:
#                 image_in_path = image_in_path.split("/")[1]
#                 img_file.append(image_in_path)
#             overview_image_in_path = r["history"]["overview_image_in_path"]
#             if overview_image_in_path:
#                 overview_image_in_path = overview_image_in_path.split("/")[1]
#                 img_file.append(overview_image_in_path)
#             overview_image_out_path = r["history"]["overview_image_out_path"]
#             if overview_image_out_path:
#                 overview_image_out_path = overview_image_out_path.split("/")[1]
#                 img_file.append(overview_image_out_path)

# PATH_IMG =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\2024-06-05"
# PATH_IMG_OUT =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\cong43-2024-06-05-plate"

# with open(r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\save_history.json", "r") as f:
#     image_map = json.load(f)

# list_files = os.listdir(PATH_IMG)
# for filename in list_files:
#     file_path = os.path.join(PATH_IMG, filename)
#     if filename in image_map:
#         shutil.copyfile(file_path, os.path.join(PATH_IMG_OUT, filename))

# PATH_IMG =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\congvom"

# with open(r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\save_history.json", "r") as f:
#     image_map = json.load(f)

# list_date = os.listdir(PATH_IMG)
# for date_str in list_date:
#     date_path = os.path.join(PATH_IMG, date_str)
#     list_file = os.listdir(date_path)
#     for filename in list_file:
#         file_path = os.path.join(date_path, filename)
#         if filename not in image_map:
#             os.remove(file_path)


# PATH_IMG =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\cong43\2024-06-05"
# PATH_OUT =  r"E:\Image_Picture_VTV_to_train\NEW_04_06_24\cong43\infer-error-2024-06-05"

# files = [
#     "21e63668-4806-4385-a265-e991f0653952.jpg", 
#     "36fb00ad-d470-4b2a-bfda-dc41d8ddb898.jpg", 
#     "65c006cc-95d0-4ebd-a64d-b931f32cddd6.jpg",
#     ]
# files = os.listdir(PATH_OUT)

# model = PlateRecognition()

# for file_name in files:
#     path_file = os.path.join(PATH_IMG, file_name)
#     img = cv2.imread(path_file)
#     result = model.get_result(img)
#     if result is not None:
#         bbox = result["bbox"]
#         recognizer_result = result["recognizer_result"]
#         img = cv2.putText(img, recognizer_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
#                    1, (255, 0, 0) , 2, cv2.LINE_AA) 
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=2)
#         cv2.imwrite(os.path.join(PATH_OUT, file_name.replace(".jpg", "-new.jpg")), img)