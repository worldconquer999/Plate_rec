import os
import shutil
import json

def get_all_file_name(root_path):
    all_file = {} # filename: path
    folder_1ths = os.listdir(root_path)
    folder_1th_paths = [os.path.join(root_path, folder_1th) \
                                    for folder_1th in folder_1ths \
                                        if os.path.isdir(os.path.join(root_path, folder_1th))]
    
    for folder_1th_path in folder_1th_paths:
        folder_2ths = os.listdir(folder_1th_path)
        folder_2th_paths = [os.path.join(folder_1th_path, folder_2th) \
                                    for folder_2th in folder_2ths \
                                        if os.path.isdir(os.path.join(folder_1th_path, folder_2th))]
        for folder_2th_path in folder_2th_paths:
            list_file: str = os.listdir(folder_2th_path)
            for file_name in list_file:
                if not file_name.endswith('.jpg'): continue
                all_file[file_name] = os.path.join(folder_2th_path, file_name)
    
    return all_file

all_file = get_all_file_name(r"E:\Image_Picture_VTV_to_train\TO_CHECK\FROM_VTV")
all_file_in_history = get_all_file_name(r"E:\Image_Picture_VTV_to_train\TO_CHECK\RESULT_HAS_HISTORY")
all_file_not_in_history = get_all_file_name(r"E:\Image_Picture_VTV_to_train\TO_CHECK\RESULT_NO_HISTORY")
all_file_infer_error = get_all_file_name(r"E:\Image_Picture_VTV_to_train\TO_CHECK\INFER_ERROR")

# dict_img_infer = {**all_file_in_history, **all_file_not_in_history}
# for file_name in dict_img_infer.keys():
#     src_path = all_file[file_name]
#     shutil.copy(src_path, os.path.join(r"E:\Image_Picture_VTV_to_train\TO_CHECK\ALL_IMAGE_INFER", file_name))

# dict_img_infer = all_file_infer_error
# for file_name in dict_img_infer.keys():
#     try:
#         src_path = all_file[file_name]
#         shutil.copy(src_path, os.path.join(r"E:\Image_Picture_VTV_to_train\TO_CHECK\IMAGE_ERROR_TO_LABEL", file_name))
#     except:
#         print(dict_img_infer[file_name])

with open("result_window.json", "r") as f:
    result_window = json.load(f)

with open("result_ubuntu.json", "r") as f:
    result_ubuntu = json.load(f)

num_match = 0
for filename, label in result_ubuntu.items():
    label_window = result_window[filename]
    print(label, label_window)
    if label == label_window: num_match+=1

print(num_match, len(list(result_ubuntu.keys())))