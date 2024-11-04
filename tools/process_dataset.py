import glob, cv2, json, numpy, os, time

root_image_folder = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\Image"
label = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\Label\*.json"
save_crop_folder = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\crop"
save_ocr_folder = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\ocr"
ocr_label = r"D:\TECHPRO\Source Code\Evaluate Plate Recognition Model\BienXanh\ocr_label.txt"

labels = glob.glob(label)

def get_json_label(path: str) -> dict:
    json_data = json.load(open(path,"r"))
    data = json_data["_via_img_metadata"]
    data = list(data.values())[0]

    return data

def get_class_id(data: dict) -> int:
        cls = None
        regions = data["regions"]
        if len(regions) == 1 or len(regions) == 2 :
            cls = 0
        elif len(regions) == 3:
            cls = 1
        
        return cls

def crop_plate():
    for path in labels:
        data = get_json_label(path)

        name = data["filename"]
        image = cv2.imread(root_image_folder + "/" +name)

        try:
            H,W,_ = image.shape
        except:
            print(name)
            continue

        regions = data["regions"]
        region = regions[-1]
        xs = region["shape_attributes"]["all_points_x"]
        ys = region["shape_attributes"]["all_points_y"]
        min_x = max(min(xs) - 3, 0)
        min_y = max(min(ys) - 3, 0)
        max_x = min(max(xs) + 3, W)
        max_y = min(max(ys) + 3, H)
        image = image[min_y:max_y, min_x:max_x, :]
        
        cv2.imwrite(save_crop_folder + "/" + name,image)

def process_ocr_dataset():
    from src.ai_engine.model.recognizer import CornerDetector, process_segment, crop_transform
    corner = CornerDetector(None)

    for path in labels:
        data = get_json_label(path)

        name = data["filename"]
        cls = get_class_id(data)
        if int(cls) == 0:
            shape = (128, 32)
        elif int(cls) == 1:
            shape = (128, 64)
        else:
            return ""
        
        image = cv2.imread(save_crop_folder + "/" +name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))

        image = image.copy()
        
        segs = corner.predict(image)
        res_seg = [process_segment(segs[i], 0.07) *
                   4 for i in range(segs.shape[0])]
        if not res_seg[0].size:
            return ""
        
        res_seg = [crop_transform(res_seg[0], image)]
        res_seg = cv2.resize(res_seg[0], shape)
        str_line_label = ""

        if int(cls) == 0:
            file_name = name
            path_save = os.path.join(save_ocr_folder, file_name)
            
            regions = data["regions"]
            region = regions[0]
            label = region["region_attributes"]["label"]
            str_line_label = f"{file_name} {label}"

            cv2.imwrite(path_save, res_seg)

        elif int(cls) == 1:
            seg1 = numpy.array(res_seg)[:32, :]
            file_name = "1_" + name
            path_save = os.path.join(save_ocr_folder, file_name)
            
            regions = data["regions"]
            region = regions[0]
            label = region["region_attributes"]["label"]
            str_line_label = f"{file_name} {label}"
            cv2.imwrite(path_save, seg1)

            seg2 = numpy.array(res_seg)[32:, :]
            file_name = "2_" + name
            path_save = os.path.join(save_ocr_folder, file_name)
            
            regions = data["regions"]
            region = regions[1]
            label = region["region_attributes"]["label"]
            str_line_label += f"\n{file_name} {label}"
            cv2.imwrite(path_save, seg2)
        
        with open(ocr_label, "a+") as f:
            if str_line_label:
                f.write(f"{str_line_label}\n")

def get_more_data():
    import random, shutil
    more_label_file = ""
    more_folder_image = ""
    folder_image = ""
    label_file = ""
    
    dataset = []
    with open(more_label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n" or not line:
                continue
            line = line.replace("\n", "")
            dataset.append(line)
    
    random.shuffle(dataset)
    dataset = dataset[:300]
    for line in dataset:
        line = line.split(" ")
        stop_path = 0
        for index, text in enumerate(line):
            if ".jpg" in text:
                stop_path = index
        stop_path += 1
        path = " ".join(line[:stop_path])
        label = " ".join(line[stop_path:])
        if not os.path.exists(os.path.join(more_folder_image, path)):
            continue
        
        shutil.copy(os.path.join(more_folder_image, path), os.path.join(folder_image, path))
        with open(label_file, "a+"):
            f.write(f"\n{path} {label}")
        

crop_plate()
# process_ocr_dataset()