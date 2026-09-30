import os
from PIL import Image
import cv2
import json



def main(image_path, output_dir, gaze_data_124, gaze_data_512, context_patch_sizes):
    """Experiments to be done:
    1. Gaze as local context + Image Resize as global context
    2. Gaze as local context + Center Zoom as global context
    3. Gaze as local context + Gaze Zoom as global context

    """

    #First lets deal with the 124 patch size Experiment 1:
    experiment_1_folder = "Resize_experiment"
    Experiment_1_context_patches_path = os.path.join(output_dir, experiment_1_folder)
    if not os.path.exists(Experiment_1_context_patches_path):
        os.makedirs(Experiment_1_context_patches_path, exist_ok=True)
    for image_file in os.listdir(image_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image = Image.open(os.path.join(image_path, image_file))
            #lets work on image resize first:
            name = "context_patches"
            for context_size in context_patch_sizes:
                context_patch_dir = os.path.join(Experiment_1_context_patches_path, name + str(context_size))
                if not os.path.exists(context_patch_dir):
                    os.makedirs(context_patch_dir, exist_ok=True)
                resized_image = image.resize((context_size, context_size))
                resized_image.save(os.path.join(context_patch_dir, f"{os.path.splitext(image_file)[0]}_resized_{context_size}.jpg"))
            
    #Experiment 2: Gaze as local context + Center Zoom as global context
    experiment_2_folder = "Center_zoom_experiment"
    Experiment_2_context_patches_path = os.path.join(output_dir, experiment_2_folder)
    if not os.path.exists(Experiment_2_context_patches_path):
        os.makedirs(Experiment_2_context_patches_path, exist_ok=True)
    for image_file in os.listdir(image_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image = Image.open(os.path.join(image_path, image_file))
            #lets work on center zoom first:
            name = "context_patches"
            for context_size in context_patch_sizes:
                context_patch_dir = os.path.join(Experiment_2_context_patches_path, name + str(context_size))
                if not os.path.exists(context_patch_dir):
                    os.makedirs(context_patch_dir, exist_ok=True)
                width, height = image.size
                left = (width - context_size) / 2
                top = (height - context_size) / 2
                right = (width + context_size) / 2
                bottom = (height + context_size) / 2
                center_cropped_image = image.crop((left, top, right, bottom))
                center_cropped_image.save(os.path.join(context_patch_dir, f"{os.path.splitext(image_file)[0]}_center_cropped_{context_size}.jpg"))

    #Experiment 3: Gaze as local context + Gaze Zoom as global context
    experiment_3_folder = "Gaze_zoom_experiment"
    Experiment_3_context_patches_path = os.path.join(output_dir, experiment_3_folder)
    if not os.path.exists(Experiment_3_context_patches_path):
        os.makedirs(Experiment_3_context_patches_path, exist_ok=True)
    for image_file in os.listdir(image_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image = Image.open(os.path.join(image_path, image_file))
            #lets work on gaze zoom first:
            name = "context_patches"
            for context_size in context_patch_sizes:
                with open(gaze_data_124, "r") as f:
                    gaze_124_data = json.load(f)
                context_patch_dir = os.path.join(Experiment_3_context_patches_path, name + str(context_size))
                if not os.path.exists(context_patch_dir):
                    os.makedirs(context_patch_dir, exist_ok=True)
                #print(type(gaze_124_data))
                counter = 0
                for gaze_entry in gaze_124_data:
                    counter += 1
                    if gaze_entry["video_name"] == image_file.split("_")[0]:
                       
                        non_normalized_x = gaze_entry["x"]
                        non_normalized_y = gaze_entry["y"]
                        width, height = image.size
                        gaze_x = int(non_normalized_x * width)
                        gaze_y = int(non_normalized_y * height)
                        left = max(0, gaze_x - context_size // 2)
                        top = max(0, gaze_y - context_size // 2)
                        right = min(width, gaze_x + context_size // 2)
                        bottom = min(height, gaze_y + context_size // 2)
                        gaze_cropped_image = image.crop((left, top, right, bottom))
                        gaze_cropped_image.save(os.path.join(context_patch_dir, f"{os.path.splitext(image_file)[0]}_gaze_cropped_{context_size}.jpg"))
                print(f'the counter is {counter}')
    return 0    

if __name__ == "__main__":

    images_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/gaze/P01-20240202-161354_gaze_frames_data/images"
    output_dir = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/gaze/P01-20240202-161354_gaze_frames_data"
    gaze_data_124 = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/gaze/P01-20240202-161354_gaze_frames_data/data_124/data_124_json/124_gaze_data.json"
    gaze_data_512 = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/gaze/P01-20240202-161354_gaze_frames_data/data_512/data_512_json/512_gaze_data.json"
    context_patch_sizes = [224, 228, 320, 384, 512, 640, 704, 896]
    main(images_path, output_dir, gaze_data_124, gaze_data_512, context_patch_sizes)