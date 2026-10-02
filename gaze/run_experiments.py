import torch
import os
import json
import random
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
# --- Set Up Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "P01-20240202-161354_gaze_frames_data")
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# Removed LOCAL_PATCH_SIZES. We only use your exact anotated bbox now.
CONTEXT_SIZES = [224, 288, 320, 384, 512, 640, 704, 896]
SEED = 42
def calculate_cosine_similarity(vec1, vec2):
    # Compute the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the L2 norms (magnitudes) of each vector
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Cosine similarity is the dot product divided by the product of the norms
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Handle edge case of zero vectors
        
    return float(dot_product / (norm1 * norm2))

def load_llava_model(model_id="llava-hf/llava-1.5-7b-hf"):
    print(f"Downloading and loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True
    )
    print("================ LLaVA 1.5 7B Architecture ================\n")
    print(model)
    print("\n================ Model Loaded Successfully ================")
    return processor, model

def analyze_multi_image(processor, model, context_img, local_img, query):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context Image:\n"},
                {"type": "image"},
                {"type": "text", "text": f"\nTarget Zoom Image:\n"},
                {"type": "image"},
                {"type": "text", "text": f"\nQuestion: {query}"},
                {"type": "text", "text": f"\n Answer the question based on the context image and the target zoom image. Provide your answer in 1 or 2 words."}
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=[context_img, local_img], text=prompt, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)

    full_response = processor.decode(output[0], skip_special_tokens=True)
    return full_response.split("ASSISTANT:")[-1].strip()
def run_experiment(experiment):
    #Load the model
    processor, model = load_llava_model()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    #load the qa json file
    images_path = experiment["images_path"]
    path_to_qa_json = experiment["path_to_qa_json"]
   
    gaze_boxes_path = [experiment["path_to_512_gaze_bbox"], experiment["path_to_124_gaze_bbox"]]
    with open(gaze_boxes_path[0], "r") as f:
        gaze_512_data = json.load(f)
    with open(gaze_boxes_path[1], "r") as f:
        gaze_124_data = json.load(f)
    gaze_boxes = [gaze_512_data, gaze_124_data]
    with open(path_to_qa_json, "r") as f:
        qa_data = json.load(f)
    
    experiment_dir = experiment["path_to_experiment_images"]
    results_file_path = experiment["result_dir"]

    current_gaze_box_size = 512 if "512" in gaze_boxes_path[0] else 124
    for gaze_box in gaze_boxes:
        list_of_results_for_experiment = {}
        for folder in os.listdir(experiment_dir):
            print(f'------------------{folder.split("_")[-1]}---------------------')
            folder_path = os.path.join(experiment_dir, folder)
            if os.path.isdir(folder_path):
                data_for_images = []
                for image_file in os.listdir(folder_path):
                    data_for_image = {}
                    if image_file.endswith(".jpg") or image_file.endswith(".png"):
                        image_path = os.path.join(folder_path, image_file)
                        # Load the resized image
                        index = image_file.split("_")[0] + "_" + image_file.split("_")[1]+".jpg"  # Assuming the index is the first two parts of the filename
                        image = Image.open(images_path+"/"+index)
                        center_zoom_image = Image.open(image_path)
                        bbox = next((item["bbox"] for item in gaze_box if item["frame"] == int((index.split('_')[1]).split(".")[0])), None )
                        if bbox:
                            # Pass the bbox coordinates as a tuple to crop the original high-resolution image
                            cropped_patch = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
                            # You can now save or process this cropped patch for your VQA experiments
                            # cropped_patch.save(f"cropped_{index}")
                        else:
                            print(f"Skipping {index}: No bounding box found.")

                        
                        #print(index.split("_")[0], "     ",(index.split('_')[1]).split(".")[0], "       ", bbox)
                        # Get the corresponding QA pair
                        qa_pair = qa_data.get(index, None)[0]
                        response = analyze_multi_image(processor, model, center_zoom_image, cropped_patch, qa_pair.get("Question"))
                        encoded_response = embed_model.encode((response.lower()).strip())
                        #print("\n\n",response +"  ",index,"\n\n")
                        answers = qa_pair.get("answer").split('/')
                        scores = []
                        
                        for answer in answers:
                            encoded_answer = embed_model.encode((answer.lower()).strip())
                            score = calculate_cosine_similarity(encoded_answer, encoded_response)
                            scores.append(score)
                        final_score = max(scores)
                        print("\n\n",qa_pair.get("Question"),"      ",response +"  ",index,"    ",final_score,"     ", scores, "\n\n")
                        data = {
                            "gaze_patch_size": current_gaze_box_size,
                            "context_image_size": int((folder.split("_")[-1]).split("s")[-1]),
                            "gaze_bbox": bbox,
                            "question": qa_pair.get("Question"),
                            "model_response": response,
                            "ground_truth_answers": answers,
                            "final_score": final_score,
                        }
                        data_for_image[image_file.split("_")[0]+image_file.split("_")[1]] = data
                        data_for_images.append(data_for_image)
            list_of_results_for_experiment[folder] = data_for_images
            # Save the results for this resized image folder   
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(results_file_path, "w") as results_file:
                json.dump(list_of_results_for_experiment, results_file, indent=4)
        if current_gaze_box_size == 512:
            current_gaze_box_size = 124
        else:
            current_gaze_box_size = 512

    return 0

if __name__ == "__main__":
    resize_experiment = {
        "path_to_experiment_images": os.path.join(DATA_DIR,"a_Resize_experiment"),
        "images_path" : os.path.join(DATA_DIR, "images"),
        "path_to_qa_json": os.path.join(DATA_DIR, "question_answer_pairs/vqa_annotations.json"),
        "path_to_512_gaze_bbox" : os.path.join(DATA_DIR, "data_512/data_512_json/512_gaze_data.json"),
        "path_to_124_gaze_bbox": os.path.join(DATA_DIR, "data_124/data_124_json/124_gaze_data.json"),
        "result_dir": os.path.join(RESULTS_DIR, f"resize_experiment.json")
    }
    center_zoom_experiment = {
        "path_to_images": os.path.join(DATA_DIR,"b_Center_zoom_experiment"),
        "images_path" : os.path.join(DATA_DIR, "images"),
        "path_to_qa_json": os.path.join(DATA_DIR, "question_answer_pairs/vqa_annotations.json"),
        "path_to_512_gaze_bbox" : os.path.join(DATA_DIR, "data_512/data_512_json/512_gaze_data.json"),
        "path_to_124_gaze_bbox": os.path.join(DATA_DIR, "data_124/data_124_json/124_gaze_data.json"),
        "result_dir": os.path.join(RESULTS_DIR, f"center_zoom_experiment.json")
    }
    gaze_zoom_experiment = {
        "path_to_images": os.path.join(DATA_DIR,"c_Gaze_zoom_experiment"),
        "images_path" : os.path.join(DATA_DIR, "images"),
        "path_to_qa_json": os.path.join(DATA_DIR, "question_answer_pairs/vqa_annotations.json"),
        "path_to_512_gaze_bbox" : os.path.join(DATA_DIR, "data_512/data_512_json/512_gaze_data.json"),
        "path_to_124_gaze_bbox": os.path.join(DATA_DIR, "data_124/data_124_json/124_gaze_data.json"),
        "result_dir": os.path.join(RESULTS_DIR, f"gaze_zoom_experiment.json")
    }
    run_experiment(resize_experiment)
    run_experiment(center_zoom_experiment)
    run_experiment(gaze_zoom_experiment)
