
import os
import random
import shutil
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Configurations ---
MAIN_DATA_PATH = "/scratch/shared/beegfs/gaurikad/hd-epic/frames"
# Creates the data directory inside the folder where main.py is located
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_SAMPLES = 15 # Adjust this to sample more or fewer images

def setup_and_sample_data():
    """Samples images from different P0X folders if the data folder is empty."""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    existing_images = [f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith(('.jpg', '.png'))]
    if existing_images:
        print(f"Data folder already contains {len(existing_images)} images. Skipping sampling.")
        return existing_images

    print("Scanning dataset to sample diverse images...")
    all_frames = []
    
    # 1. Iterate through P0X folders
    if os.path.exists(MAIN_DATA_PATH):
        p_dirs = [d for d in os.listdir(MAIN_DATA_PATH) if d.startswith('P0') and os.path.isdir(os.path.join(MAIN_DATA_PATH, d))]
        
        for p_dir in p_dirs:
            p_path = os.path.join(MAIN_DATA_PATH, p_dir)
            video_dirs = [d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))]
            
            # 2. Iterate through video folders (e.g., P01-20240202-110250)
            for v_dir in video_dirs:
                v_path = os.path.join(p_path, v_dir)
                frames = [f for f in os.listdir(v_path) if f.endswith('.jpg')]
                
                if frames:
                    # Pick 1 random frame per video folder to ensure high diversity
                    random_frame = random.choice(frames)
                    all_frames.append(os.path.join(v_path, random_frame))
                    
        # 3. Randomly select the final subset
        sampled = random.sample(all_frames, min(NUM_SAMPLES, len(all_frames)))
        
        for src in sampled:
            # Prefix the filename with the video folder so names don't clash
            parent_name = os.path.basename(os.path.dirname(src))
            img_name = os.path.basename(src)
            dest_name = f"{parent_name}_{img_name}"
            shutil.copy(src, os.path.join(LOCAL_DATA_DIR, dest_name))
            
        print(f"Successfully copied {len(sampled)} images to {LOCAL_DATA_DIR}")
        return [f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith(('.jpg', '.png'))]
    else:
        print(f"Warning: {MAIN_DATA_PATH} not found. Running in UI-only mode.")
        return []

class AnnotatorApp:
    def __init__(self, root, images):
        self.root = root
        self.root.title("HD-Epic VQA Annotator")
        self.images = sorted(images)
        self.current_idx = 0
        
        self.current_bbox = None # [xmin, ymin, xmax, ymax]
        self.rect_id = None
        self.box_size = tk.IntVar(value=512) # Default to 512x512
        
        self.setup_ui()
        self.load_image()

    def setup_ui(self):
        # Top Frame: Controls
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.lbl_info = tk.Label(top_frame, text="", font=("Arial", 14, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=20)
        
        tk.Label(top_frame, text="Box Size:").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(top_frame, text="512x512", variable=self.box_size, value=512).pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="124x124", variable=self.box_size, value=124).pack(side=tk.LEFT)
        
        # Middle Frame: Canvas with Scrollbars (for HD images)
        mid_frame = tk.Frame(self.root)
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(mid_frame, cursor="cross", bg="gray")
        hbar = tk.Scrollbar(mid_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Bottom Frame: Q&A and Save
        bot_frame = tk.Frame(self.root, pady=10)
        bot_frame.pack(fill=tk.X)
        
        tk.Label(bot_frame, text="Question:").grid(row=0, column=0, padx=10, sticky="e")
        self.ent_question = tk.Entry(bot_frame, width=80)
        self.ent_question.grid(row=0, column=1, pady=5)
        self.ent_question.insert(0, "What is the primary object inside the green box? Answer concisely.")
        
        tk.Label(bot_frame, text="Answer:").grid(row=1, column=0, padx=10, sticky="e")
        self.ent_answer = tk.Entry(bot_frame, width=80)
        self.ent_answer.grid(row=1, column=1, pady=5)
        
        btn_save = tk.Button(bot_frame, text="Save & Next", command=self.save_and_next, bg="green", fg="white", font=("Arial", 12, "bold"))
        btn_save.grid(row=0, column=2, rowspan=2, padx=20, sticky="nsew")

    def load_image(self):
        if self.current_idx >= len(self.images):
            messagebox.showinfo("Done", "All images have been annotated!")
            self.root.quit()
            return
            
        img_name = self.images[self.current_idx]
        self.lbl_info.config(text=f"Image {self.current_idx + 1} / {len(self.images)}: {img_name}")
        
        img_path = os.path.join(LOCAL_DATA_DIR, img_name)
        self.pil_img = Image.open(img_path).convert("RGB")
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.config(scrollregion=(0, 0, self.pil_img.width, self.pil_img.height))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        self.current_bbox = None
        self.ent_answer.delete(0, tk.END)

    def on_canvas_click(self, event):
        # Adjust for scrollbar position
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        size = self.box_size.get()
        half_s = size / 2
        
        xmin, ymin = cx - half_s, cy - half_s
        xmax, ymax = cx + half_s, cy + half_s
        self.current_bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            
        self.rect_id = self.canvas.create_rectangle(xmin, ymin, xmax, ymax, outline="lime", width=3)

    def save_and_next(self):
        if not self.current_bbox:
            messagebox.showwarning("Warning", "Please click on the image to draw a bounding box.")
            return
            
        answer = self.ent_answer.get().strip()
        if not answer:
            messagebox.showwarning("Warning", "Please provide an answer.")
            return
            
        img_name = self.images[self.current_idx]
        question = self.ent_question.get().strip()
        
        # Save to JSON
        data = {
            "image": img_name,
            "bbox": self.current_bbox,
            "question": question,
            "answer": answer
        }
        
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(LOCAL_DATA_DIR, json_name)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        # Move to next
        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    images = setup_and_sample_data()
    if images:
        root = tk.Tk()
        # Maximize window depending on OS
        try:
            root.state('zoomed') 
        except:
            root.attributes('-zoomed', True)
            
        app = AnnotatorApp(root, images)
        root.mainloop()