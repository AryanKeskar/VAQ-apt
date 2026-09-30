import os
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class VQAAnnotator:
    def __init__(self, root, output_file, path_to_visualization):
        self.root = root
        self.root.title("HD-EPIC VQA Annotator")
        
        # Force a set window size so macOS does not render a blank gray screen
        self.root.geometry("1200x800")
        
        self.output_file = output_file
        self.path_to_visualization = path_to_visualization
        
        # 1. Load existing annotations to enforce the "1 image 1 time" rule
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            with open(self.output_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
            
        # 2. Get all images, then strictly filter out those already in the JSON
        all_images = sorted([f for f in os.listdir(path_to_visualization) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.images = [img for img in all_images if img not in self.annotations]
        
        self.current_idx = 0
        
        # --- UI SETUP ---
        # Main container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side: Image container
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(side=tk.LEFT, expand=True)
        
        # Right side: Control panel for Q&A
        self.control_frame = tk.Frame(self.main_frame, width=400)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=50)
        
        # Q&A Input fields
        tk.Label(self.control_frame, text="Question:", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 5))
        self.q_entry = tk.Entry(self.control_frame, width=35, font=("Arial", 14))
        self.q_entry.pack(anchor="w", pady=(0, 20))
        
        tk.Label(self.control_frame, text="Answer:", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 5))
        self.a_entry = tk.Entry(self.control_frame, width=35, font=("Arial", 14))
        self.a_entry.pack(anchor="w", pady=(0, 40))
        
        # Buttons
        self.save_btn = tk.Button(self.control_frame, text="Save & Next (Enter)", command=self.save_and_next, fg="green", font=("Arial", 14))
        self.save_btn.pack(fill=tk.X, pady=10)
        
        self.skip_btn = tk.Button(self.control_frame, text="Skip Image", command=self.skip, font=("Arial", 14))
        self.skip_btn.pack(fill=tk.X, pady=10)
        
        self.quit_btn = tk.Button(self.control_frame, text="Save & Quit", command=root.quit, font=("Arial", 14))
        self.quit_btn.pack(fill=tk.X, pady=10)
        
        # Bind the Enter key to the save function
        self.root.bind('<Return>', lambda event: self.save_and_next())
        
        # 3. Start the loop or exit if everything is already annotated
        if not self.images:
            messagebox.showinfo("Complete", "All images in this folder have already been annotated!")
            self.root.after(100, self.root.quit)
        else:
            self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.images):
            messagebox.showinfo("Complete", "You have finished all available images!")
            self.root.quit()
            return
            
        image_name = self.images[self.current_idx]
        image_path = os.path.join(self.path_to_visualization, image_name)
        
        try:
            img = Image.open(image_path)
            # Cap the size so it fits alongside the text boxes
            img.thumbnail((750, 750)) 
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            self.skip()
            return
            
        self.root.title(f"HD-EPIC VQA Annotator - {image_name} ({self.current_idx + 1}/{len(self.images)})")
        
        self.q_entry.delete(0, tk.END)
        self.a_entry.delete(0, tk.END)
        self.q_entry.focus_set()

    def save_and_next(self):
        question = self.q_entry.get().strip()
        answer = self.a_entry.get().strip()
        
        if not question or not answer:
            messagebox.showwarning("Missing Data", "Please enter both a Question and an Answer, or click 'Skip Image'.")
            return
            
        image_name = self.images[self.current_idx]
        
        # Write exactly in the requested format: { image_name : [{Question: Q, answer: A}] }
        self.annotations[image_name] = [{
            "Question": question,
            "answer": answer
        }]
        
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)
            
        self.current_idx += 1
        self.load_image()
        
    def skip(self):
        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    data_dir = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/gaze/P01-20240202-161354_gaze_frames_data"
    path_to_visualization = os.path.join(data_dir, "data_124", "visualization_124")
    
    output_dir = os.path.join(data_dir, "question_answer_pairs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "vqa_annotations.json")
    
    root = tk.Tk()
    app = VQAAnnotator(root, output_file, path_to_visualization)
    root.mainloop()