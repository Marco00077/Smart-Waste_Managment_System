import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Waste Classifier")
        
        # Get screen dimensions and set window to 90% of screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.85)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#1a1a2e')
        self.root.state('zoomed')  # Maximize window on Windows
        
        self.model = None
        self.class_names = []
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#16213e', height=90)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🌍 AI Waste Classifier",
            font=('Segoe UI', 26, 'bold'),
            bg='#16213e',
            fg='#00ff88'
        )
        title_label.pack(pady=(12, 2))
        
        subtitle_label = tk.Label(
            header_frame,
            text="Identify Biodegradable & Non-Biodegradable Waste",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='#a8dadc'
        )
        subtitle_label.pack()
        
        # Main content with scrollbar
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(main_container, bg='#1a1a2e', highlightthickness=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        scrollable_frame = tk.Frame(self.canvas, bg='#1a1a2e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window centered
        self.canvas_window = self.canvas.create_window(
            (0, 0), 
            window=scrollable_frame, 
            anchor="n"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Center the content when canvas is resized
        def _configure_canvas(event):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        self.canvas.bind("<Configure>", _configure_canvas)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Content frame inside scrollable area - centered
        content_frame = tk.Frame(scrollable_frame, bg='#1a1a2e')
        content_frame.pack(expand=True, pady=20)
        
        # Image display area
        self.image_frame = tk.Frame(
            content_frame, 
            bg='#0f3460', 
            relief=tk.FLAT,
            highlightbackground='#00ff88',
            highlightthickness=2
        )
        self.image_frame.pack(pady=10)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="📷\n\nNo Image Selected\n\nClick 'Select Image' to begin",
            font=('Segoe UI', 14),
            bg='#0f3460',
            fg='#a8dadc',
            justify=tk.CENTER,
            compound='center'
        )
        # Set minimum size for empty state
        self.image_label.config(width=85, height=25)
        self.image_label.pack(padx=30, pady=30)
        
        # File path display
        self.path_label = tk.Label(
            content_frame,
            text="",
            font=('Segoe UI', 9),
            bg='#1a1a2e',
            fg='#7f8c8d',
            wraplength=900
        )
        self.path_label.pack(pady=8)
        
        # Result display
        self.result_frame = tk.Frame(
            content_frame, 
            bg='#16213e',
            relief=tk.FLAT,
            highlightbackground='#00ff88',
            highlightthickness=2
        )
        self.result_frame.pack(fill=tk.X, pady=10)
        
        result_title = tk.Label(
            self.result_frame,
            text="Classification Result",
            font=('Segoe UI', 11, 'bold'),
            bg='#16213e',
            fg='#a8dadc'
        )
        result_title.pack(pady=(12, 5))
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Awaiting Classification...",
            font=('Segoe UI', 17, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        )
        self.result_label.pack(pady=8)
        
        self.confidence_label = tk.Label(
            self.result_frame,
            text="Confidence: -",
            font=('Segoe UI', 12),
            bg='#16213e',
            fg='#00ff88'
        )
        self.confidence_label.pack(pady=(0, 12))
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg='#1a1a2e')
        button_frame.pack(pady=20)
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁  Select Image",
            command=self.select_image,
            font=('Segoe UI', 13, 'bold'),
            bg='#0077b6',
            fg='white',
            activebackground='#005f8f',
            activeforeground='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.FLAT,
            borderwidth=0
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔍  Classify Waste",
            command=self.classify_image,
            font=('Segoe UI', 13, 'bold'),
            bg='#555555',
            fg='#999999',
            activebackground='#00cc6f',
            activeforeground='#1a1a2e',
            padx=30,
            pady=15,
            cursor='hand2',
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.classify_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️  Clear",
            command=self.clear_image,
            font=('Segoe UI', 13, 'bold'),
            bg='#e63946',
            fg='white',
            activebackground='#c92a35',
            activeforeground='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.FLAT,
            borderwidth=0
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Footer
        footer_frame = tk.Frame(content_frame, bg='#1a1a2e')
        footer_frame.pack(pady=15)
        
        footer_label = tk.Label(
            footer_frame,
            text="Powered by TensorFlow & MobileNetV2",
            font=('Segoe UI', 9),
            bg='#1a1a2e',
            fg='#7f8c8d'
        )
        footer_label.pack()
    
    def load_model(self):
        """Load the trained model and class names"""
        model_path = 'waste_classifier_model.h5'
        class_names_path = 'class_names.txt'
        
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Not Found",
                f"Model file '{model_path}' not found!\n\n"
                "Please train the model first using:\n"
                "python model/train_simple.py"
            )
            self.root.destroy()
            return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.class_names = ['biodegradable', 'non_biodegradable']
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select a waste image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                self.current_image = image
                self.current_image_path = file_path
                
                # Resize for display - much larger size
                display_image = image.copy()
                # Calculate size to fit nicely (max 700x500)
                max_width = 700
                max_height = 500
                
                # Get original dimensions
                orig_width, orig_height = display_image.size
                
                # Calculate scaling to fit within max dimensions while maintaining aspect ratio
                scale = min(max_width / orig_width, max_height / orig_height)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                
                display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                
                # Remove width/height constraints when showing image
                self.image_label.configure(image=photo, text="", width=0, height=0)
                self.image_label.image = photo
                
                # Show file path
                filename = os.path.basename(file_path)
                self.path_label.config(text=f"📄 {filename}")
                
                # Enable classify button with proper colors
                self.classify_btn.config(state=tk.NORMAL, bg='#00ff88', fg='#1a1a2e')
                
                # Reset results
                self.result_label.config(text="Awaiting Classification...", fg='#ffffff')
                self.confidence_label.config(text="Confidence: -")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def classify_image(self):
        """Classify the selected image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if self.model is None:
            messagebox.showwarning("No Model", "Model not loaded!")
            return
        
        try:
            # Show processing status
            self.result_label.config(text="🔄 Processing...", fg='#00ff88')
            self.root.update()
            
            # Preprocess image
            img = self.current_image.resize((224, 224))
            img_array = np.array(img)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # Debug: Print predictions
            print(f"Predictions: {predictions[0]}")
            print(f"Predicted class index: {predicted_class}")
            print(f"Class name: {self.class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Display results
            class_name = self.class_names[predicted_class].replace('_', ' ').upper()
            
            # Color code and emoji based on classification
            is_biodegradable = 'biodegradable' in self.class_names[predicted_class].lower() and 'non' not in self.class_names[predicted_class].lower()
            
            if is_biodegradable:
                color = '#00ff88'  # Green
                emoji = '🌱'
                description = "Can decompose naturally"
            else:
                color = '#ff6b6b'  # Red
                emoji = '♻️'
                description = "Cannot decompose naturally • Requires proper disposal"
            
            self.result_label.config(
                text=f"{emoji}  {class_name}",
                fg=color
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1f}% • {description}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
    
    def clear_image(self):
        """Clear the current image and results"""
        self.current_image = None
        self.current_image_path = None
        # Restore width/height for empty state
        self.image_label.configure(
            image='',
            text="📷\n\nNo Image Selected\n\nClick 'Select Image' to begin",
            width=85,
            height=25
        )
        self.path_label.config(text="")
        self.result_label.config(text="Awaiting Classification...", fg='#ffffff')
        self.confidence_label.config(text="Confidence: -")
        self.classify_btn.config(state=tk.DISABLED, bg='#555555')

def main():
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
