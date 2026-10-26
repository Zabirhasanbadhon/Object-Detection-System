import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


# ============================================================
# 1. CONFIGURATION: YOLO MODEL FILE PATHS
# ============================================================

cfg_file = (

    r"C:\Users\ZaBir-HaSan\Desktop\Object Detection System\yolov3.cfg"
)

weights_file = (

    r"C:\Users\ZaBir-HaSan\Desktop\Object Detection System\yolov3.weights"
)

names_file = (
    r"C:\Users\ZaBir-HaSan\Desktop\Object Detection System\coco.names"
)


# ============================================================
# 2. COLOR THEME
# ============================================================

BG_COLOR = "#F8F5EE"
CARD_COLOR = "#FFFEFA"
PRIMARY_COLOR = "#426B5A"
PRIMARY_HOVER = "#345546"
SECONDARY_COLOR = "#E8E1D5"
TEXT_COLOR = "#26352E"
MUTED_TEXT = "#7D877F"
BORDER_COLOR = "#E7E1D7"
SUCCESS_COLOR = "#3F7958"
DANGER_COLOR = "#B84D4D"
SHADOW_COLOR = "#D8D1C5"

CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40
INPUT_SIZE = (416, 416)


# ============================================================
# 3. MAIN APPLICATION
# ============================================================

class YOLODetectionApp:

    def __init__(self, root):

        self.root = root

        self.root.title("ZaBir | IntelliVision Studio")
        self.root.geometry("1250x850")
        self.root.minsize(1050, 720)
        self.root.configure(bg=BG_COLOR)

        # Model information
        self.net = None
        self.classes = []
        self.output_layers = []

        # Image information
        self.image_path = None
        self.original_image = None
        self.detected_image = None
        self.detection_results = []

        # Image references
        self.original_photo = None
        self.output_photo = None

        # Zoom controls
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        # Detection state
        self.is_detecting = False

        self.setup_styles()
        self.create_interface()

        # Load model in background
        self.status_var.set("Loading YOLOv3 model...")
        self.detect_button.configure(state="disabled")

        threading.Thread(
            target=self.load_model,
            daemon=True
        ).start()

    # ========================================================
    # 4. STYLES
    # ========================================================

    def setup_styles(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Modern.Treeview",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            fieldbackground=CARD_COLOR,
            rowheight=30,
            borderwidth=0,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Modern.Treeview.Heading",
            background=SECONDARY_COLOR,
            foreground=TEXT_COLOR,
            font=("Segoe UI", 10, "bold"),
            padding=8,
            borderwidth=0
        )

        style.map(
            "Modern.Treeview",
            background=[("selected", "#DCE8DF")],
            foreground=[("selected", TEXT_COLOR)]
        )

        style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor=SECONDARY_COLOR,
            background=PRIMARY_COLOR,
            bordercolor=BG_COLOR,
            lightcolor=PRIMARY_COLOR,
            darkcolor=PRIMARY_COLOR
        )

    # ========================================================
    # 5. SHADOW BUTTON
    # ========================================================

    def create_shadow_button(
        self,
        parent,
        text,
        command,
        bg=PRIMARY_COLOR,
        fg="white",
        shadow=SHADOW_COLOR,
        width=180,
        height=43,
        font=("Segoe UI", 10, "bold")
    ):

        container = tk.Frame(
            parent,
            bg=shadow,
            width=width,
            height=height
        )

        container.pack_propagate(False)

        button = tk.Button(
            container,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=PRIMARY_HOVER,
            activeforeground="white",
            font=font,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=5
        )

        button.place(
            x=0,
            y=0,
            relwidth=1,
            relheight=1,
            width=-4,
            height=-4
        )

        def on_enter(event):

            if str(button["state"]) != "disabled":
                button.configure(bg=PRIMARY_HOVER)

        def on_leave(event):

            if str(button["state"]) != "disabled":
                button.configure(bg=bg)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        return container, button

    # ========================================================
    # 6. BUILD GUI
    # ========================================================

    def create_interface(self):

        main_frame = tk.Frame(
            self.root,
            bg=BG_COLOR
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=18
        )

        # HEADER
        header = tk.Frame(
            main_frame,
            bg=BG_COLOR
        )

        header.pack(
            fill="x",
            pady=(0, 18)
        )

        title_frame = tk.Frame(
            header,
            bg=BG_COLOR
        )

        title_frame.pack(
            side="left",
            fill="x",
            expand=True
        )

        tk.Label(
            title_frame,
            text="🤖 ZaBir HaSan",
            font=("Segoe UI", 25, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Intelligent Object Detection System",
            font=("Segoe UI", 12),
            fg=MUTED_TEXT,
            bg=BG_COLOR
        ).pack(anchor="w", pady=(3, 0))

        self.model_badge = tk.Label(
            header,
            text="●  INITIALIZING",
            font=("Segoe UI", 9, "bold"),
            fg="#9A762F",
            bg="#F0E8D7",
            padx=14,
            pady=9
        )

        self.model_badge.pack(
            side="right",
            anchor="center"
        )

        # CONTROL CARD
        control_card = tk.Frame(
            main_frame,
            bg=CARD_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )

        control_card.pack(
            fill="x",
            pady=(0, 18)
        )

        control_inner = tk.Frame(
            control_card,
            bg=CARD_COLOR
        )

        control_inner.pack(
            fill="x",
            padx=18,
            pady=15
        )

        # Select Image button
        self.select_container, self.select_button = (
            self.create_shadow_button(
                control_inner,
                "＋  Select Image",
                self.select_image,
                width=165
            )
        )

        self.select_container.pack(
            side="left",
            padx=(0, 12)
        )

        # Detect Objects button
        self.detect_container, self.detect_button = (
            self.create_shadow_button(
                control_inner,
                "✦  Detect Objects",
                self.start_detection,
                width=180
            )
        )

        self.detect_container.pack(
            side="left",
            padx=(0, 12)
        )

        # Save Output button
        self.save_container, self.save_button = (
            self.create_shadow_button(
                control_inner,
                "↓  Save Output",
                self.save_output,
                bg="#E9E3D8",
                fg=TEXT_COLOR,
                shadow="#D0C7B9",
                width=155
            )
        )

        self.save_container.pack(
            side="left",
            padx=(0, 12)
        )

        self.save_button.configure(
            activebackground="#DCD3C5",
            activeforeground=TEXT_COLOR,
            state="disabled"
        )

        # NEW: Reset button beside Save Output
        self.reset_container, self.reset_button = (
            self.create_shadow_button(
                control_inner,
                "↻  Clear",
                self.reset_application,
                bg="#AC7571",
                fg=TEXT_COLOR,
                shadow="#D0C7B9",
                width=115
            )
        )

        self.reset_container.pack(
            side="left"
        )

        # Filename display
        self.file_label = tk.Label(
            control_inner,
            text="No image selected",
            font=("Segoe UI", 10),
            fg=MUTED_TEXT,
            bg=CARD_COLOR,
            anchor="e"
        )

        self.file_label.pack(
            side="right",
            fill="x",
            expand=True,
            padx=(15, 0)
        )

        # IMAGE PREVIEW AREA
        preview_container = tk.Frame(
            main_frame,
            bg=BG_COLOR
        )

        preview_container.pack(
            fill="both",
            expand=True
        )

        preview_container.grid_columnconfigure(
            0,
            weight=1,
            uniform="preview"
        )

        preview_container.grid_columnconfigure(
            1,
            weight=1,
            uniform="preview"
        )

        preview_container.grid_rowconfigure(
            0,
            weight=1
        )

        # Original image card
        original_card = tk.Frame(
            preview_container,
            bg=CARD_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )

        original_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 9)
        )

        # Output image card
        output_card = tk.Frame(
            preview_container,
            bg=CARD_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )

        output_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(9, 0)
        )

        # Original image heading
        tk.Label(
            original_card,
            text="ORIGINAL IMAGE",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_COLOR,
            bg=CARD_COLOR
        ).pack(
            anchor="w",
            padx=16,
            pady=(15, 2)
        )

        tk.Label(
            original_card,
            text="Your selected image",
            font=("Segoe UI", 9),
            fg=MUTED_TEXT,
            bg=CARD_COLOR
        ).pack(
            anchor="w",
            padx=16,
            pady=(0, 12)
        )

        # Original image display
        self.original_display = tk.Label(
            original_card,
            text="Select an image to get started\n\n"
                 "JPG  •  JPEG  •  PNG",
            font=("Segoe UI", 12),
            fg=MUTED_TEXT,
            bg="#F6F3EC",
            relief="flat",
            anchor="center"
        )

        self.original_display.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 14)
        )

        # Output image heading
        output_header = tk.Frame(
            output_card,
            bg=CARD_COLOR
        )

        output_header.pack(
            fill="x",
            padx=16,
            pady=(15, 2)
        )

        tk.Label(
            output_header,
            text="DETECTED IMAGE",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_COLOR,
            bg=CARD_COLOR
        ).pack(side="left")

        self.detection_badge = tk.Label(
            output_header,
            text="0 OBJECTS",
            font=("Segoe UI", 8, "bold"),
            fg=PRIMARY_COLOR,
            bg="#E5EEE7",
            padx=9,
            pady=4
        )

        self.detection_badge.pack(side="right")

        tk.Label(
            output_card,
            text="AI detection results",
            font=("Segoe UI", 9),
            fg=MUTED_TEXT,
            bg=CARD_COLOR
        ).pack(
            anchor="w",
            padx=16,
            pady=(0, 12)
        )

        # Output image display with zoom
        self.output_display = tk.Canvas(
            output_card,
            bg="#F6F3EC",
            highlightthickness=0,
            bd=0,
            cursor="crosshair"
        )

        self.output_display.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 14)
        )

        # Mouse-wheel zoom
        self.output_display.bind(
            "<MouseWheel>",
            self.zoom_output
        )

        # Drag to move enlarged image
        self.output_display.bind(
            "<ButtonPress-1>",
            self.start_pan
        )

        self.output_display.bind(
            "<B1-Motion>",
            self.pan_output
        )

        # Double-click to reset zoom
        self.output_display.bind(
            "<Double-Button-1>",
            self.reset_zoom
        )

        # RESULTS SECTION
        results_card = tk.Frame(
            main_frame,
            bg=CARD_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )

        results_card.pack(
            fill="x",
            pady=(18, 0)
        )

        results_header = tk.Frame(
            results_card,
            bg=CARD_COLOR
        )

        results_header.pack(
            fill="x",
            padx=16,
            pady=(12, 8)
        )

        tk.Label(
            results_header,
            text="Detection Summary",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_COLOR,
            bg=CARD_COLOR
        ).pack(side="left")

        self.total_label = tk.Label(
            results_header,
            text="Total detections: 0",
            font=("Segoe UI", 10, "bold"),
            fg=PRIMARY_COLOR,
            bg=CARD_COLOR
        )

        self.total_label.pack(side="right")

        # Results table
        table_frame = tk.Frame(
            results_card,
            bg=CARD_COLOR
        )

        table_frame.pack(
            fill="x",
            padx=16,
            pady=(0, 12)
        )

        self.results_table = ttk.Treeview(
            table_frame,
            columns=("number", "object", "confidence"),
            show="headings",
            height=3,
            style="Modern.Treeview"
        )

        self.results_table.heading(
            "number",
            text="#"
        )

        self.results_table.heading(
            "object",
            text="Detected Object"
        )

        self.results_table.heading(
            "confidence",
            text="Confidence"
        )

        self.results_table.column(
            "number",
            width=55,
            anchor="center"
        )

        self.results_table.column(
            "object",
            width=300,
            anchor="w"
        )

        self.results_table.column(
            "confidence",
            width=150,
            anchor="center"
        )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.results_table.yview
        )

        self.results_table.configure(
            yscrollcommand=scrollbar.set
        )

        self.results_table.pack(
            side="left",
            fill="x",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        # FOOTER
        footer = tk.Frame(
            main_frame,
            bg=BG_COLOR
        )

        footer.pack(
            fill="x",
            pady=(12, 0)
        )

        self.status_var = tk.StringVar(
            value="Initializing application..."
        )

        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=MUTED_TEXT,
            bg=BG_COLOR,
            anchor="w"
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

        tk.Label(
            footer,
            text="YOLOv3  |  OpenCV  |  CPU Mode",
            font=("Segoe UI", 9),
            fg=MUTED_TEXT,
            bg=BG_COLOR
        ).pack(side="right")

    # ========================================================
    # 7. LOAD YOLO MODEL
    # ========================================================

    def load_model(self):

        try:

            for file_path in (
                cfg_file,
                weights_file,
                names_file
            ):

                if not os.path.isfile(file_path):

                    raise FileNotFoundError(
                        f"Required file not found:\n{file_path}"
                    )

            with open(
                names_file,
                "r",
                encoding="utf-8"
            ) as file:

                classes = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            net = cv2.dnn.readNetFromDarknet(
                cfg_file,
                weights_file
            )

            net.setPreferableBackend(
                cv2.dnn.DNN_BACKEND_OPENCV
            )

            net.setPreferableTarget(
                cv2.dnn.DNN_TARGET_CPU
            )

            layer_names = net.getLayerNames()

            output_layers = [
                layer_names[i - 1]
                for i in net.getUnconnectedOutLayers().flatten()
            ]

            self.root.after(
                0,
                lambda: self.on_model_loaded(
                    net,
                    classes,
                    output_layers
                )
            )

        except Exception as error:

            error_message = str(error)

            self.root.after(
                0,
                lambda msg=error_message:
                self.on_model_error(msg)
            )

    def on_model_loaded(
        self,
        net,
        classes,
        output_layers
    ):

        self.net = net
        self.classes = classes
        self.output_layers = output_layers

        self.model_badge.configure(
            text="●  MODEL READY",
            fg=SUCCESS_COLOR,
            bg="#E5EEE7"
        )

        self.detect_button.configure(
            state="normal"
        )

        self.status_var.set(
            "YOLOv3 loaded successfully. Select an image to begin."
        )

    def on_model_error(self, error_message):

        self.model_badge.configure(
            text="●  MODEL ERROR",
            fg=DANGER_COLOR,
            bg="#F5E5E2"
        )

        self.status_var.set(
            "Could not load the YOLOv3 model."
        )

        messagebox.showerror(
            "Model Loading Error",
            "Unable to load YOLOv3.\n\n"
            f"{error_message}\n\n"
            "Please check the paths to your CFG, weights, "
            "and COCO names files."
        )

    # ========================================================
    # 8. SELECT IMAGE
    # ========================================================

    def select_image(self):

        if self.is_detecting:
            return

        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:

            img = cv2.imread(
                file_path,
                cv2.IMREAD_COLOR
            )

            if img is None:
                raise ValueError(
                    "OpenCV could not read this image."
                )

            self.image_path = file_path
            self.original_image = img
            self.detected_image = None
            self.detection_results = []

            # Reset zoom for the new image
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0

            self.show_image(
                self.original_image,
                self.original_display,
                "original"
            )

            # Reset output preview
            self.output_display.delete("all")

            self.output_display.create_text(
                self.output_display.winfo_width() // 2,
                self.output_display.winfo_height() // 2,
                text="Ready for detection\n\n"
                     "Click Detect Objects to begin",
                fill=MUTED_TEXT,
                font=("Segoe UI", 11),
                justify="center",
                tags="placeholder"
            )

            self.output_photo = None

            self.clear_results()

            self.file_label.configure(
                text=os.path.basename(file_path),
                fg=TEXT_COLOR
            )

            self.detection_badge.configure(
                text="0 OBJECTS"
            )

            self.save_button.configure(
                state="disabled"
            )

            self.status_var.set(
                f"Image loaded: {os.path.basename(file_path)}"
            )

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                f"Unable to open this image.\n\n{error}"
            )

    # ========================================================
    # 9. DISPLAY IMAGE
    # ========================================================

    def show_image(
        self,
        image,
        display_widget,
        image_type
    ):

        if image is None:
            return

        # Original image preview
        if image_type == "original":

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(rgb_image)

            display_widget.update_idletasks()

            max_width = max(
                display_widget.winfo_width() - 20,
                250
            )

            max_height = max(
                display_widget.winfo_height() - 20,
                200
            )

            pil_image.thumbnail(
                (max_width, max_height),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(pil_image)

            display_widget.configure(
                image=photo,
                text="",
                bg="#F6F3EC"
            )

            self.original_photo = photo

            return

        # Output image with zoom
        self.output_display.update_idletasks()

        canvas_width = self.output_display.winfo_width()
        canvas_height = self.output_display.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            return

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb_image)

        max_width = max(
            canvas_width - 20,
            100
        )

        max_height = max(
            canvas_height - 20,
            100
        )

        fit_image = pil_image.copy()

        fit_image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        new_width = max(
            1,
            int(fit_image.width * self.zoom_factor)
        )

        new_height = max(
            1,
            int(fit_image.height * self.zoom_factor)
        )

        zoomed_image = fit_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(zoomed_image)

        self.output_display.delete("all")

        self.output_display.create_image(
            canvas_width // 2 + self.pan_x,
            canvas_height // 2 + self.pan_y,
            image=photo,
            anchor="center",
            tags="output_image"
        )

        self.output_photo = photo

    # ========================================================
    # 10. OUTPUT IMAGE ZOOM CONTROLS
    # ========================================================

    def zoom_output(self, event):

        if self.detected_image is None:
            return

        if event.delta > 0:
            self.zoom_factor *= 1.15
        else:
            self.zoom_factor /= 1.15

        self.zoom_factor = max(
            0.5,
            min(self.zoom_factor, 5.0)
        )

        self.show_image(
            self.detected_image,
            self.output_display,
            "output"
        )

    def start_pan(self, event):

        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def pan_output(self, event):

        if self.detected_image is None:
            return

        self.pan_x += event.x - self.drag_start_x
        self.pan_y += event.y - self.drag_start_y

        self.drag_start_x = event.x
        self.drag_start_y = event.y

        self.show_image(
            self.detected_image,
            self.output_display,
            "output"
        )

    def reset_zoom(self, event=None):

        if self.detected_image is None:
            return

        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.show_image(
            self.detected_image,
            self.output_display,
            "output"
        )

    # ========================================================
    # 11. RESET APPLICATION
    # ========================================================

    def reset_application(self):

        # Do not reset while detection is running
        if self.is_detecting:
            return

        # Clear selected image and detection data
        self.image_path = None
        self.original_image = None
        self.detected_image = None
        self.detection_results = []

        # Reset zoom and image position
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.drag_start_x = 0
        self.drag_start_y = 0

        # Clear original image preview
        self.original_display.configure(
            image="",
            text="Select an image to get started\n\n"
                 "JPG  •  JPEG  •  PNG",
            font=("Segoe UI", 12),
            fg=MUTED_TEXT,
            bg="#F6F3EC"
        )

        self.original_photo = None

        # Clear output image preview
        self.output_display.delete("all")

        self.output_display.create_text(
            self.output_display.winfo_width() // 2,
            self.output_display.winfo_height() // 2,
            text="Your detected image will appear here\n\n"
                 "Select an image and click Detect Objects",
            fill=MUTED_TEXT,
            font=("Segoe UI", 11),
            justify="center",
            tags="placeholder"
        )

        self.output_photo = None

        # Reset filename
        self.file_label.configure(
            text="No image selected",
            fg=MUTED_TEXT
        )

        # Clear results table
        self.clear_results()

        # Reset save button
        self.save_button.configure(
            state="disabled"
        )

        # Restore detection badge
        self.detection_badge.configure(
            text="0 OBJECTS"
        )

        # Keep the loaded model ready for the next image
        if self.net is not None:

            self.detect_button.configure(
                state="normal"
            )

            self.model_badge.configure(
                text="●  MODEL READY",
                fg=SUCCESS_COLOR,
                bg="#E5EEE7"
            )

            self.status_var.set(
                "Reset complete. Select a new image to begin."
            )

        else:

            self.detect_button.configure(
                state="disabled"
            )

            self.status_var.set(
                "Reset complete. Waiting for YOLOv3 model..."
            )

    # ========================================================
    # 12. START OBJECT DETECTION
    # ========================================================

    def start_detection(self):

        if self.net is None:

            messagebox.showwarning(
                "Model Not Ready",
                "The YOLOv3 model has not loaded successfully."
            )

            return

        if self.original_image is None:

            messagebox.showwarning(
                "No Image Selected",
                "Please select an image before detecting objects."
            )

            return

        if self.is_detecting:
            return

        self.is_detecting = True

        self.detect_button.configure(
            state="disabled",
            text="Detecting..."
        )

        self.select_button.configure(
            state="disabled"
        )

        self.save_button.configure(
            state="disabled"
        )

        self.status_var.set(
            "Detecting objects... Please wait."
        )

        threading.Thread(
            target=self.detect_objects_worker,
            daemon=True
        ).start()

    # ========================================================
    # 13. YOLO OBJECT DETECTION
    # ========================================================

    def detect_objects_worker(self):

        try:

            img = self.original_image.copy()

            height, width = img.shape[:2]

            blob = cv2.dnn.blobFromImage(
                img,
                scalefactor=1 / 255.0,
                size=INPUT_SIZE,
                mean=(0, 0, 0),
                swapRB=True,
                crop=False
            )

            self.net.setInput(blob)

            outputs = self.net.forward(
                self.output_layers
            )

            boxes = []
            confidences = []
            class_ids = []

            # Process predictions
            for output in outputs:

                for detection in output:

                    objectness = float(detection[4])
                    class_scores = detection[5:]

                    if len(class_scores) == 0:
                        continue

                    class_id = int(
                        np.argmax(class_scores)
                    )

                    class_probability = float(
                        class_scores[class_id]
                    )

                    confidence = (
                        objectness * class_probability
                    )

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    center_x = int(
                        detection[0] * width
                    )

                    center_y = int(
                        detection[1] * height
                    )

                    box_width = int(
                        detection[2] * width
                    )

                    box_height = int(
                        detection[3] * height
                    )

                    x = int(
                        center_x - box_width / 2
                    )

                    y = int(
                        center_y - box_height / 2
                    )

                    boxes.append(
                        [x, y, box_width, box_height]
                    )

                    confidences.append(confidence)
                    class_ids.append(class_id)

            # Remove overlapping boxes
            indexes = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD
            )

            if len(indexes) > 0:

                indexes = np.array(
                    indexes
                ).flatten().tolist()

            else:

                indexes = []

            results = []

            # Draw boxes and labels
            for i in indexes:

                x, y, box_width, box_height = boxes[i]

                class_id = class_ids[i]
                confidence = confidences[i]

                if class_id >= len(self.classes):
                    continue

                label = self.classes[class_id]

                x1 = max(0, x)
                y1 = max(0, y)

                x2 = min(
                    width - 1,
                    x + box_width
                )

                y2 = min(
                    height - 1,
                    y + box_height
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # Green bounding box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (70, 150, 95),
                    2
                )

                label_text = (
                    f"{label} {confidence * 100:.1f}%"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2

                (text_width, text_height), baseline = (
                    cv2.getTextSize(
                        label_text,
                        font,
                        font_scale,
                        thickness
                    )
                )

                label_y = max(
                    y1,
                    text_height + 10
                )

                # Label background
                cv2.rectangle(
                    img,
                    (
                        x1,
                        label_y - text_height - 8
                    ),
                    (
                        min(
                            width - 1,
                            x1 + text_width + 8
                        ),
                        label_y + baseline - 4
                    ),
                    (70, 150, 95),
                    cv2.FILLED
                )

                # Label text
                cv2.putText(
                    img,
                    label_text,
                    (x1 + 4, label_y - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

                results.append(
                    {
                        "name": label,
                        "confidence": confidence
                    }
                )

            # Update GUI on main thread
            self.root.after(
                0,
                lambda result_image=img,
                       result_list=results:
                self.on_detection_complete(
                    result_image,
                    result_list
                )
            )

        except Exception as error:

            error_message = str(error)

            self.root.after(
                0,
                lambda msg=error_message:
                self.on_detection_error(msg)
            )

    # ========================================================
    # 14. UPDATE GUI AFTER DETECTION
    # ========================================================

    def on_detection_complete(
        self,
        result_image,
        results
    ):

        self.is_detecting = False

        self.detect_button.configure(
            state="normal",
            text="✦  Detect Objects"
        )

        self.select_button.configure(
            state="normal"
        )

        self.detected_image = result_image
        self.detection_results = results

        # Reset zoom for each new detection
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.show_image(
            self.detected_image,
            self.output_display,
            "output"
        )

        # Update results table
        self.clear_results()

        for index, item in enumerate(
            self.detection_results,
            start=1
        ):

            self.results_table.insert(
                "",
                "end",
                values=(
                    index,
                    item["name"],
                    f"{item['confidence'] * 100:.2f}%"
                )
            )

        total = len(self.detection_results)

        self.total_label.configure(
            text=f"Total detections: {total}"
        )

        self.detection_badge.configure(
            text=f"{total} OBJECTS"
        )

        self.save_button.configure(
            state="normal"
        )

        self.status_var.set(
            f"Detection complete. Found {total} object(s)."
        )

        self.model_badge.configure(
            text="●  MODEL READY",
            fg=SUCCESS_COLOR,
            bg="#E5EEE7"
        )

    def on_detection_error(self, error_message):

        self.is_detecting = False

        self.detect_button.configure(
            state="normal",
            text="✦  Detect Objects"
        )

        self.select_button.configure(
            state="normal"
        )

        self.status_var.set(
            "An error occurred during object detection."
        )

        messagebox.showerror(
            "Detection Error",
            "Unable to detect objects.\n\n"
            f"{error_message}"
        )

    # ========================================================
    # 15. CLEAR RESULTS
    # ========================================================

    def clear_results(self):

        for item in self.results_table.get_children():
            self.results_table.delete(item)

        self.total_label.configure(
            text="Total detections: 0"
        )

        self.detection_badge.configure(
            text="0 OBJECTS"
        )

    # ========================================================
    # 16. SAVE DETECTED IMAGE
    # ========================================================

    def save_output(self):

        if self.detected_image is None:

            messagebox.showwarning(
                "No Output",
                "Please detect objects before saving the output."
            )

            return

        default_name = "detected_output.jpg"

        if self.image_path:

            original_name = os.path.splitext(
                os.path.basename(self.image_path)
            )[0]

            default_name = (
                f"{original_name}_detected.jpg"
            )

        save_path = filedialog.asksaveasfilename(
            title="Save Detected Image",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png"),
                ("All Files", "*.*")
            ]
        )

        if not save_path:
            return

        try:

            success = cv2.imwrite(
                save_path,
                self.detected_image
            )

            if not success:

                raise IOError(
                    "OpenCV could not write the output file."
                )

            self.status_var.set(
                f"Output saved: {os.path.basename(save_path)}"
            )

            messagebox.showinfo(
                "Image Saved",
                "Your detected image has been saved successfully!\n\n"
                f"Location:\n{save_path}"
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                f"Unable to save the output image.\n\n{error}"
            )


# ============================================================
# 17. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = YOLODetectionApp(root)

    root.mainloop()