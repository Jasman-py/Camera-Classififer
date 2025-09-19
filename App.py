import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import numpy as np

import Camera  # your Camera.py
import model   # model.py with train, predict, load_trained_model functions

class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title
        self.window.title(self.window_title)

        self.counters = [1, 1]
        self.auto_predict = False

        # Initialize camera
        self.camera = Camera.Camera()

        # Load model if exists
        self.model = model.load_trained_model()

        # Ask class names
        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        # Initialize GUI
        self.init_gui()

        # Update loop
        self.delay = 15
        self.update()

        self.window.attributes('-topmost', True)

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50,
                                       command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50,
                                       command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict_frame)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not ret:
            print("No frame captured")
            return

        # Create directories if not exist
        if not os.path.exists('1'):
            os.mkdir('1')
        if not os.path.exists('2'):
            os.mkdir('2')

        # Save frame
        cv.imwrite(f'{class_num}/frame{self.counters[class_num-1]}.jpg',
                   cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        print(f"Saved frame{self.counters[class_num-1]} to directory {class_num}")

        self.counters[class_num-1] += 1

    def reset(self):
        for directory in ['1', '2']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

        self.counters = [1, 1]
        self.class_label.config(text="CLASS")

    def train_model(self):
        self.model = model.train()  # trains and loads the model

    def predict_frame(self):
        if self.model is None:
            print("Model not loaded.")
            return
        ret, frame = self.camera.get_frame()
        if not ret:
            print("No frame captured")
            return

        # Convert to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        class_idx = model.predict(gray_frame, self.model)
        class_name = self.classname_one if class_idx == 0 else self.classname_two
        self.class_label.config(text=f"CLASS: {class_name}")

    def update(self):
        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.auto_predict and self.model:
            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            class_idx = model.predict(gray_frame, self.model)
            class_name = self.classname_one if class_idx == 0 else self.classname_two
            self.class_label.config(text=f"CLASS: {class_name}")

        self.window.after(self.delay, self.update)


if __name__ == "__main__":
    App()
    tk.mainloop()
