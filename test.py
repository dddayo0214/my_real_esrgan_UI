import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# model_path = 'models/RealESRGAN_x4plus.pth'
# state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']

# model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
# model.load_state_dict(state_dict, strict=True)

# upsampler = RealESRGANer(
#     scale = 4,
#     model_path=model_path,
#     model=model,
#     tile = 0,
#     pre_pad = 0,
#     half=False,
# )

# img = Image.open('inputs/input.jpg').convert('RGB')
# img = np.array(img)

# output, _ = upsampler.enhance(img, outscale=4)

# output_img = Image.fromarray(output)
# output_img.save('output.png')

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("圖片超解析度處理")
        self.root.geometry("800x600")

        # 創建 GUI 介面
        self.create_widgets()

    def create_widgets(self):
        self.original_frame = tk.LabelFrame(self.root, text="原始圖片")
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.original_label = tk.Label(self.original_frame)
        self.original_label.pack()

        self.processed_frame = tk.LabelFrame(self.root, text="處理後圖片")
        self.processed_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.processed_label = tk.Label(self.processed_frame)
        self.processed_label.pack()

        # 按鈕區域
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        self.select_button = tk.Button(button_frame, text="選擇圖片", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.process_button = tk.Button(button_frame, text="處理圖片", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="保存圖片", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片文件", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_label)
            self.process_button.config(state=tk.NORMAL)

    def process_image(self):
        model_path = 'models/RealESRGAN_x4plus.pth'
        state_dict = torch.load(model_path)['params_ema']

        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        model.load_state_dict(state_dict, strict=True)

        self.model = RealESRGANer(
            scale = 4,
            model_path=model_path,
            model=model,
            tile = 0,
            pre_pad = 0,
            half=False,
        )

        if not hasattr(self, 'original_image'):
            messagebox.showwarning("警告", "請先選擇圖片！")
            return

        img = self.original_image
        if img.mode == "P":  # Palette 模式（可能有透明度）
            img = img.convert("RGBA")  # 先轉為 RGBA
        if img.mode == "RGBA":  # 轉換為 RGB，避免透明通道問題
            img = img.convert("RGB")

        img = np.array(img)  # 轉換為 NumPy 陣列

        try:
            output, _ = self.model.enhance(img, outscale=4)

            self.processed_image = Image.fromarray(output)
            self.display_image(self.processed_image, self.processed_label)
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("錯誤", f"圖片處理失敗！\n{e}")

    def display_image(self, image, label):
        width, height = image.size
        ratio = min(350 / width, 350 / height)
        new_size = (int(width * ratio), int(height * ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(resized_image)
        label.config(image=photo)
        label.image = photo

    def save_image(self):
        if not hasattr(self, 'processed_image'):
            messagebox.showwarning("警告", "請先處理圖片！")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg")]
        )
        if file_path:
            self.processed_image.save(file_path)
            messagebox.showinfo("保存成功", "圖片已成功保存！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
