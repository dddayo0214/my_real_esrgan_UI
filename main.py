import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
from os import listdir
from os.path import join

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("圖片超解析度處理")
        self.root.geometry("1000x600")

        # 檢查並載入 Real-ESRGAN 模型
        if not hasattr(ImageProcessorApp, 'model'):
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )
            weight_path = 'models/RealESRGAN_x4plus.pth'
            if not os.path.exists(weight_path):
                messagebox.showerror("錯誤", f"模型權重檔案不存在: {weight_path}")
                sys.exit(1)

            state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)['params_ema']
            model.load_state_dict(state_dict, strict=True)

            ImageProcessorApp.model = RealESRGANer(
                scale=4, model_path=weight_path, model=model,
                tile=0, tile_pad=10, pre_pad=0, half=False
            )
        self.model = ImageProcessorApp.model

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

        self.select_img_button = tk.Button(button_frame, text="選擇檔案", command=self.select_image)
        self.select_img_button.pack(side=tk.LEFT, padx=5)

        self.select_dir_button = tk.Button(button_frame, text="選擇檔案", command=self.select_dir)
        self.select_dir_button.pack(side=tk.LEFT, padx=5)

        self.process_img_button = tk.Button(button_frame, text="處理圖片", command=self.start_img_process, state=tk.DISABLED)
        self.process_img_button.pack(side=tk.LEFT, padx=5)

        self.process_dir_button = tk.Button(button_frame, text="處理資料夾", command=self.start_dir_process, state=tk.DISABLED)
        self.process_dir_button.pack(side=tk.LEFT, padx=5)

        self.save_img_button = tk.Button(button_frame, text="保存圖片", command=self.save_image, state=tk.DISABLED)
        self.save_img_button.pack(side=tk.LEFT, padx=5)

        self.save_dir_button = tk.Button(button_frame, text="保存資料夾", command=self.save_dir, state=tk.DISABLED)
        self.save_dir_button.pack(side=tk.LEFT, padx=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片文件", "*.jpg *.jpeg")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_label)
            self.process_img_button.config(state=tk.NORMAL)

    def select_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            imgs = listdir(dir_path)
            self.original_dir = []
            self.original_dirname = []
            for path in imgs:
                self.original_dirname.append(path)
                path = join(dir_path, path)
                self.original_dir.append(Image.open(path))
                self.process_dir_button.config(state=tk.NORMAL)

    def process_image(self):
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("警告", "請先選擇圖片！")
            return

        img = self.original_image
        if img.mode == "P":  # Palette 模式（可能有透明度）
            img = img.convert("RGBA")  # 先轉為 RGBA
        if img.mode == "RGBA":  # 轉換為 RGB，避免透明通道問題
            img = img.convert("RGB")

        img = np.array(img)  # 轉換為 NumPy 陣列
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 轉換為 OpenCV 格式 (BGR)

        try:
            self.progress = ttk.Progressbar(root, length=200, mode='indeterminate')
            self.progress.place(relx=0.5, rely=0.7, anchor="center")
            self.progress.start()

            # 使用 Real-ESRGAN 進行超解析度處理
            output, _ = self.model.enhance(img, outscale=4)

            #進度條
            self.progress.stop()
            self.progress.destroy()
            self.show_complete_window()

            # 將處理後的圖片轉換回 PIL 格式
            self.processed_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            self.display_image(self.processed_image, self.processed_label)
            self.save_img_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("錯誤", f"圖片處理失敗！\n{e}")

    def process_dir(self):
        if not hasattr(self, 'original_dir'):
            messagebox.showwarning("警告", "請先選擇圖片！")
            return
        
        try:
            self.processed_dir = []
            for img in self.original_dir:
                if img.mode == "P":  # Palette 模式（可能有透明度）
                    img = img.convert("RGBA")  # 先轉為 RGBA
                if img.mode == "RGBA":  # 轉換為 RGB，避免透明通道問題
                    img = img.convert("RGB")

                img = np.array(img)  # 轉換為 NumPy 陣列
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 轉換為 OpenCV 格式 (BGR)

                
                self.progress = ttk.Progressbar(root, length=200, mode='indeterminate')
                self.progress.place(relx=0.5, rely=0.7, anchor="center")
                self.progress.start()

                # 使用 Real-ESRGAN 進行超解析度處理
                output, _ = self.model.enhance(img, outscale=4)

                #進度條
                self.progress.stop()
                self.progress.destroy()
                
                # 將處理後的圖片轉換回 PIL 格式
                self.processed_dir.append(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))

            self.show_complete_window()
            self.save_dir_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("錯誤", f"圖片處理失敗！\n{e}")

    def start_img_process(self):
        thread = threading.Thread(target=self.process_image)
        thread.start()

    def start_dir_process(self):
        thread = threading.Thread(target=self.process_dir)
        thread.start()

    def show_complete_window(self):
        complete_window = tk.Toplevel(self.root)  # 創建新的視窗
        complete_window.title("Completion")
        complete_label = tk.Label(complete_window, text="Progress Complete!", font=("Arial", 14))
        complete_label.pack(pady=20)
        ok_button = tk.Button(complete_window, text="OK", command=complete_window.destroy)
        ok_button.pack(pady=10)

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

    def save_dir(self):
        file_path = filedialog.askdirectory()
        if not file_path:
            file_path = "D:\\alan_program\\my_real_esrgan_UI\\outputs"
        i = 0
        for img in self.processed_dir:
            img.save(join(file_path, f"{self.original_dirname[i]}_output.png"))
            i = i + 1
        messagebox.showinfo("保存成功", "資料夾已成功保存！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
