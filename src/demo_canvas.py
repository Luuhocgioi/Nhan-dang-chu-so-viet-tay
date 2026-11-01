"""Demo Tkinter: canvas và mở ảnh dùng chung Predictor."""
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from .predict import Predictor
from .preprocess import read_image


class DigitApp:
    def __init__(self, root, predictor):
        self.root, self.predictor = root, predictor
        root.title('Nhận dạng chữ số viết tay — CNN / MNIST')
        root.configure(bg='#f1f5f9')
        root.resizable(False, False)
        tk.Label(root, text='NHẬN DẠNG CHỮ SỐ', font=('Segoe UI', 20, 'bold'), bg='#f1f5f9', fg='#0f172a').pack(pady=(20, 4))
        tk.Label(root, text='Vẽ một chữ số 0–9 ở giữa ô, rồi bấm Nhận dạng.', bg='#f1f5f9').pack(pady=(0, 12))
        frame = tk.Frame(root, bg='#f1f5f9')
        frame.pack(padx=24)
        self.canvas = tk.Canvas(frame, width=280, height=280, bg='black', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0, 20))
        self.preview = tk.Label(frame, bg='#e2e8f0', text='Ảnh 28 × 28', width=20)
        self.preview.grid(row=0, column=1)
        self.result = tk.Label(frame, text='—', font=('Segoe UI', 32, 'bold'), bg='#f1f5f9', fg='#2563eb')
        self.result.grid(row=1, column=1)
        self.scores = tk.Label(frame, text='', font=('Consolas', 10), bg='#f1f5f9', justify='left')
        self.scores.grid(row=2, column=1)
        buttons = tk.Frame(root, bg='#f1f5f9')
        buttons.pack(pady=16)
        for label, command in [('Nhận dạng', self.recognize), ('Xóa', self.clear), ('Mở ảnh', self.open_image)]:
            tk.Button(buttons, text=label, command=command, font=('Segoe UI', 11), padx=14, pady=7).pack(side='left', padx=5)
        tk.Label(root, text='Điểm mô hình không bảo đảm kết quả đúng. Mỗi lần chỉ nhập một chữ số.', bg='#f1f5f9').pack(padx=20, pady=(0, 16))
        self.canvas.bind('<Button-1>', self.start)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', lambda event: setattr(self, 'last', None))
        self.clear()

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 0)
        self.painter = ImageDraw.Draw(self.image)
        self.last = None
        self.result.config(text='—')
        self.scores.config(text='')
        self.preview.config(image='', text='Ảnh 28 × 28')

    def start(self, event):
        self.last = (event.x, event.y)
        self.canvas.create_oval(event.x-9, event.y-9, event.x+9, event.y+9, fill='white', outline='white')
        self.painter.ellipse((event.x-9, event.y-9, event.x+9, event.y+9), fill=255)

    def draw(self, event):
        if self.last is None:
            self.start(event)
            return
        x, y = event.x, event.y
        self.canvas.create_line(*self.last, x, y, fill='white', width=18, capstyle=tk.ROUND)
        self.painter.line([self.last, (x, y)], fill=255, width=18)
        self.painter.ellipse((x-9, y-9, x+9, y+9), fill=255)
        self.last = (x, y)

    def show_prediction(self, image):
        try:
            result, processed = self.predictor.predict(image)
            self.preview_image = ImageTk.PhotoImage(Image.fromarray(processed).resize((112, 112), Image.Resampling.NEAREST))
            self.preview.config(image=self.preview_image, text='', width=112)
            self.result.config(text=str(result['digit']))
            self.scores.config(text='\n'.join(f'{i}: {score:6.1%}' for i, score in enumerate(result['probabilities'])))
        except ValueError as exc:
            self.result.config(text='—')
            self.scores.config(text='')
            self.preview.config(image='', text='Ảnh 28 × 28', width=20)
            messagebox.showinfo('Kiểm tra ảnh', str(exc))

    def recognize(self):
        self.show_prediction(np.array(self.image))

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Ảnh', '*.png *.jpg *.jpeg *.bmp'), ('Tất cả', '*.*')])
        if path:
            try:
                self.show_prediction(read_image(path))
            except ValueError as exc:
                messagebox.showerror('Không đọc được ảnh', str(exc))


def main():
    parser = argparse.ArgumentParser(description='Demo canvas và ảnh')
    parser.add_argument('--model')
    args = parser.parse_args()
    try:
        predictor = Predictor(args.model)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')
    root = tk.Tk()
    DigitApp(root, predictor)
    root.mainloop()


if __name__ == '__main__':
    main()
