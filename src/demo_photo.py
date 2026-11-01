"""Giao diện chính: đọc ảnh chữ số viết trên giấy, chọn vùng và nhận dạng."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from .predict import Predictor
from .preprocess import read_image
from .detect import recognize_digits, save_result, to_bgr


class PhotoApp:
    WIDTH, HEIGHT = 620, 460

    def __init__(self, root, predictor):
        self.root, self.predictor = root, predictor
        self.image = None
        self.roi = None
        self.drag_start = None
        self.selection_id = None
        root.title('Đọc chữ số viết tay trên giấy')
        root.configure(bg='#f1f5f9')
        root.resizable(False, False)
        tk.Label(root, text='ĐỌC CHỮ SỐ TRÊN GIẤY', font=('Segoe UI', 21, 'bold'), bg='#f1f5f9', fg='#0f172a').pack(pady=(18, 4))
        tk.Label(root, text='Viết nhiều chữ số trên giấy → mở ảnh → tự tìm số và vẽ khung kết quả',
                 font=('Segoe UI', 11), bg='#f1f5f9').pack(pady=(0, 12))
        controls = tk.Frame(root, bg='#f1f5f9')
        controls.pack(fill='x', padx=20, pady=(0, 10))
        for title, callback in [('Mở ảnh giấy…', self.open_image), ('↶ 90°', lambda:self.rotate(False)), ('↷ 90°', self.rotate),
                                ('Chọn toàn ảnh', self.reset_selection), ('Nhận dạng', self.recognize),
                                ('Lưu kết quả…', self.export_result)]:
            ttk.Button(controls, text=title, command=callback).pack(side='left', padx=(0, 8))
        self.advanced_button=ttk.Button(controls,text='Nâng cao ▸',command=self.toggle_advanced)
        self.advanced_button.pack(side='left')
        self.advanced_host=ttk.Frame(root)
        self.advanced_host.pack(fill='x',padx=20)
        self.advanced_panel=ttk.Frame(self.advanced_host)
        self.advanced_open=False
        self.blue_ink = tk.BooleanVar(value=False)
        self.ink_predictor = None
        ttk.Label(self.advanced_panel,text='Các lựa chọn dưới đây chỉ dùng để so sánh thử nghiệm. Dùng hằng ngày không cần thay đổi.').pack(anchor='w')
        ttk.Checkbutton(self.advanced_panel, text='Lọc mực xanh/tím (thử nghiệm, có thể bỏ nét mực khác màu hoặc nét nhỏ)', variable=self.blue_ink).pack(anchor='w')
        selection=ttk.Frame(self.advanced_panel);selection.pack(anchor='w',pady=4)
        ttk.Label(selection,text='Phương pháp:').pack(side='left',padx=(0,8))
        self.method=tk.StringVar(value='Mặc định')
        self.method_models={}
        ttk.Combobox(selection,textvariable=self.method,values=['Mặc định','Dãy số rời','EMNIST chữ/số','YOLO EMNIST','YOLO','CNN'],state='readonly',width=23).pack(side='left')
        ttk.Label(selection,text='Xoay để các chữ số đứng thẳng trước khi nhận dạng.').pack(side='left',padx=12)
        self.filename = tk.StringVar(value='Chưa chọn ảnh')
        tk.Label(root, textvariable=self.filename, bg='#f1f5f9', fg='#475569').pack(anchor='w', padx=20)
        frame = tk.Frame(root, bg='#f1f5f9')
        frame.pack(padx=20, pady=8)
        self.canvas = tk.Canvas(frame, width=self.WIDTH, height=self.HEIGHT, bg='#e2e8f0', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, padx=(0, 18))
        self.canvas.create_text(self.WIDTH//2, self.HEIGHT//2, text='Mở ảnh chụp chữ số trên giấy', font=('Segoe UI', 15), fill='#475569')
        panel = tk.Frame(frame, bg='#f1f5f9', width=240)
        panel.grid(row=0, column=1, sticky='n')
        tk.Label(panel, text='CÁC VÙNG CHỮ SỐ', font=('Segoe UI', 11, 'bold'), bg='#f1f5f9').pack(pady=8)
        self.preview = tk.Label(panel, bg='#e2e8f0', width=22, height=7, text='Chưa nhận dạng')
        self.preview.pack()
        self.result = tk.Label(panel, text='—', font=('Segoe UI', 24, 'bold'), wraplength=230, fg='#2563eb', bg='#f1f5f9')
        self.result.pack(pady=6)
        self.scores = tk.Label(panel, text='', font=('Consolas', 11), bg='#f1f5f9', justify='left')
        self.scores.pack()
        self.status = tk.StringVar(value='Dùng giấy sáng, nét mực đậm; viết các chữ số cách nhau. Mở ảnh rồi bấm Nhận dạng.')
        if getattr(predictor,'numeric_rows',False):
            self.status.set('Chế độ dãy số rời: ưu tiên hàng có ít nhất 3 ký tự cùng cỡ, cách xa nhau; chữ viết cách xa tương tự vẫn có thể gây nhầm.')
        tk.Label(root, textvariable=self.status, bg='#f1f5f9', fg='#334155', wraplength=880, justify='left').pack(fill='x', padx=20, pady=(0, 8))
        tk.Label(root, text='Ưu tiên dãy số rời hoặc bảng số đều. Chữ có bố cục tương tự vẫn có thể bị nhận nhầm.',
                 bg='#f1f5f9', fg='#64748b').pack(padx=20, pady=(0, 16))
        self.canvas.bind('<Button-1>', self.start_selection)
        self.canvas.bind('<B1-Motion>', self.drag_selection)
        self.canvas.bind('<ButtonRelease-1>', self.end_selection)

    def toggle_advanced(self):
        self.advanced_open=not self.advanced_open
        if self.advanced_open:
            self.advanced_panel.pack(fill='x',pady=(0,8))
            self.advanced_button.config(text='Nâng cao ▾')
        else:
            self.advanced_panel.pack_forget()
            self.advanced_button.config(text='Nâng cao ▸')
            self.method.set('Mặc định')
            self.blue_ink.set(False)
            if self.image is not None:self.show_image()

    def clear_result(self):
        self.last_result = None
        self.annotated = None
        self.result.config(text='—')
        self.scores.config(text='')
        self.preview.config(image='', text='Chưa nhận dạng', width=22, height=7)

    def load_path(self, path):
        # Đọc thành công rồi mới thay ảnh đang làm việc.
        image = read_image(path)
        self.image = image
        self.filename.set(Path(path).name)
        self.show_image()

    def open_image(self):
        path = filedialog.askopenfilename(title='Chọn ảnh chụp chữ số trên giấy',
                 filetypes=[('Ảnh', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff'), ('Tất cả', '*.*')])
        if path:
            try:
                self.load_path(path)
            except ValueError as exc:
                messagebox.showerror('Không đọc được ảnh', str(exc))

    def render_image(self, image):
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
        pil = Image.fromarray(rgba)
        background = Image.new('RGBA', pil.size, 'white')
        background.alpha_composite(pil)
        display = background.convert('RGB')
        display.thumbnail((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
        self.display_width, self.display_height = display.size
        self.offset_x = (self.WIDTH-self.display_width)//2
        self.offset_y = (self.HEIGHT-self.display_height)//2
        self.display_image = ImageTk.PhotoImage(display)
        self.canvas.delete('all')
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.display_image, anchor='nw')
        self.selection_id = None

    def show_image(self):
        self.render_image(self.image)
        self.reset_selection()

    def reset_selection(self):
        if self.image is not None:
            self.render_image(self.image)
        self.roi = None
        self.drag_start = None
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None
        self.clear_result()
        if self.image is not None:
            self.status.set('Đang chọn toàn ảnh. Bấm Nhận dạng để tự tìm các chữ số; có thể kéo chuột chọn một vùng giấy.')

    def rotate(self, clockwise=True):
        if self.image is not None:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.show_image()

    def point(self, event):
        return (max(self.offset_x, min(event.x, self.offset_x+self.display_width)),
                max(self.offset_y, min(event.y, self.offset_y+self.display_height)))

    def start_selection(self, event):
        if self.image is None:
            return
        self.reset_selection()
        self.drag_start = self.point(event)
        self.selection_id = self.canvas.create_rectangle(*self.drag_start, *self.drag_start, outline='#2563eb', width=2)

    def drag_selection(self, event):
        if self.drag_start is not None:
            self.canvas.coords(self.selection_id, *self.drag_start, *self.point(event))

    def end_selection(self, event):
        if self.drag_start is None:
            return
        self.drag_selection(event)
        x0, y0 = self.drag_start
        x1, y1 = self.point(event)
        self.drag_start = None
        if abs(x1-x0) < 5 or abs(y1-y0) < 5:
            self.reset_selection()
            return
        height, width = self.image.shape[:2]
        left = int((min(x0,x1)-self.offset_x)*width/self.display_width)
        right = min(width, int((max(x0,x1)-self.offset_x)*width/self.display_width)+1)
        top = int((min(y0,y1)-self.offset_y)*height/self.display_height)
        bottom = min(height, int((max(y0,y1)-self.offset_y)*height/self.display_height)+1)
        self.roi = (left, top, right, bottom)
        self.status.set('Đã chọn vùng giấy. Bấm Nhận dạng để tìm các chữ số trong vùng này.')

    def recognize(self):
        self.clear_result()
        if self.image is None:
            messagebox.showinfo('Chọn ảnh', 'Hãy mở ảnh chụp chữ số trên giấy trước.')
            return
        self.render_image(self.image)
        image = self.image
        if self.roi:
            left, top, right, bottom = self.roi
            image = image[top:bottom, left:right]
        try:
            predictor = self.predictor
            if self.method.get() != 'Mặc định':
                method={'Dãy số rời':'numeric-rows','EMNIST chữ/số':'emnist','YOLO EMNIST':'yolo-emnist','YOLO':'yolo','CNN':'cnn'}[self.method.get()]
                if method not in self.method_models:
                    from .yolo_predict import create_predictor
                    self.method_models[method]=create_predictor(method)
                predictor=self.method_models[method]
            if self.blue_ink.get():
                if self.ink_predictor is None:
                    from .ink_predict import InkPredictor
                    self.ink_predictor = InkPredictor()
                predictor = self.ink_predictor
            result, annotated, thumbnails = recognize_digits(image, predictor)
            self.last_result, self.annotated = result, annotated
            if self.roi:
                display = to_bgr(self.image)
                display[top:bottom, left:right] = cv2.resize(annotated, (right-left, bottom-top))
                self.render_image(display)
            else:
                self.render_image(annotated)
            if not thumbnails:
                self.result.config(text='Không có số')
                self.status.set(result.get('message','Không có chữ số đủ điểm nhận dạng.'))
                return
            montage = Image.new('L', (140, 28*((min(len(thumbnails), 20)+4)//5)))
            for i, thumb in enumerate(thumbnails[:20]):
                montage.paste(Image.fromarray(thumb), ((i%5)*28, (i//5)*28))
            self.preview_image = ImageTk.PhotoImage(montage.resize((210, montage.height*3//2), Image.Resampling.NEAREST))
            self.preview.config(image=self.preview_image, text='', width=210, height=montage.height*3//2)
            self.result.config(text=result['text'])
            if len(result['detections']) == 1 and result['detections'][0].get('probabilities') is not None:
                self.scores.config(text='\n'.join(f'{i}: {score:6.1%}' for i, score in enumerate(result['detections'][0]['probabilities'])))
            else:
                self.scores.config(text='\n'.join(f"{i+1}. {d['digit']}  {d['score']:.1%}" for i, d in enumerate(result['detections'][:10])))
            self.status.set(f"Tìm thấy {len(thumbnails)} vùng chữ số. Khung xanh và nhãn vàng là dự đoán của AI; hãy đối chiếu ảnh gốc. Lưu kết quả để lấy ảnh đầy đủ và JSON.")
        except (ValueError,FileNotFoundError) as exc:
            self.status.set(str(exc))
            messagebox.showinfo('Kiểm tra vùng ảnh', str(exc))

    def export_result(self):
        if self.last_result is None:
            messagebox.showinfo('Lưu kết quả', 'Hãy nhận dạng ảnh trước.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('Ảnh PNG', '*.png')], initialfile='ket_qua_nhan_dang.png')
        if path:
            try:
                save_result(path, self.last_result, self.annotated)
                self.status.set('Đã lưu ảnh có khung và file JSON chứa nhãn, điểm mô hình, vị trí từng chữ số.')
            except OSError as exc:
                messagebox.showerror('Không lưu được', str(exc))


def main():
    parser = argparse.ArgumentParser(description='Nhận dạng ảnh chữ số viết trên giấy')
    parser.add_argument('--model')
    parser.add_argument('--backend', choices=['auto', 'yolo', 'cnn', 'emnist', 'yolo-emnist', 'numeric-rows'], default='auto')
    parser.add_argument('--image', help='Mở và nhận dạng ảnh này khi khởi động')
    parser.add_argument('--rotation',type=int,choices=[0,90,180,270],default=0,help='Xoay ảnh theo chiều kim đồng hồ trước khi nhận dạng')
    args = parser.parse_args()
    try:
        from .yolo_predict import create_predictor
        predictor = create_predictor(args.backend, args.model)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')
    root = tk.Tk()
    app = PhotoApp(root, predictor)
    if args.image:
        try:
            app.load_path(args.image)
            for _ in range(args.rotation//90):app.rotate()
            root.after(200, app.recognize)
        except ValueError as exc:
            messagebox.showerror('Không mở được ảnh', str(exc))
    root.mainloop()


if __name__ == '__main__':
    main()
