from tkinter import *
from PIL import Image, ImageTk
from tkinter.font import Font

image_open = "Stars2.png"
bg_color = "#f4f6ff"
fake_image = "fake.png"
real_image = "real.png"

font_family = "Bahnschrift SemiLight"
class MyDialog:
    def __init__(self, parent, res):
        top = self.top = Toplevel(parent)
        self.top.configure(background=bg_color)
        self.top.geometry("225x250")
        self.top.title('Detector')
        text = 'Real'
        if res[0] > res[1]:
            text = 'Fake'
        img_p = str.lower(text) + ".png"
        print(img_p)
        self.detect = Label(self.top, text=text, bg=bg_color, fg="#4a566e", font=Font(family=font_family, size=24))
        self.detect.place(relx=0.5, rely=0.1, width=80, height=30, anchor='n')
        self.photo = ImageTk.PhotoImage(Image.open(img_p).resize((120, 120), Image.ANTIALIAS))
        self.detect_img = Label(self.top, image=self.photo, bg=bg_color)
        self.detect_img.place(relx=0.5, rely=0.3, width=120, height=120, anchor='n')

        self.real = Label(self.top, text="Real: " + str(res[1]), bg=bg_color, fg="#00c184",
                          font=Font(family=font_family, size=10))
        self.real.place(relx=0.3, rely=0.8, width=100, height=25, anchor='n')

        self.fake = Label(self.top, text="Fake: " + str(res[0]), bg=bg_color, fg="#fc646f",
                          font=Font(family=font_family, size=10))
        self.fake.place(relx=0.7, rely=0.8, width=100, height=25, anchor='n')
        # self.mySubmitButton = Button(top, text='Done!', command=self.send)
        # self.mySubmitButton.pack()

    def send(self):
        self.top.destroy()
