from tkinter import filedialog
import math
from model import *
from UI import *
import cv2
bulid_model()
img_path = 'noimage.png'
prev_img ='noimage.png'
inputDialog = None
fake = 0
real = 0
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def verify():
    print(img_path)
    global fake, real, prev_img,inputDialog
    if inputDialog != None:
        inputDialog.send()
    if img_path == 'noimage.png':
        return
    fake, real = detect(img_path)
    prev_img = img_path
    if fake > real:
        fake = truncate(fake, 2)
        real = truncate(100 - fake, 2)

    else:
        real = truncate(real, 2)
        fake = truncate(100 - real, 2)



    inputDialog = MyDialog(root,[fake,real])
    root.wait_window(inputDialog.top)



def OpenDialog():
    global my_image
    global img_path
    img_path = filedialog.askopenfilename(initialdir = "/", title="Select a jgp File", filetypes=(("jpg files", "*.jpg"),("all files", "*.jpg")))
    print(img_path)
    my_image= ImageTk.PhotoImage(Image.open(img_path).resize((350,350),Image.ANTIALIAS))
    my_image_lable = Label(image = my_image)
    my_image_lable.place(relx=0.5, rely=0.2, width =350, height=350, anchor='n')

def camera():
    global img_path
    global my_image


    video = cv2.VideoCapture(0)
    # 2. Variable
    a = 0
    # 3. While loop
    while True:
        a = a + 1
        # 4.Create a frame object
        check, frame = video.read()
        # Converting to grayscale
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 5.show the frame!
        cv2.imshow("Capturing", frame)
        # plt.imshow(frame)
        # plt.show()
        # 6.for playing
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    # 7. image saving
    name = "cam_images/img"+ ".jpg"
    showPic = cv2.imwrite(name, frame)
    img_path = name
    print(img_path)


    # 8. shutdown the camera
    video.release()

    cv2.waitKey(1)
    my_image = ImageTk.PhotoImage(Image.open(img_path).resize((350, 350), Image.ANTIALIAS))
    my_image_lable = Label(image=my_image)
    my_image_lable.place(relx=0.5, rely=0.2, width=350, height=350, anchor='n')


root = Tk()
root.title('Facial Forgery Detector')
root.geometry("600x600")
root.configure(background=bg_color)



photo = PhotoImage(file="upload.png")
photo2 = PhotoImage(file="scan.png")
photo3 = PhotoImage(file="icons8-camera-48.png")

title = Label(root,text='Facial Forgery Detector', bg=bg_color, fg="#4a566e", font=Font(family=font_family, size=24))
title.place(relx=0.5, rely=0.05, anchor='n')

my_image= ImageTk.PhotoImage(Image.open(img_path).resize((350,350),Image.ANTIALIAS))
my_image_lable = Label(image = my_image, bg=bg_color)
my_image_lable.place(relx=0.5, rely=0.15, width =350, height=350, anchor='n')

upload = Button(root, bg=bg_color,image = photo, border=0, activebackground = bg_color, command = OpenDialog)
upload.place(relx=0.35, rely=0.8, width =64, height=64, anchor='n')

cam = Button(root, bg=bg_color,image = photo3, border=0, activebackground = bg_color, command = camera)
cam.place(relx=0.5, rely=0.8, width =64, height=64, anchor='n')

verify = Button(root, bg=bg_color, image =photo2, border=0, activebackground = bg_color, command = verify)
verify.place(relx=0.65, rely=0.8, width =64, height=64, anchor='n')

upload_label = Label(root, text="Upload", bg=bg_color, fg='#4a566e',font=Font(family=font_family, size=14))
upload_label.place(relx=0.34, rely=0.9, width=64, height=30, anchor='n')

cam_label = Label(root, text="Camera", bg=bg_color, fg='#4a566e',font=Font(family=font_family, size=14))
cam_label.place(relx=0.5, rely=0.9, width=64, height=30, anchor='n')

verify_label = Label(root, text="Verify", bg=bg_color, fg='#4a566e',font=Font(family=font_family, size=14))
verify_label.place(relx=0.65, rely=0.9, width=64, height=30, anchor='n')




#root.mainloop()

images=['salma','essa','fouad']
labels=[1,2,3]

input_queue = tf.train.slice_input_producer([images, labels], shuffle=False)
print(input_queue[0])