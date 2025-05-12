from tkinter import *
import cv2
from PIL import Image, ImageTk

cam = cv2.VideoCapture(0)

window = Tk()

picture = Label(window)
picture.pack()

def open_camera():
    ret, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    photo_image = ImageTk.PhotoImage(image=img)
    picture.photo_image = photo_image
    picture.configure(image=photo_image)
    picture.after(10,open_camera)

button1 = Button(window, text="Open Camera", command=open_camera)
button1.pack()

window.mainloop()