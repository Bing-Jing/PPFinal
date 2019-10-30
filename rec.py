from PIL import ImageGrab
import numpy as np
import cv2
import threading

imgCounter = 0

def screenREC(x_Start, y_Start, x_End, y_End):
    if x_Start > x_End:
        x_Start, x_End = x_End, x_Start
    if y_Start > y_End:
        y_Start, y_End = y_End, y_Start

    im = ImageGrab.grab((x_Start, y_Start, x_End, y_End))
    a, b = im.size
    video = cv2.VideoWriter("test.mp4",cv2.VideoWriter_fourcc('M','J','P','G'), 5, (a, b))#輸出檔案命名為test.mp4,幀率為5
    while True:
        im = ImageGrab.grab((x_Start, y_Start, x_End, y_End))
        imm=cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        video.write(imm)
        cv2.imshow('im', imm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
def snapshot(x_Start, y_Start, x_End, y_End):
    global imgCounter
    if x_Start > x_End:
        x_Start, x_End = x_End, x_Start
    if y_Start > y_End:
        y_Start, y_End = y_End, y_Start
    im = np.array(ImageGrab.grab((x_Start, y_Start, x_End, y_End)),np.uint8)
    cv2.imwrite("shot_{}.jpg".format(imgCounter),im)
    imgCounter += 1
    print("snapshot")

def loopShot(time,x_Start, y_Start, x_End, y_End):
    snapshot(x_Start, y_Start, x_End, y_End)
    global timer
    timer = threading.Timer(time, loopShot, [time,x_Start, y_Start, x_End, y_End])
    timer.start()
if __name__ == "__main__":
    timer = threading.Timer(0.5, loopShot, [0.5,1200,1300,100.5,110.5])
    timer.start()
