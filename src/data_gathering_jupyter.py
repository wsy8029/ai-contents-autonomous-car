import cv2
import os

camera = cv2.VideoCapture(0)
camera.set(3,320)
camera.set(4,240)


print('Step 1. Data Gathering..')

# gathering image data to each labeled folder
def gathering(path):
    # check latest file index
    try: # if dir is not empty, file idx continue
        flist = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in flist]
        latest = max(paths, key=os.path.getctime) # latest file
        latest = os.path.basename(latest)
        latest = int(os.path.splitext(latest)[0]) # latest file idx num
        idx = latest +1
        
    except: # if dir is empty, first file idx is 1
        idx = 1

    while camera.isOpened():
        _,frame = camera.read()
        cv2.imshow("frame",frame) # streaming camera
        
        c = cv2.waitKey(1)
        if c == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break
        
        elif c == ord('c'):
            
            imgpath = path + '/' + str(idx) + '.jpg'
            print(imgpath)
            cv2.imwrite(imgpath,frame)
            print('image saved')
            idx += 1

        
#check and make directory labeled
def check_dir(path,label):
    path = os.path.join(path,label)
    print("Image Data Path : " + path)
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    return path

while 1:
    path = '../data/'
    label = input('type label name : ')
    path = check_dir(path,label)
    gathering(path)