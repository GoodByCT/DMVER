import os
import cv2

_PRE_PATH = "E:\\DSVER_NEW\\IEMOCAP\\"

def analyze(line, mode):
    line = line.split(' , ')
    label = line[1]
    try:
        path = line[0].split('/')
        videoPath = path[-3] + "\\" + path[-2] + "\\" + path[-1]
    except IndexError:
        path = line[0].split('\\')
        videoPath = path[-3] + "\\" + path[-2] + "\\" + path[-1]
    savePath = mode + "\\" + videoPath

    return videoPath, savePath, label

def copyWav(path, mode):
    path = path.split('\\')
    wavPath = "WAV\\" + path[1] + "\\" + path[2].split('.')[0] + ".wav "
    savePath = mode + "\\" + wavPath
    os.system("copy "+_PRE_PATH+wavPath+_PRE_PATH+savePath)
    return savePath


def resizeVideo(videoPath, savePath):
    video = cv2.VideoCapture(_PRE_PATH+videoPath)

    fps = video.get(cv2.CAP_PROP_FPS)
    size = (224, 224)

    writer = cv2.VideoWriter(_PRE_PATH+savePath, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), size)

    i = 0
    while True:
        success, frame = video.read()
        if success:
            i += 1
            if (i >= 1 and i <= 8000):
                frame = cv2.resize(frame, size)
                writer.write(frame)

            if (i > 8000):
                break
        else:
            break


neu = open("../IEMOCAP/VIDEO/neu.txt", 'r')
sad = open("../IEMOCAP/VIDEO/sad.txt", 'r')
ang = open("../IEMOCAP/VIDEO/ang.txt", 'r')
hap = open("../IEMOCAP/VIDEO/hap.txt", 'r')

train = open("../IEMOCAP/train_4/train_video.txt", 'a')
val = open("../IEMOCAP/val_4/val_video.txt", 'a')
test = open("../IEMOCAP/test_4/test_video.txt", 'a')

train_w = open("../IEMOCAP/train_4/train_wav.txt", 'a')
val_w = open("../IEMOCAP/val_4/val_wav.txt", 'a')
test_w = open("../IEMOCAP/test_4/test_wav.txt", 'a')

neu_l = neu.readline()
if neu_l != '':
    neu_f = 0
else:
    neu_f = 1
sad_l = sad.readline()
if sad_l != '':
    sad_f = 0
else:
    sad_f = 1
ang_l = ang.readline()
if ang_l != '':
    ang_f = 0
else:
    ang_f = 1
hap_l = hap.readline()
if hap_l != '':
    hap_f = 0
else:
    hap_f = 1

all = neu_f + sad_f + ang_f + hap_f

con = 0
while all < 4:
    if con % 5 == 3:
        writer = test
        writer_w = test_w
        mode = "test_4"
    elif con % 5 == 4:
        writer = val
        writer_w = val_w
        mode = "val_4"
    else:
        writer = train
        writer_w = train_w
        mode = "train_4"

    if neu_f == 0:
        video, save, label = analyze(neu_l, mode)
        resizeVideo(video, save)
        writer.write(save+" , "+label)
        wavSave = copyWav(video, mode)
        writer_w.write(wavSave+" , "+label)
    if sad_f == 0:
        video, save, label = analyze(sad_l, mode)
        resizeVideo(video, save)
        writer.write(save+" , "+label)
        wavSave = copyWav(video, mode)
        writer_w.write(wavSave+" , "+label)
    if ang_f == 0:
        video, save, label = analyze(ang_l, mode)
        resizeVideo(video, save)
        writer.write(save+" , "+label)
        wavSave = copyWav(video, mode)
        writer_w.write(wavSave+" , "+label)
    if hap_f == 0:
        video, save, label = analyze(hap_l, mode)
        resizeVideo(video, save)
        writer.write(save+" , "+label)
        wavSave = copyWav(video, mode)
        writer_w.write(wavSave+" , "+label)

    neu_l = neu.readline()
    if neu_l != '':
        neu_f = 0
    else:
        neu_f = 1
    sad_l = sad.readline()
    if sad_l != '':
        sad_f = 0
    else:
        sad_f = 1
    ang_l = ang.readline()
    if ang_l != '':
        ang_f = 0
    else:
        ang_f = 1
    hap_l = hap.readline()
    if hap_l != '':
        hap_f = 0
    else:
        hap_f = 1

    all = neu_f + sad_f + ang_f + hap_f
    con += 1

neu.close()
sad.close()
ang.close()
hap.close()
train.close()
val.close()
test.close()
