import os
from moviepy.editor import *

_PRE_SESSION = "D:\\Download\\IEMOCAP\\IEMOCAP_full_release\\Session"
_SAVE_PATH = "E:\\DSVER_NEW\\IEMOCAP\\"

for i in range(3):
    emoPath = _PRE_SESSION + str(i+3) + "\\dialog/EmoEvaluation\\"
    vidPath = _PRE_SESSION + str(i+3) + "\\dialog\\avi\\DivX\\"
    wavPath = _PRE_SESSION + str(i+3) + "\\sentences\\wav\\"
    for j in os.listdir(emoPath):
        if os.path.isfile(emoPath+j):
            if j.split('_')[1].split('0')[0] == "impro":
                with open(emoPath+j) as f:
                    video = VideoFileClip(vidPath+j.split('.')[0]+'.avi')
                    line = f.readline()
                    while line != '':
                        if line[0] == '[':
                            datas = line.split('\t')
                            label = datas[2] if datas[2] != "xxx" else "oth"
                            cliPath = datas[1]
                            times = datas[0]
                            sTime = float(times.split('[')[1].split(' - ')[0])
                            eTime = float(times.split(' - ')[1].split(']')[0])
                            clip = video.subclip(sTime, eTime)
                            saveVideo = _SAVE_PATH+"VIDEO\\"+label+"\\"+cliPath+".mp4"
                            oriWav = wavPath + j.split('.')[0] + "\\" + cliPath + ".wav "
                            saveWav = _SAVE_PATH+"\\WAV\\"+label+"\\"+cliPath+".wav"
                            with open(_SAVE_PATH+"\\VIDEO\\"+label+".txt", 'a') as v:
                                v.write(saveVideo+" , "+label+"\n")
                            with open(_SAVE_PATH+"\\WAV\\"+label+".txt", 'a') as w:
                                w.write(saveWav+" , "+label+"\n")
                            clip.write_videofile(saveVideo, audio=False)
                            os.system("copy " + oriWav + saveWav)
                        line = f.readline()
                print("\n\n\nFile process successed: "+emoPath+j+"\n\n\n")
