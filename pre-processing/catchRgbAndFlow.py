import os
import cv2
import numpy as np

_PRE_PATH = "E:\\DSVER_NEW\\IEMOCAP\\"

_VIDEO_FRAMES = 79


def catch(videoPath):
    video = cv2.VideoCapture(videoPath)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []
    all_flow = []

    if frames < _VIDEO_FRAMES:
        for i in range(_VIDEO_FRAMES):
            video.set(cv2.CAP_PROP_POS_FRAMES, i % frames)
            all_frames.append(video.read()[1])
    else:
        step = int(frames/_VIDEO_FRAMES)
        for i in range(_VIDEO_FRAMES):
            video.set(cv2.CAP_PROP_POS_FRAMES, i*step)
            all_frames.append(video.read()[1])
    # print('complete rgb: ' + videoPath)

    prev = all_frames[0]
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for frame_curr in range(_VIDEO_FRAMES):
        curr = all_frames[frame_curr]
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(prev, curr)
        all_flow.append(flow)
        prev = curr
    # print('complete flow: ' + videoPath)

    return all_frames, all_flow


def compute_TVL1(prev, curr, bound=15):
    """comput the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

if __name__ == "__main__":
    with open("../IEMOCAP/train_4/train_video.txt", 'r') as f:
        with open("../IEMOCAP/train_4/train_rgb.txt", 'a') as w_r:
            with open("../IEMOCAP/train_4/train_flow.txt", 'a') as w_f:
                line = f.readline()
                while line != '':
                    line = line.split(' , ')
                    videoPath = line[0]
                    label = line[1]
                    paths = videoPath.split('\\')
                    rgbPath = paths[0] + '\\RGB\\' + paths[2] + '\\' + paths[3].split('.')[0] + "_rgb.npy"
                    flowPath = paths[0] + '\\Flow\\' + paths[2] + '\\' + paths[3].split('.')[0] + "_flow.npy"
                    rgb, flow = catch(_PRE_PATH + videoPath)
                    np.save(_PRE_PATH+rgbPath, rgb)
                    w_r.write(rgbPath+' , '+label)
                    np.save(_PRE_PATH+flowPath, flow)
                    w_f.write(flowPath+' , '+label)
                    line = f.readline()
