import os 
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng
from moviepy.editor import VideoFileClip
from scipy.spatial import distance

np.set_printoptions(threshold=sys.maxsize)

''' Useful Functions '''
# - Start - #

def weighted_img(img, initial_img, α=0.5, β=1, γ=1):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def nothing(x):
    pass

def display_images(images, c=4, r=4):
    w = 20
    h = 20
    fig = plt.figure(figsize=(100, 100))
    columns = c
    rows = r

    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def call_with_trackbars(parameters, variables, callback):
    cv2.namedWindow('Variables', cv2.WINDOW_NORMAL)

    for variable in variables:
        cv2.createTrackbar(variable['name'], 'Variables', variable['default'], variable['max'], nothing)

    cv2.resizeWindow('Variables', 500, 200)
    cv2.moveWindow('Variables', 0, 500)  # Doesn't work on macOS at the moment due to a bug in OpenCV (https://github.com/opencv/opencv/issues/16343)

    values = {}
    last_values = None
    while (1):
        # Break when the "Escape" key is pressed.
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        for variable in variables:
            values[variable['name']] = cv2.getTrackbarPos(variable['name'], 'Variables')

        if (values == last_values):
            continue
        last_values = values.copy()

        print(values)
        callback({**parameters, **values})

# - End - #

# - Start - #
def pipeline(img):

    h, w = img.shape[:2]
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    s = hls[:,:,2]
    
    mask = cv2.inRange(s, 107, 255)
    mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_list = []
    for i in contours:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.01 * perimeter, True)
        bound_box = cv2.boundingRect(approx) # returns [x,y,w,h]
        box_area = bound_box[2] * bound_box[3]
        #print("box:", box_area)
        if box_area > 20000:
            boxes_list.append(bound_box)
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            # cv2.rectangle(img, bound_box, color=color, thickness=2)

    
    if len(boxes_list) == 2:
        box1 = boxes_list[0]
        box2 = boxes_list[1]

        center1 = [box1[0] + box1[2]//2, box1[1] + box1[3]//2]
        center2 = [box2[0] + box2[2]//2, box2[1] + box2[3]//2]

        dst = int(distance.euclidean(center1, center2))
        dst_text = "Distance: {} px".format(dst)
        text_x = int(w*0.05)
        text_y = int(h*0.05)
        cv2.line(img, tuple(center1), tuple(center2), color=(220,0,0), thickness=2)
        cv2.putText(img, dst_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(255,255,255), thickness=3)
        cv2.putText(img, dst_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(220,0,0), thickness=2)

    return img

# - End - #

def main():

    os.chdir(".")   
    if not os.path.isdir("input_videos"):
        print("input_videos folder not found.")
        return
    
    os.makedirs(".\\output_videos\\", exist_ok=True)
    video_folder_dir = sys.path[0] + "\\input_videos\\"

    videos = []
    images = []
    for video_file in os.listdir(os.fsencode(video_folder_dir)):
        filename = os.fsdecode(video_file)
        if filename.endswith(".mp4"):
            input_video = VideoFileClip(video_folder_dir+filename)
            #first_frame = input_video.get_frame(0)
            #images.append(first_frame)
            output_video = input_video.fl_image(pipeline)
            output_video_name = "output_videos\\{}_processed.mp4".format(filename[:-4])
            output_video.write_videofile(output_video_name)
            

    ''' Trackbar parameters'''
    variables = [
        {
            'name': 'tLow',
            'default': 0,
            'max': 255
        },
    ]

    # launch trackbars callback function
    #cv2.destroyAllWindows()
    #call_with_trackbars({ 'img': img }, variables, prototypeAlg)
    
    # End main()

if __name__ == '__main__':
    main()
    