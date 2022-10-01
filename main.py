import cv2
import time
import numpy as np
import serial

average_speed_cars = int(input("[INPUT]Please enter the average speed of cars: "))
current_flow_time = 0

num_cars_array = []
cars = []
cap_array = []
road_list = ["Ayeduase_Conti Road", "Obeng_Poku Road"]
traffic_signals = ["C", "A", "B"]

cap_east = cv2.VideoCapture(1)
cap_west = cv2.VideoCapture(0)
cap_south = cv2.VideoCapture(3)
cap_north = cv2.VideoCapture(3)
#
# cap.set(3, 640)
# cap.set(4, 480)

frame_width = 480
frame_height = 240

classFile = 'resources/coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'resources/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

thres = 0.5  # Threshold to detect object
lane_index = 0
other_lane_index = 1

arduino = serial.Serial(port='/dev/cu.usbserial-140', baudrate=9600, timeout=.1)

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


def write_text(img, lane, num_cars=0):
    cv2.putText(img=img, text="Number of cars: " + str(num_cars), org=(20, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.putText(img=img, text="Lane: " + str(lane), org=(20, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0), thickness=1)


def mobile_ssd_detector(img):
    ssd_image = img
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    cars_detected = []
    if len(classIds) != 0:  # or classId < (len(classIds)-1):
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 3:
                cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)

                cv2.putText(ssd_image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                cars_detected.append(box)
    cv2.imshow("Detections", ssd_image)
    return len(cars_detected)


def stack_images(img_array, scale, labels=[]):
    sizeW = img_array[0][0].shape[1]
    sizeH = img_array[0][0].shape[0]
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (sizeW, sizeH), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (sizeW, sizeH), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        hor_con = np.concatenate(img_array)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


# def write_text():


def switch_traffic_signal(lane_index, yellow_signal=False):
    other_lane_index = abs(lane_index - 1)

    lane_of_interest = str(road_list[lane_index])
    other_lane = str(road_list[other_lane_index])

    # if yellow_signal is False:
    write_read(traffic_signals[lane_index])
    # else:
    #     write_read(traffic_signals[2])

    print("[INFO]Turning " + str(other_lane) + " " + str(traffic_signals[0]))
    print("[INFO]Turning " + str(lane_of_interest) + " " + str(traffic_signals[1]))


def flow_time(index_num_cars, lane_of_interest):
    flow_time = ((index_num_cars * 10) / average_speed_cars) + 2  # Calculate flow time
    print("Flow time for " + lane_of_interest + " is calculated to be: " + str(flow_time))

    return flow_time


start_time = time.time()  # Set off Timer
time_elapsed = 0
timer_set_off = False

img_array = []

change_frame = False
change_frame_timer_elapsed = 0
change_frame_timer_time = time.time()


while True:
    # cap = cap_array[lane_index]
    _, east_lane_image = cap_east.read()
    _, west_lane_image = cap_west.read()
    _, north_lane_image = cap_north.read()
    _, south_lane_image = cap_south.read()

    # Resize images and Bind Cameras together
    east_lane_image = cv2.resize(east_lane_image, (frame_width, frame_height))
    west_lane_image = cv2.resize(west_lane_image, (frame_width, frame_height))
    ayeduase_conti_lane = [east_lane_image, west_lane_image]

    north_lane_image = cv2.resize(north_lane_image, (frame_width, frame_height))
    south_lane_image = cv2.resize(south_lane_image, (frame_width, frame_height))
    obeng_poku_lane = [north_lane_image, south_lane_image]

    img_array = [ayeduase_conti_lane, obeng_poku_lane]

    # cars = mobile_ssd_detector(img_array[lane_index][0])
    # cars = mobile_ssd_detector(img_array[lane_index][1]) + cars

    if timer_set_off is False:
        # print("Lane is: " + str(lane_index))
        cars = 0

        for img in img_array[lane_index]:
            # time.sleep(2)
            # print(len(img))
            cars = mobile_ssd_detector(img) + cars
            # cars = mobile_ssd_detector(img_array[lane_index][1]) + cars
        # if change_frame is False:
        #     cars = mobile_ssd_detector(img_array[lane_index][0])
        # else:
        #     cars = mobile_ssd_detector(img_array[lane_index][1]) + cars
        # num_cars_array.append(len(cars))  # Store number of cars

    # Choose lane of interest
    # other_lane_index = abs(lane_index - 1)
    # num_cars = num_cars_array[0]

    if cars > 0 and timer_set_off is False:
        print("Cars are: " + str(cars))
        # for car in cars:
        #     # (x, y, w, h) = car
        #     # cv2.rectangle(image_1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        current_flow_time = flow_time(cars, road_list[lane_index])  # Calculate flow time
        # switch_traffic_signal(str(road_list[other_lane_index]), lane_index)    # Switch traffic signal of
        # other lane
        timer_set_off = True
        start_time = time.time()
        # time_elapsed = 0
        switch_traffic_signal(lane_index)

    # Switch Traffic Signals
    # bolo = abs(current_flow_time) > time_elapsed
    # print("Current flow is: " + str(abs(current_flow_time)))
    # print("Time Elapsed is one: " + str(time_elapsed))
    # print("bolo is: " + str(bolo))
    #
    if timer_set_off is True and current_flow_time > time_elapsed:
        end_time = time.time()
        time_elapsed = round(end_time - start_time)
        if time_elapsed >= (0.7 * current_flow_time):
            # switch_traffic_signal(lane_index, yellow_signal=True)  # Switch traffic signal of primary lane
            print("Switching to yellow")
            # switch_traffic_signal(lane_of_interest, traffic_signals[1])
            # switch_traffic_signal(str(road_list[other_lane_index]), traffic_signals[1])
        print("Time Elapsed is: " + str(time_elapsed))
        # time.sleep(1)
    else:
        timer_set_off = False
        start_time = 0
        num_cars_array = []
        time_elapsed = 0
        car = 0

        if lane_index == 1:
            lane_index = 0
        else:
            lane_index = lane_index + 1

    # for img_list in img_array:
    #     for img in img_list:
    #         write_text(img, road_list[0])
    stackedImages = stack_images(([east_lane_image, west_lane_image],
                                  [north_lane_image, south_lane_image]), 0.6)
    cv2.imshow("Stacked Images", stackedImages)

    # cv2.imshow("Frame", image_1)
    # cv2.imshow("Frame_2", image_2)
    # cv2.imshow("Frame_3", image_3)
    cv2.waitKey(1)
