from __future__ import print_function
import time
import os
from config import Configs
from models.faceboxes.inference import build_net, get_face_boxes
from utils.video_helper import VideoHelper
from tracker.multiple_object_controller import MultipleObjectController
from utils.get_ID_emotion import get_face_ID
from utils.visualizer import Visualizer

import warnings
# 忽略警告
warnings.filterwarnings('ignore')

# set gpu card id (0 - 7)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run():
    # step 1: initialization
    # set configures: parameters are set here
    start_pre = time.time()
    configs = Configs()
    # video helper: read in / write out video rames
    video_helper = VideoHelper(configs)
    im_height = round(configs.FRAME_RESIZE * video_helper.frame_height)
    im_width = round(configs.FRAME_RESIZE * video_helper.frame_width)
    print('frame size: ', im_height, im_width)
    """ Detection Part """
    # build net then use 'get_face_boxes' in faceboxes inference
    net, prior_data = build_net((im_height, im_width), device='cpu')

    # object controller: objects are managed in this class
    object_controller = MultipleObjectController(configs, video_helper)
    print("Pre Time: ", (time.time() - start_pre) * 1000, " ms")

    # step 2: main loop
    cur_frame_counter = 0
    detection_loop_counter = 0
    while video_helper.not_finished(cur_frame_counter):
        print("###################################################### frame: ", cur_frame_counter)
        # get frame from video
        # frame is the raw frame, frame_show is used to show the result
        frame, frame_show = video_helper.get_frame()
        # if we detect every frame
        if configs.NUM_JUMP_FRAMES == 0:
            # detect objects in a frame; return us detected face boxes
            # detected results: [{'face':[left, right, top, bottom]}, {'face':[left, right, top, bottom]}, ...]
            start_turn = time.time()
            detects = get_face_boxes(net, prior_data, frame, configs.FRAME_RESIZE, cur_frame_counter)
            time_spend_of_detection = time.time() - start_turn

            object_controller.update(detects, cur_frame_counter, frame)
            get_face_ID(configs, frame, object_controller.instances, cur_frame_counter)
            time_spend_of_tracking = time.time() - start_turn - time_spend_of_detection
            print("Detecting Time: ", time_spend_of_detection * 1000, " ms.")
            print("Tracking Time with detection: ", time_spend_of_tracking * 1000, " ms.")
        else:
            # we ignore to detect some frames
            if detection_loop_counter % configs.NUM_JUMP_FRAMES == 0:
                start_turn_with_detection = time.time()
                # here we need to detect the frame
                detection_loop_counter = 0
                detects = get_face_boxes(net, prior_data, frame, configs.FRAME_RESIZE, cur_frame_counter)
                time_spend_of_detection = time.time() - start_turn_with_detection
                object_controller.update(detects, cur_frame_counter, frame)
                get_face_ID(configs, frame, object_controller.instances, cur_frame_counter)
                time_spend_of_tracking = time.time() - start_turn_with_detection - time_spend_of_detection
                print("Detecting Time: ", time_spend_of_detection * 1000, " ms.")
                print("Tracking Time with detection: ", time_spend_of_tracking * 1000, " ms.")
            else:
                # here we needn't to detect the frame
                start_turn_without_detection = time.time()
                object_controller.update_without_detection(cur_frame_counter, frame)
                time_spend_of_tracking = time.time() - start_turn_without_detection
                print("Tracking Time without detection: ", time_spend_of_tracking * 1000, " ms.")

        # 可视化
        visualizer = Visualizer(configs)
        show_temporal_information = True
        visualizer.drawing_tracking(frame_show,
                                    object_controller.instances,
                                    cur_frame_counter,
                                    show_temporal_information)

        cur_frame_counter += 1
        detection_loop_counter += 1
    video_helper.end()


if __name__ == "__main__":
    run()

