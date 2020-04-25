import cv2
import numpy as np
import math


class KalmanFilter(object):
    # 6 elements: [xc, yc, vx, vy, w, h] —> predict(X):n
    # 4 elements: [zxc, zyc, zw, zh] —> measure(H):k
    def __init__(self, video_helper):
        self.dynamParamsSize = 6  # state element count
        self.measureParamsSize = 4  # measure(detect) element count
        self.kalman = cv2.KalmanFilter(dynamParams=self.dynamParamsSize,
                                       measureParams=self.measureParamsSize)
        self.first_run = True
        dT = 1. / video_helper.frame_fps
        # Transiftion Matrix：F->nxn
        self.kalman.transitionMatrix = np.array([[1, 0, dT, 0, 0, 0],  # xc
                                                 [0, 1, 0, dT, 0, 0],  # yc
                                                 [0, 0, 1, 0, 0, 0],   # vx
                                                 [0, 0, 0, 1, 0, 0],   # vy
                                                 [0, 0, 0, 0, 1, 0],   # h
                                                 [0, 0, 0, 0, 0, 1]],  # w
                                                np.float32)

        # error estimate covariance matrix:S->nxn
        self.kalman.errorCovPre = np.eye(6, dtype=np.float32)
        # self.kalman.errorCovPost =

        # external influence (process noise) covariance matrix:Q->nxn
        self.kalman.processNoiseCov = np.array([[0.01, 0, 0, 0, 0, 0],  # xc
                                                [0, 0.01, 0, 0, 0, 0],  # yc
                                                [0, 0, 5.0, 0, 0, 0],  # vx
                                                [0, 0, 0, 5.0, 0, 0],  # vy
                                                [0, 0, 0, 0, 0.01, 0],  # h
                                                [0, 0, 0, 0, 0, 0.01]],   # w
                                               np.float32)
        # Measurement Matrix: H->kxn
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],   # xc
                                                  [0, 1, 0, 0, 0, 0],   # yc
                                                  [0, 0, 0, 0, 1, 0],   # h
                                                  [0, 0, 0, 0, 0, 1]],  # w
                                                 np.float32)

        # Measurement noise covariance Matrix: R->kxk
        self.kalman.measurementNoiseCov = np.array([[0.1, 0, 0, 0],
                                                    [0, 0.1, 0, 0],
                                                    [0, 0, 0.1, 0],
                                                    [0, 0, 0, 0.1]], np.float32)

    def get_predicted_bbox(self):
        """
        x_k+1 = F_k+1 * x_k + B_k * u_k
        S_k+1 = F_k+1 * S_k * F_k+1^T + Q_k+1
        :return: [left, right, top, bottom]
        """
        # predicted_res: [xc, yc, vx, vy, h, w]
        predicted_res = self.kalman.predict().T[0]
        predicted_bbox = self.get_bbox_from_kalman_form(predicted_res)
        return predicted_bbox

    def correct(self, bbox):
        """
        x^hat_k+1 = x_k+1 + K^hat * (z_k+1 - H_k+1 * x_k+1)
        S^hat_k+1 = S_k+1 - K^hat * H_k+1 * S_k+1
        :param bbox:[left, right, top, bottom]
        :return:
        """
        # bbox: l, r, t, b —> 即我们的观测
        w = bbox[1] - bbox[0] + 1       # 2nd, 3rd —> length = 2nd + 3rd = 2 pixels = 3 - 2 + 1
        h = bbox[3] - bbox[2] + 1
        xc = int(bbox[0] + w / 2.)
        yc = int(bbox[2] + h / 2.)
        measurement = np.array([[xc, yc, w, h]], dtype=np.float32).T

        if self.first_run:
            self.kalman.statePre = np.array([measurement[0], measurement[1],
                                             [0], [0],
                                             measurement[2], measurement[3]], dtype=np.float32)
            self.first_run = False
        corrected_res = self.kalman.correct(measurement).T[0]
        corrected_bbox = self.get_bbox_from_kalman_form(corrected_res)
        return corrected_bbox

    def get_bbox_from_kalman_form(self, kalman_form):
        """
        :param kalman_form: [xc, yc, vx, vy, h, w]
        :return: [left, right, top, bottom]
        """
        xc = kalman_form[0]
        yc = kalman_form[1]
        w = kalman_form[4]
        h = kalman_form[5]
        left = math.ceil(xc - w / 2.)
        right = math.ceil(xc + w / 2.)
        top = math.ceil(yc - h / 2.)
        bottom = math.ceil(yc + h / 2.)
        return [left, right, top, bottom]
