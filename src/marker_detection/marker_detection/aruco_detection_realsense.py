import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ArucoDetectorNode_Realsense(Node):
    def __init__(self):
        super().__init__('aruco_detector_node_realsense')
        self.bridge = CvBridge()

        # 카메라 파라미터 저장
        self.camera_matrix = None # 카메라 행렬
        self.dist_coeffs = None # 렌즈 왜곡 계수 d

        # 아루코 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # 마커 사전 선택 (4x4, 50개)
        self.aruco_params = cv2.aruco.DetectorParameters_create() # 마커 검출 세부 설정 (기본값)
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.marker_length = 0.06 # 마커 한 변 길이 (m)

        # Subscriber
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Publisher
        self.image_pub = self.create_publisher(Image, '/aruco_detector/image', 10)
        self.pose_pub = self.create_publisher(Pose, '/aruco_detector/pose', 10)

        self.get_logger().info('ArucoDetectorNode start ~~ (Realsense)')


    # 카메라 정보 받기
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info('CameraInfo OK ~~')


    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        # ROS Image -> OpenCV 이미지
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 아루코 마커 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0: # 마커 검출에 성공했다면?
            # 마커 3차원 위치 추정
            result = self.my_estimatePoseSingleMarkers(corners, ids)
            if result is None:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
                return
            rvec, tvec, quaternion = result

            # 마커 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # 좌표축 그리기 (x: 빨강, y: 초록, z: 파랑)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

            tvec = tvec.flatten()

            # 결과 Publish
            pose_msg = Pose()
            pose_msg.position.x = tvec[0]
            pose_msg.position.y = tvec[1]
            pose_msg.position.z = tvec[2]
            pose_msg.orientation.x = quaternion[0]
            pose_msg.orientation.y = quaternion[1]
            pose_msg.orientation.z = quaternion[2]
            pose_msg.orientation.w = quaternion[3]
            self.pose_pub.publish(pose_msg)

            self.get_logger().info(f'{pose_msg.position}, {pose_msg.orientation}', throttle_duration_sec=1.0)

        # OpenCV -> ROS Image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        # 결과 이미지 Publish
        self.image_pub.publish(out_msg)


    def my_estimatePoseSingleMarkers(self, corners, ids):
        # 마커 좌표
        object_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],   # 좌상단 좌표
            [self.marker_length / 2, self.marker_length / 2, 0],    # 우상단 좌표
            [self.marker_length / 2, -self.marker_length / 2, 0],   # 우하단 좌표
            [-self.marker_length / 2, -self.marker_length / 2, 0]   # 좌하단 좌표
        ], dtype=np.float32)

        if ids is not None:
            for i in range(len(ids)):
                corner = corners[i][0]

                # solvePnP를 사용하여 3차원 위치 추정
                success, rvec, tvec = cv2.solvePnP(object_points, corner, self.camera_matrix , self.dist_coeffs,
                                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)

                R = Rotation.from_rotvec(rvec.flatten())
                quaternion = R.as_quat() # xyzw

                if success:
                    return rvec, tvec, quaternion

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode_Realsense()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()