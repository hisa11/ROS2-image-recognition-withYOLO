import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import numpy as np
import torch

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.publisher_ = self.create_publisher(String, 'detected_objects', 10)
        self.model = YOLO('/home/hisa/強化学習/best.pt')

        # GPUが使用可能ならばGPUを使用する
        if torch.cuda.is_available():
            self.model.to('cuda')
        else:
            self.get_logger().info('CUDA is not available. Using CPU.')

        self.cap = cv2.VideoCapture(0)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 解像度設定
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 解像度設定
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # フレームレート設定

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30FPS に設定

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return

        frame = cv2.resize(frame, (640, 480))  # フレームのサイズをリサイズ

        # 入力画像を正規化
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)

        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()

        # 物体検出
        results = self.model.predict(source=frame_tensor)

        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf[0].cpu().numpy()
            cls = det.cls[0].cpu().numpy()

            class_name = self.model.names[int(cls)]
            if class_name in ['red', 'blue']:
                message = f'{class_name.capitalize()} detected at: x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}'
                self.get_logger().info(message)

                # トピックに座標を送信
                msg = String()
                msg.data = message
                self.publisher_.publish(msg)

                label = f'{class_name} {conf:.2f}'
                color = (0, 0, 255) if class_name == 'red' else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('YOLOv8 Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

