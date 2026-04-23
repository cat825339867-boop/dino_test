import cv2
import os


def extract_frames(video_path, output_dir, interval_sec=2):
    """
    每隔 interval_sec 秒抽一帧保存为 JPG
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    # 获取视频 FPS（每秒多少帧）
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频 FPS: {fps}")

    # 计算间隔多少帧取一张
    frame_interval = int(fps * interval_sec)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 frame_interval 保存一张
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"保存: {filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ 完成，共保存 {saved_count} 张图片")


if __name__ == "__main__":
    video_path = "/home/ubuntu/yzm_workspace/video_agent/456.mp4"      # 你的视频
    output_dir = "frames_456"         # 输出目录

    extract_frames(video_path, output_dir, interval_sec=10)