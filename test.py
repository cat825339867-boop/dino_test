...
这个文件用于测试/dino/find-similar-region接口
...

import cv2
import requests

url = "http://127.0.0.1:8003/dino/find-similar-region"

cap = cv2.VideoCapture("/home/ubuntu/yzm_workspace/compare_embedding/test_video2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
ret, frame = cap.read()

if not ret or frame is None:
    cap.release()
    raise RuntimeError("读取视频帧失败")

ok, buffer = cv2.imencode(".jpg", frame)
if not ok:
    cap.release()
    raise RuntimeError("视频帧编码失败")

files = {
    "frame_file": ("frame.jpg", buffer.tobytes(), "image/jpeg"),
}
data = {
    # "small_image_path": "/home/ubuntu/yzm_workspace/compare_embedding/chen.jpg",
    "save_annotated_image": "true",
    # "annotated_output_path": "/home/ubuntu/yzm_workspace/compare_embedding/output/matched_frame.jpg",
}

response = requests.post(url, files=files, data=data)

print(response.status_code)

try:
    result = response.json()
except Exception:
    print(response.text)
    cap.release()
    cv2.destroyAllWindows()
    raise

print(result)

timing = result.get("timing", {})
match_detail = timing.get("match_detail", {})
scale_details = match_detail.get("scale_details", [])

print("\n===== 耗时统计 =====")
print(f"read_frame_elapsed_sec: {timing.get('read_frame_elapsed_sec')}")
print(f"decode_frame_elapsed_sec: {timing.get('decode_frame_elapsed_sec')}")
print(f"load_small_image_elapsed_sec: {timing.get('load_small_image_elapsed_sec')}")
print(f"match_elapsed_sec: {timing.get('match_elapsed_sec')}")
print(f"save_image_elapsed_sec: {timing.get('save_image_elapsed_sec')}")
print(f"total_elapsed_sec: {timing.get('total_elapsed_sec')}")

print("\n===== 匹配阶段明细 =====")
print(f"big_token_elapsed_sec: {match_detail.get('big_token_elapsed_sec')}")
print(f"postprocess_elapsed_sec: {match_detail.get('postprocess_elapsed_sec')}")
print(f"match_total_elapsed_sec: {match_detail.get('total_elapsed_sec')}")

print("\n===== 各尺度耗时 =====")
for item in scale_details:
    print(
        f"scale={item.get('scale')}, "
        f"skipped={item.get('skipped')}, "
        f"small_token={item.get('small_token_elapsed_sec')}, "
        f"match={item.get('match_elapsed_sec')}, "
        f"total={item.get('total_elapsed_sec')}, "
        f"candidates={item.get('candidate_count')}"
    )

cap.release()
cv2.destroyAllWindows()
