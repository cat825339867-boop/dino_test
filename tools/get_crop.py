import argparse
import ast
import base64
import io
import math
import os
import time
from typing import List, Tuple

from openai import OpenAI
from PIL import Image, ImageColor, ImageDraw, ImageFont


# 直接沿用参考项目里的 Key 和兼容接口配置
DEFAULT_API_KEY = "sk-04e41645f3714adb9e4a524ff845ce5b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_ID = "qwen3-vl-plus"
DEFAULT_SYS_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = """
识别图片中的金属水槽区域。
请只返回 JSON 数组，数组中每个元素包含：
- "label": 中文名称
- "bbox_2d": [x1, y1, x2, y2]

要求：
1. 坐标必须基于原始输入图片尺寸
2. 坐标归一化到 0~1000
3. x1 < x2, y1 < y2
4. 只返回 JSON，不要解释

返回样例:
[
    {
        "label": "金属水槽",
        "bbox_2d": [32, 18, 128, 55]
    }
]
"""

ADDITIONAL_COLORS = [name for name, _ in ImageColor.colormap.items()]
DEFAULT_MIN_PIXELS = 64 * 32 * 32
DEFAULT_MAX_PIXELS = 2560 * 32 * 32


def smart_resize_qwen2_5_vl(
    img: Image.Image,
    min_pixels: int = 32 * 32 * 4,
    max_pixels: int = 2560 * 32 * 32,
) -> Tuple[int, int]:
    """
    参考原项目的缩放策略，保证输入图片像素规模适配模型。
    返回值顺序保持与参考实现一致：先高后宽。
    """
    width, height = img.size
    h_bar = round(height / 32) * 32
    w_bar = round(width / 32) * 32

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / 32) * 32
        w_bar = math.floor(width / beta / 32) * 32
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / 32) * 32
        w_bar = math.ceil(width * beta / 32) * 32

    return h_bar, w_bar


def parse_json(json_output: str) -> str:
    """移除模型返回中的 Markdown 代码块包装。"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def _parse_bbox_response(bbox_response) -> List[dict]:
    """
    兼容模型返回 JSON 字符串或列表两种情况。
    """
    if isinstance(bbox_response, str):
        bbox_response = parse_json(bbox_response)
        try:
            bbox_list = ast.literal_eval(bbox_response)
        except Exception:
            end_idx = bbox_response.rfind('"}') + len('"}')
            truncated_text = bbox_response[:end_idx] + "]"
            bbox_list = ast.literal_eval(truncated_text)
    else:
        bbox_list = bbox_response

    if not isinstance(bbox_list, list):
        bbox_list = [bbox_list]

    return bbox_list


def _map_bbox_to_image_coords(
    bbox_2d,
    image_size: Tuple[int, int],
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int, int, int]:
    """
    参考原项目逻辑，将模型返回的 0-1000 坐标映射回原图坐标。
    """
    width, height = image_size
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    def _clip_coords(cx1, cy1, cx2, cy2):
        if cx1 > cx2:
            cx1, cx2 = cx2, cx1
        if cy1 > cy2:
            cy1, cy2 = cy2, cy1
        cx1 = max(0, min(int(cx1), width - 1))
        cy1 = max(0, min(int(cy1), height - 1))
        cx2 = max(0, min(int(cx2), width - 1))
        cy2 = max(0, min(int(cy2), height - 1))
        return cx1, cy1, cx2, cy2

    # 直接照搬参考实现，优先按 0-1000 归一化 xyxy 解释。
    xyxy_mapped = _clip_coords(
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )
    if xyxy_mapped[2] > xyxy_mapped[0] and xyxy_mapped[3] > xyxy_mapped[1]:
        print(f"检测到 0-1000 归一化 xyxy 坐标，映射结果: {bbox_2d} -> {list(xyxy_mapped)}")
        return xyxy_mapped

    # 兜底按 xywh 解释一次，尽量兼容模型异常输出。
    xywh_mapped = _clip_coords(
        x1 / 1000 * width,
        y1 / 1000 * height,
        (x1 + x2) / 1000 * width,
        (y1 + y2) / 1000 * height,
    )
    if xywh_mapped[2] > xywh_mapped[0] and xywh_mapped[3] > xywh_mapped[1]:
        return xywh_mapped

    raise ValueError(f"无法将 bbox 映射回原图坐标: {bbox_2d}")


def _ensure_min_pixels_for_qwen35_plus(image_bytes: bytes, model_id: str, min_pixels: int) -> bytes:
    """
    参考项目里的最小像素兜底逻辑。
    """
    if model_id not in {"qwen3.5-plus", "qwen3.5-flash"}:
        return image_bytes

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        current_pixels = width * height

        if current_pixels >= min_pixels:
            return image_bytes

        target_height, target_width = smart_resize_qwen2_5_vl(
            rgb_img,
            min_pixels=min_pixels,
        )
        target_width = max(32, int(target_width))
        target_height = max(32, int(target_height))

        resized_img = rgb_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        resized_img.save(output, format="JPEG", quality=95)

        print(
            f"检测到 {model_id} 输入图片像素不足，已自动放大: "
            f"{width}x{height} -> {target_width}x{target_height}"
        )
        return output.getvalue()


def detect_roi(
    image_path: str,
    prompt: str,
    api_key: str = DEFAULT_API_KEY,
    sys_prompt: str = DEFAULT_SYS_PROMPT,
    model_id: str = DEFAULT_MODEL_ID,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 2560 * 32 * 32,
) -> str:
    """
    调用大模型识别 ROI 区域。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    client = OpenAI(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    processed_image = _ensure_min_pixels_for_qwen35_plus(image_bytes, model_id, min_pixels)
    image_base64 = base64.b64encode(processed_image).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    start_time = time.time()
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    elapsed_time = time.time() - start_time

    content = completion.choices[0].message.content
    print(f"请求耗时: {elapsed_time:.2f} 秒")
    print(f"模型原始返回: {content}")
    return content


def convert_bbox_to_xyxy(
    bbox_response,
    image_size: Tuple[int, int],
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 2560 * 32 * 32,
) -> List[Tuple[int, int, int, int, str]]:
    """
    将模型返回统一转换为原图坐标系下的 xyxy。
    """
    bbox_list = _parse_bbox_response(bbox_response)
    result = []

    for item in bbox_list:
        if "bbox_2d" not in item:
            print(f"警告: 跳过缺少 bbox_2d 的结果: {item}")
            continue

        x1, y1, x2, y2 = _map_bbox_to_image_coords(
            item["bbox_2d"],
            image_size=image_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if x2 <= x1 or y2 <= y1:
            print(f"警告: 跳过无效框: {item}")
            continue

        label = item.get("label", f"ROI-{len(result) + 1}")
        result.append((x1, y1, x2, y2, label))

    return result


def convert_bbox_to_xywh(
    bbox_response,
    image_size: Tuple[int, int] = None,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 2560 * 32 * 32,
) -> List[List[int]]:
    """
    直接对齐参考项目，将模型返回转换为 xywh 结构。
    """
    bbox_list = _parse_bbox_response(bbox_response)
    result = []

    for item in bbox_list:
        if "bbox_2d" not in item:
            continue

        if image_size is not None:
            x1, y1, x2, y2 = _map_bbox_to_image_coords(
                item["bbox_2d"],
                image_size=image_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            x1, y1, x2, y2 = item["bbox_2d"]

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            result.append([x, y, w, h])

    return result


def draw_roi_on_image(
    image_path: str,
    bounding_boxes,
    output_path: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 2560 * 32 * 32,
) -> List[Tuple[int, int, int, int, str]]:
    """
    在原图上画出所有识别到的框，并保存结果图。
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray",
        "beige", "turquoise", "cyan", "magenta", "lime", "navy", "maroon", "teal",
        "olive", "coral", "lavender", "violet", "gold", "silver",
    ] + ADDITIONAL_COLORS

    try:
        font = ImageFont.truetype("simhei.ttf", size=24)
    except OSError:
        font = ImageFont.load_default()

    boxes = convert_bbox_to_xyxy(
        bounding_boxes,
        image_size=(width, height),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if not boxes:
        raise ValueError("模型未返回有效框，无法生成画框图")

    for i, (x1, y1, x2, y2, label) in enumerate(boxes):
        color = colors[i % len(colors)]
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=4)
        draw.text((x1 + 8, max(y1 - 30, 0)), label, fill=color, font=font)

    img.save(output_path)
    print(f"画框图已保存: {output_path}")
    return boxes


def select_roi(
    image_path: str,
    prompt: str,
    output_path: str = None,
    api_key: str = DEFAULT_API_KEY,
    sys_prompt: str = DEFAULT_SYS_PROMPT,
    model_id: str = DEFAULT_MODEL_ID,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    return_xywh: bool = True,
):
    """
    直接复用参考项目的完整 ROI 流程：
    1. 调模型
    2. 画框
    3. 返回 xywh 或原始响应
    """
    response = detect_roi(
        image_path=image_path,
        prompt=prompt,
        api_key=api_key,
        sys_prompt=sys_prompt,
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    print(response, "返回的数值")

    img = draw_roi_on_image(
        image_path=image_path,
        bounding_boxes=response,
        output_path=output_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if return_xywh:
        with Image.open(image_path) as source_img:
            image_size = source_img.size
        bbox_list = convert_bbox_to_xywh(
            response,
            image_size=image_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return img, bbox_list

    return img, response


def select_roi_with_response(
    image_path: str,
    prompt: str,
    output_path: str = None,
    api_key: str = DEFAULT_API_KEY,
    sys_prompt: str = DEFAULT_SYS_PROMPT,
    model_id: str = DEFAULT_MODEL_ID,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
):
    """
    在保留参考项目流程的基础上，同时把原始模型返回带出来，
    方便后续裁剪和排查日志，避免重复请求模型。
    """
    response = detect_roi(
        image_path=image_path,
        prompt=prompt,
        api_key=api_key,
        sys_prompt=sys_prompt,
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    print(response, "返回的数值")

    draw_roi_on_image(
        image_path=image_path,
        bounding_boxes=response,
        output_path=output_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    with Image.open(image_path) as source_img:
        image_size = source_img.size

    bbox_xywh = convert_bbox_to_xywh(
        response,
        image_size=image_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    bbox_xyxy = convert_bbox_to_xyxy(
        response,
        image_size=image_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return response, bbox_xywh, bbox_xyxy


def crop_largest_roi(
    image_path: str,
    boxes: List[Tuple[int, int, int, int, str]],
    output_path: str,
) -> Tuple[int, int, int, int, str]:
    """
    单张裁剪图只保留一个目标，这里默认裁剪面积最大的框。
    """
    if not boxes:
        raise ValueError("没有可用于裁剪的框")

    target_box = max(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
    x1, y1, x2, y2, label = target_box

    with Image.open(image_path).convert("RGB") as img:
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(output_path)

    print(f"裁剪图已保存: {output_path}")
    print(f"本次用于裁剪的框: label={label}, bbox=({x1}, {y1}, {x2}, {y2})")
    return target_box


def crop_largest_roi_from_xywh(
    image_path: str,
    boxes_xywh: List[List[int]],
    output_path: str,
) -> Tuple[int, int, int, int, str]:
    """
    参考项目返回的是 xywh，这里转成 xyxy 后再裁剪。
    """
    if not boxes_xywh:
        raise ValueError("没有可用于裁剪的框")

    xyxy_boxes = []
    for index, item in enumerate(boxes_xywh, start=1):
        x, y, w, h = item
        xyxy_boxes.append((x, y, x + w, y + h, f"ROI-{index}"))

    return crop_largest_roi(
        image_path=image_path,
        boxes=xyxy_boxes,
        output_path=output_path,
    )


def get_crop(
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    boxed_output_path: str = None,
    cropped_output_path: str = None,
    api_key: str = DEFAULT_API_KEY,
    model_id: str = DEFAULT_MODEL_ID,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict:
    """
    对外主流程：
    1. 调用大模型识别框
    2. 生成画框图
    3. 生成裁剪图
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt 不能为空")

    abs_image_path = os.path.abspath(image_path)
    base_dir = os.path.dirname(abs_image_path)
    file_stem, _ = os.path.splitext(os.path.basename(abs_image_path))

    if boxed_output_path is None:
        boxed_output_path = os.path.join(base_dir, f"{file_stem}_boxed.jpg")
    if cropped_output_path is None:
        cropped_output_path = os.path.join(base_dir, f"{file_stem}_crop.jpg")

    response, boxes_xywh, boxes = select_roi_with_response(
        image_path=abs_image_path,
        prompt=prompt,
        output_path=boxed_output_path,
        api_key=api_key,
        sys_prompt=DEFAULT_SYS_PROMPT,
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if not boxes_xywh:
        raise ValueError("模型未返回有效框，无法生成裁剪图")

    selected_box = crop_largest_roi_from_xywh(
        image_path=abs_image_path,
        boxes_xywh=boxes_xywh,
        output_path=cropped_output_path,
    )

    return {
        "boxed_image": boxed_output_path,
        "cropped_image": cropped_output_path,
        "selected_box": {
            "x1": selected_box[0],
            "y1": selected_box[1],
            "x2": selected_box[2],
            "y2": selected_box[3],
            "label": selected_box[4],
        },
        "all_boxes_count": len(boxes_xywh),
        "model_response": response,
    }


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="传入图片和提示词，输出画框图与裁剪图。")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="传给大模型的提示词，默认使用金属水槽识别提示词")
    parser.add_argument("--boxed-output", help="画框图输出路径")
    parser.add_argument("--crop-output", help="裁剪图输出路径")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="模型 API Key，默认沿用参考项目配置")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型名称，默认 qwen3-vl-plus")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="模型输入最小像素，默认与参考项目 ROI 配置一致")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="模型输入最大像素")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    result = get_crop(
        image_path=args.image,
        prompt=args.prompt,
        boxed_output_path=args.boxed_output,
        cropped_output_path=args.crop_output,
        api_key=args.api_key,
        model_id=args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print("处理完成:")
    print(result)
