# -*- coding: utf-8 -*-
import argparse
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def generate_inference_images(text, src_font_path, save_dir, char_size=150, canvas_size=256, x_offset=20, y_offset=20):
    """추론용 이미지 생성"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        src_font = ImageFont.truetype(src_font_path, size=char_size)
    except Exception as e:
        raise Exception(f"Error loading font {src_font_path}: {str(e)}")

    print(f"Generating images for {len(text)} characters...")

    for idx, char in enumerate(text):
        # 흰색 배경의 캔버스 생성
        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 문자 그리기
        draw.text((x_offset, y_offset), char, (0, 0, 0), font=src_font)

        # 추론용 이미지 포맷에 맞게 저장 (원본 이미지만)
        save_path = os.path.join(save_dir, f"source_{idx:04d}.jpg")
        img.save(save_path)

        if (idx + 1) % 10 == 0:
            print(f"Generated {idx + 1} images")

    print(f"Successfully generated {len(text)} images in {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description='Generate source images for inference')
    parser.add_argument('--text', type=str, required=True,
                        help='input text to generate images')
    parser.add_argument('--src_font', type=str, required=True,
                        help='path to source font file')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save generated images')
    parser.add_argument('--char_size', type=int, default=150,
                        help='character size (default: 150)')
    parser.add_argument('--canvas_size', type=int, default=256,
                        help='canvas size (default: 256)')
    parser.add_argument('--x_offset', type=int, default=20,
                        help='x offset (default: 20)')
    parser.add_argument('--y_offset', type=int, default=20,
                        help='y offset (default: 20)')

    args = parser.parse_args()

    # 텍스트 파일에서 읽기
    if os.path.isfile(args.text):
        with open(args.text, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = args.text

    try:
        generate_inference_images(
            text=text,
            src_font_path=args.src_font,
            save_dir=args.save_dir,
            char_size=args.char_size,
            canvas_size=args.canvas_size,
            x_offset=args.x_offset,
            y_offset=args.y_offset
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

# --src_font /Users/doheyonkim/Depot/zi2zi/0a0d1f779b0bf2b4f1c9e6a69ed03d29 --dst_font /Users/doheyonkim/data/fontbox/ttfs/fonts_all_en/70b3548482ebaec3c86f10aeee371c63 --charset ./charset/custom_chars.txt
'''
python infer-img.py \
--text "하" \
--src_font /Users/doheyonkim/Depot/zi2zi/0a0d1f779b0bf2b4f1c9e6a69ed03d29 \
--save_dir source_images \
--char_size 200 \
--canvas_size 512 \
--x_offset 30 \
--y_offset 30
'''
if __name__ == "__main__":
    main()