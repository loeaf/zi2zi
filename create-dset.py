# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import subprocess


def process_multiple_fonts():
    # 경로 설정
    src_font = "/Users/doheyonkim/Depot/zi2zi/d461cae31481b059a9e85df14fffe0d5"
    target_fonts_dir = "/Users/doheyonkim/data/fontbox/ttfs/fonts_all_ko"
    target_fonts_dir2 = "/Users/doheyonkim/data/fontbox/ttfs/fonts_all_en"
    samples_base_dir = "/Users/doheyonkim/data/fontbox/test"

    # samples 기본 디렉토리 생성
    if not os.path.exists(samples_base_dir):
        os.makedirs(samples_base_dir)

    # 타겟 폰트 파일들 리스트 가져오기
    target_fonts = []
    for file in os.listdir(target_fonts_dir)[:20]:
        target_fonts.append(os.path.join(target_fonts_dir, file))

    for file in os.listdir(target_fonts_dir2)[:20]:
        target_fonts.append(os.path.join(target_fonts_dir2, file))

    print(f"처리할 폰트 수: {len(target_fonts)}")

    # 각 타겟 폰트에 대해 처리
    for idx, target_font_path in enumerate(target_fonts):
        try:
            # 현재 폰트를 위한 샘플 디렉토리 생성
            font_sample_dir = os.path.join(samples_base_dir, f"font_{idx}")
            if not os.path.exists(font_sample_dir):
                os.makedirs(font_sample_dir)


            # font2img.py 실행
            cmd = [
                "python", "font2img.py",
                "--src_font", src_font,
                "--dst_font", target_font_path,
                "--charset", "./charset/custom_chars.txt",
                "--sample_count", "2402",
                "--sample_dir", font_sample_dir,
                "--label", str(idx),
                "--filter", "1"
            ]
            print(f"실행 명령어: {' '.join(cmd)}")
            print(f"\n처리 중 ({idx + 1}/{len(target_fonts)}): {target_font_path}")
            subprocess.call(cmd)

        except Exception as e:
            print(f"오류 발생 - {target_font_path}: {str(e)}")


if __name__ == "__main__":
    process_multiple_fonts()