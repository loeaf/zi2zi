# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_single_font(args):
    idx, target_font_path, src_font, samples_base_dir = args
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
        print(f"\n처리 중 ({idx + 1}): {target_font_path}")
        subprocess.call(cmd)
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)

def process_multiple_fonts():
    # 경로 설정
    src_font = "/Users/doheyonkim/Depot/zi2zi/d461cae31481b059a9e85df14fffe0d5"
    target_fonts_dir = "/Users/doheyonkim/data/fontbox/dataset1/fonts"
    target_fonts_dir2 = "/Users/doheyonkim/data/fontbox/ttfs/fonts_all_en"
    samples_base_dir = "/Users/doheyonkim/data/fontbox/dataset1"

    # samples 기본 디렉토리 생성
    if not os.path.exists(samples_base_dir):
        os.makedirs(samples_base_dir)

    # 타겟 폰트 파일들 리스트 가져오기
    target_fonts = []
    for file in os.listdir(target_fonts_dir):
        target_fonts.append(os.path.join(target_fonts_dir, file))

    # for file in os.listdir(target_fonts_dir2)[:20]:
    #     target_fonts.append(os.path.join(target_fonts_dir2, file))

    print(f"처리할 폰트 수: {len(target_fonts)}")

    # ProcessPoolExecutor 설정
    num_cores = multiprocessing.cpu_count()
    print(f"사용 가능한 CPU 코어 수: {num_cores}")

    # 작업 목록 생성
    tasks = [
        (idx, font_path, src_font, samples_base_dir)
        for idx, font_path in enumerate(target_fonts)
    ]

    # 병렬 처리 실행
    completed = 0
    failed = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for idx, success, error in executor.map(process_single_font, tasks):
            completed += 1
            if not success:
                failed.append((target_fonts[idx], error))
            print(f"진행 상황: {completed}/{len(target_fonts)} 완료")

    # 결과 보고
    print("\n처리 완료!")
    print(f"총 처리된 폰트: {completed}")
    if failed:
        print("\n실패한 폰트들:")
        for font, error in failed:
            print(f"- {font}: {error}")

if __name__ == "__main__":
    process_multiple_fonts()