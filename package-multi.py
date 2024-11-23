import argparse
import glob
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 진행 상황 표시용


def process_image(p, train_val_split):
    """단일 이미지 처리 함수"""
    try:
        label = int(os.path.basename(p).split("_")[0])
        with open(p, 'rb') as f:
            img_bytes = f.read()
            return (label, img_bytes, random.random() < train_val_split)
    except Exception as e:
        print(f"Error processing {p}: {str(e)}")
        return None


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
    """병렬 처리를 사용한 이미지 피클링"""
    train_examples = []
    val_examples = []

    # ThreadPoolExecutor를 사용한 병렬 처리
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # tqdm으로 진행 상황 표시
        futures = [executor.submit(process_image, p, train_val_split) for p in paths]
        for future in tqdm(futures, desc="Processing images"):
            result = future.result()
            if result:
                label, img_bytes, is_val = result
                if is_val:
                    val_examples.append((label, img_bytes))
                else:
                    train_examples.append((label, img_bytes))

    # 한 번에 파일 쓰기
    print(f"Writing {len(train_examples)} training examples...")
    with open(train_path, 'wb') as ft:
        for example in train_examples:
            pickle.dump(example, ft, protocol=2)

    print(f"Writing {len(val_examples)} validation examples...")
    with open(val_path, 'wb') as fv:
        for example in val_examples:
            pickle.dump(example, fv, protocol=2)


def main():
    parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
    parser.add_argument('--dir', dest='dir', required=True, help='path of examples')
    parser.add_argument('--save_dir', dest='save_dir', required=True, help='path to save pickled files')
    parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                        help='split ratio between train and val')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_path = os.path.join(args.save_dir, "train.obj")
    val_path = os.path.join(args.save_dir, "val.obj")

    image_paths = sorted(glob.glob(os.path.join(args.dir, "**/*.jpg"), recursive=True))
    if not image_paths:
        print(f"No images found in {args.dir}")
        return

    print(f"Found {len(image_paths)} images")
    pickle_examples(image_paths, train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)
    print("Processing completed!")


if __name__ == "__main__":
    main()