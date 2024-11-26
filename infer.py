# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif


def setup_gpu():
    """GPU 설정 최적화"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # GPU 메모리 제한 (옵션)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 11)]  # 11GB
            )
        except RuntimeError as e:
            print(e)
    return tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))


def create_parser():
    """커맨드 라인 인자 파서 생성"""
    parser = argparse.ArgumentParser(description='Inference for unseen data')

    # 필수 인자
    parser.add_argument('--model_dir', required=True,
                        help='directory that saves the model checkpoints')
    parser.add_argument('--source_obj', type=str, required=True,
                        help='the source images for inference')

    # 선택적 인자
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of examples in batch')
    parser.add_argument('--embedding_ids', type=str, default='0',
                        help='comma separated embedding ids')
    parser.add_argument('--save_dir', default='inferred_images',
                        help='path to save inferred images')
    parser.add_argument('--inst_norm', type=int, default=0,
                        help='use conditional instance normalization')

    # 보간 관련 인자
    parser.add_argument('--interpolate', type=int, default=0,
                        help='interpolate between embeddings')
    parser.add_argument('--steps', type=int, default=10,
                        help='interpolation steps')
    parser.add_argument('--output_gif', type=str, default=None,
                        help='output gif filename')
    parser.add_argument('--uroboros', type=int, default=0,
                        help='connect first and last embeddings')

    return parser


def interpolate_embeddings(model, args, embeddings):
    """임베딩 벡터 간 보간 수행"""
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embedding ids for interpolation")

    chains = embeddings[:]
    if args.uroboros:
        chains.append(chains[0])

    # 연속된 임베딩 쌍 생성
    pairs = [(chains[i], chains[i + 1]) for i in range(len(chains) - 1)]

    # 각 쌍에 대해 보간 수행
    for start, end in pairs:
        print(f"Interpolating between {start} and {end}")
        model.interpolate(
            model_dir=args.model_dir,
            source_obj=args.source_obj,
            between=[start, end],
            save_dir=args.save_dir,
            steps=args.steps
        )

    # GIF 생성 (지정된 경우)
    if args.output_gif:
        gif_path = os.path.join(args.save_dir, args.output_gif)
        compile_frames_to_gif(args.save_dir, gif_path)
        print(f"GIF saved at {gif_path}")


def main(_):
    # 인자 파싱 및 GPU 설정
    args = create_parser().parse_args()
    config = setup_gpu()

    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    print("Configuration:")
    print(f"- Model directory: {args.model_dir}")
    print(f"- Source object: {args.source_obj}")
    print(f"- Save directory: {args.save_dir}")
    print(f"- Batch size: {args.batch_size}")

    with tf.compat.v1.Session(config=config) as sess:
        # 모델 초기화
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)

        # embedding_ids 파싱
        try:
            embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
            print(f"Using embedding ids: {embedding_ids}")
        except ValueError as e:
            raise ValueError(f"Invalid embedding ids format: {args.embedding_ids}") from e

        # 추론 또는 보간 실행
        if not args.interpolate:
            # 단일 임베딩 추론
            if len(embedding_ids) == 1:
                embedding_ids = embedding_ids[0]
            print(f"Running inference with embedding(s): {embedding_ids}")
            model.infer(
                model_dir=args.model_dir,
                source_obj=args.source_obj,
                embedding_ids=embedding_ids,
                save_dir=args.save_dir
            )
        else:
            # 임베딩 간 보간
            print("Running interpolation")
            interpolate_embeddings(model, args, embedding_ids)
'''
# 기본 추론
python inference.py \
--model_dir ./checkpoints \
--source_obj ./source_images \
--embedding_ids 0,1,2 \
--save_dir ./inferred_images

# 보간 with GIF
python inference.py \
--model_dir ./checkpoints \
--source_obj ./source_images \
--embedding_ids 0,1,2 \
--interpolate 1 \
--steps 10 \
--output_gif transition.gif \
--save_dir ./interpolated_images

# 단일 스타일 변환
python inference.py \
--model_dir ./checkpoints \
--source_obj ./source_images \
--embedding_ids 0 \
--save_dir ./single_style
'''

if __name__ == '__main__':
    tf.compat.v1.app.run()