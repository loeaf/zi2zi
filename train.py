# -*- coding: utf-8 -*-



import tensorflow as tf
# GPU 메모리 분할 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 두 GPU에 작업 분산을 위한 설정
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    except RuntimeError as e:
        print(e)

import argparse

from model.unet import UNet

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', dest='Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=4,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', dest='fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', dest='flip_labels', type=int, default=None,
                    help='whether flip training data labels or not, in fine tuning')
args = parser.parse_args()
'''
python train.py \
--experiment_dir=/data/dataset \
--experiment_id=0 \
--batch_size=400 \
--lr=0.001 \
--epoch=400 \
--sample_steps=10 \
--schedule=20 \
--L1_penalty=100 \
--Lconst_penalty=15
'''

'''
python train.py --experiment_dir=/data/dataset2 \
                --experiment_id=0 \
                --batch_size=400 \
                --lr=0.0005 \  # 더 낮은 학습률
                --epoch=400 \
                --sample_steps=10 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --fine_tune=<영문폰트레이블> \
                --resume=1 \
                --freeze_encoder=1  # 인코더 고정
'''


def main(_):
    # GPU 메모리 설정
    config = tf.compat.v1.ConfigProto()
    # GPU 메모리 제한 설정
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 각 GPU의 90% 사용

    # GPU 간 작업 분배 설정
    config.gpu_options.visible_device_list = "0,1"
    config.allow_soft_placement = True


    # 두 GPU에 작업 분산
    with tf.device('/cpu:0'):
        tower_grads = []

        with tf.compat.v1.Session(config=config) as sess:
            model = UNet(args.experiment_dir, batch_size=args.batch_size,
                         experiment_id=args.experiment_id,
                         input_width=args.image_size, output_width=args.image_size,
                         embedding_num=args.embedding_num,
                         embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty,
                         Lconst_penalty=args.Lconst_penalty,
                         generator_dim=16, discriminator_dim=16,
                         Ltv_penalty=args.Ltv_penalty,
                         Lcategory_penalty=args.Lcategory_penalty)

            # GPU 사용량을 균등하게 분배
            split_batch_size = args.batch_size // 2

            # 각 GPU에 모델 복제
            with tf.device('/gpu:0'):
                model_gpu0 = model
                model_gpu0.batch_size = split_batch_size

            with tf.device('/gpu:1'):
                model_gpu1 = model
                model_gpu1.batch_size = split_batch_size

            model.register_session(sess)
            if args.flip_labels:
                model.build_model(is_training=True, inst_norm=args.inst_norm,
                                  no_target_source=True)
            else:
                model.build_model(is_training=True, inst_norm=args.inst_norm)

            # 나머지 training 코드
            fine_tune_list = None
            if args.fine_tune:
                ids = args.fine_tune.split(",")
                fine_tune_list = set([int(i) for i in ids])

            model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                        schedule=args.schedule, freeze_encoder=args.freeze_encoder,
                        fine_tune=fine_tune_list,
                        sample_steps=args.sample_steps,
                        checkpoint_steps=args.checkpoint_steps,
                        flip_labels=args.flip_labels)


if __name__ == '__main__':
    tf.compat.v1.app.run()