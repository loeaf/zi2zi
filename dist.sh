python font2img.py --src_font=korean.ttf \
                   --dst_font=english.ttf \
                   --charset=custom_chars.txt \  # 공통 지원 문자만 포함
                   --sample_count=1000 \
                   --sample_dir=samples \
                   --label=0 \
                   --filter=1                    # 필터링 활성화



python package.py --dir=/Volumes/Extreme SSD/dataset \    # 이미지가 있는 디렉토리
                 --save_dir=/Volumes/Extreme SSD/dataset \ # .obj 파일이 저장될 디렉토리
                 --split_ratio=0.9          # train.obj와 val.obj로 분할 비율


python package.py \
    --dir="/Volumes/Extreme SSD/dataset" \
    --save_dir="/Volumes/Extreme SSD/dataset" \
    --split_ratio=0.9

python package-multi.py \
    --dir="/Volumes/Extreme SSD/dataset2" \
    --save_dir="/Volumes/Extreme SSD/dataset2" \
    --split_ratio=0.1


experiment/
└── data/
    ├── train.obj  # 학습 데이터
    └── val.obj    # 검증 데이터


pip install --upgrade pip
pip install tensorflow-macos
pip install tensorflow-metal  # Apple GPU 지원을 위한 패키지

 python train.py --experiment_dir='/Volumes/Extreme SSD/dataset2' \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=40 \
                --sample_steps=50 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --embedding_num=4


 python train.py --experiment_dir="/data/dataset2" \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=40 \
                --sample_steps=50 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --embedding_num=4



# 단순 변환
python infer.py --model_dir=experiment/checkpoint \
                --batch_size=16 \
                --source_obj=binary/val.obj \
                --embedding_ids=0 \
                --save_dir=results

# 스타일 보간 (여러 스타일 사이 중간 단계 생성)
python infer.py --model_dir=experiment/checkpoint \
                --batch_size=16 \
                --source_obj=binary/val.obj \
                --embedding_ids=0,1 \
                --save_dir=frames \
                --output_gif=transition.gif \
                --interpolate=1 \
                --steps=10