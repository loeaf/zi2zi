# 1. 현재 환경 삭제
conda deactivate
conda remove --name zi2zi --all

# 2. 새로운 환경 생성과 함께 특정 버전의 패키지들 설치
conda create -n zi2zi python=3.9
conda activate zi2zi

# 3. 필요한 패키지들을 특정 버전으로 설치
pip install numpy==1.24.3  # NumPy 1.x 버전 사용
pip install tensorflow-macos==2.13.0  # Apple Silicon용 TensorFlow
pip install tensorflow-metal  # GPU 지원
pip install Pillow
pip install scipy
pip install scikit-learn
# 4. 추가로 필요한 패키지들
pip install tqdm
pip install opencv-python




# linux 환경



# 1. 기존 환경 제거
conda deactivate
conda remove -n zi2zi --all

# 2. 새로운 환경 생성 (cuda-toolkit 포함)
conda create -n zi2zi python=3.9
conda activate zi2zi

# 3. CUDA 11.0 호환 TensorFlow 설치
conda install -c conda-forge cudatoolkit=11.0
conda install -c conda-forge cudnn=8.0
pip install tensorflow==2.9.0

# 4. CUDA path 설정
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 5. 필요한 추가 패키지 설치
pip install numpy==1.24.3
pip install Pillow
pip install scipy
pip install imageio
pip install tqdm