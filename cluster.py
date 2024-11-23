
import os
# OpenMP 관련 모든 환경 변수 설정
# Intel MKL을 GNU OpenMP나 TBB로 전환
os.environ['MKL_THREADING_LAYER'] = 'TBB'  # 또는 'TBB'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
'''
# Intel OpenMP 비활성화
export KMP_DUPLICATE_LIB_OK=TRUE

# 또는 LLVM OpenMP 사용
export KMP_DUPLICATE_LIB_OK=TRUE
# set env

'''


def analyze_clusters(pickle_file):
    """
    클러스터링 결과를 분석하고 통계를 출력합니다.

    Args:
        pickle_file: 저장된 클러스터링 결과 파일 경로
    """
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    # 클러스터별 폰트 수 계산
    cluster_counts = {}
    for key, cluster_id in results.items():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    # 기본 통계
    n_clusters = len(cluster_counts)
    total_samples = len(results)
    avg_cluster_size = total_samples / n_clusters

    print("\n=== Clustering Analysis ===")
    print(f"Total number of samples: {total_samples}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {avg_cluster_size:.2f}")

    # 가장 큰/작은 클러스터
    max_cluster = max(cluster_counts.items(), key=lambda x: x[1])
    min_cluster = min(cluster_counts.items(), key=lambda x: x[1])

    print(f"\nLargest cluster (ID: {max_cluster[0]}): {max_cluster[1]} samples")
    print(f"Smallest cluster (ID: {min_cluster[0]}): {min_cluster[1]} samples")

    # 클러스터 크기 분포 시각화
    plt.figure(figsize=(12, 6))
    plt.hist(cluster_counts.values(), bins=30)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters')
    plt.show()

    return results


def view_cluster_samples(pickle_file, cluster_id, n_samples=5):
    """
    특정 클러스터에 속한 샘플들을 출력합니다.

    Args:
        pickle_file: 저장된 클러스터링 결과 파일 경로
        cluster_id: 보고 싶은 클러스터 ID
        n_samples: 출력할 샘플 수
    """
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    # 해당 클러스터의 샘플들 찾기
    cluster_samples = [(k, v) for k, v in results.items() if v == cluster_id]

    print(f"\n=== Samples from Cluster {cluster_id} ===")
    for font_char, _ in cluster_samples[:n_samples]:
        font_name, char = font_char.split('_')
        print(f"Font: {font_name}, Character: {char}")


def interactive_cluster_viewer(pickle_file):
    """
    대화형으로 클러스터를 탐색하는 함수
    """
    results = analyze_clusters(pickle_file)

    while True:
        print("\n=== Cluster Viewer ===")
        print("1. View specific cluster")
        print("2. View cluster statistics")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            cluster_id = int(input("Enter cluster ID: "))
            n_samples = int(input("How many samples to show? "))
            view_cluster_samples(pickle_file, cluster_id, n_samples)
        elif choice == '2':
            analyze_clusters(pickle_file)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")


def find_optimal_clusters(features, max_clusters=100, step=5):
    """
    엘보우 메소드와 실루엣 분석을 통해 최적의 클러스터 수를 찾습니다.

    Args:
        features: 특징 벡터
        max_clusters: 테스트할 최대 클러스터 수
        step: 클러스터 수 증가 단위

    Returns:
        optimal_k_elbow: 엘보우 메소드로 찾은 최적 클러스터 수
        optimal_k_silhouette: 실루엣 분석으로 찾은 최적 클러스터 수
    """
    print("Finding optimal number of clusters...")

    # 테스트할 클러스터 수 범위
    k_range = range(2, max_clusters + 1, step)

    # 결과 저장용 리스트
    inertias = []
    silhouette_scores = []

    # 각 k에 대해 클러스터링 수행
    for k in tqdm(k_range, desc="Testing different cluster numbers"):
        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)

        # 엘보우 메소드를 위한 inertia 저장
        inertias.append(kmeans.inertia_)

        # 실루엣 스코어 계산 (샘플링하여 계산 속도 향상)
        sample_size = min(10000, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        score = silhouette_score(
            features[indices],
            kmeans.predict(features[indices]),
            sample_size=sample_size
        )
        silhouette_scores.append(score)

    # 엘보우 포인트 찾기
    elbow = KneeLocator(
        list(k_range),
        inertias,
        curve='convex',
        direction='decreasing'
    )
    optimal_k_elbow = elbow.knee

    # 최적의 실루엣 스코어를 가진 k 찾기
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 엘보우 곡선
    ax1.plot(k_range, inertias, 'bx-')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    if optimal_k_elbow:
        ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--')

    # 실루엣 스코어
    ax2.plot(k_range, silhouette_scores, 'rx-')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()

    print(f"Optimal number of clusters:")
    print(f"- Elbow method: {optimal_k_elbow}")
    print(f"- Silhouette analysis: {optimal_k_silhouette}")

    return optimal_k_elbow, optimal_k_silhouette

def create_character_image(char, font_path, size=(256, 256), font_size=200):
    """단일 문자의 이미지를 생성"""
    try:
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)

        # 폰트 로드
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            return None

        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 중앙 정렬 위치 계산
            x = (size[0] - text_width) / 2
            y = (size[1] - text_height) / 2

            # 텍스트 그리기
            draw.text((x, y), char, fill='black', font=font)
            return img

        except Exception as e:
            print(f"Error processing font {font_path}: {e}")
            return None

    except Exception as e:
        print(f"Unexpected error for {font_path}: {e}")
        return None


class FontDataset(Dataset):
    def __init__(self, font_folders, characters, max_fonts=50):
        self.images = []
        self.font_chars = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # 폰트 파일 수집
        font_paths = []
        for folder in font_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    font_paths.append(os.path.join(folder, file))

        if len(font_paths) > max_fonts:
            print(f"Limiting to {max_fonts} fonts from {len(font_paths)} total fonts")
            font_paths = font_paths[:max_fonts]

        print(f"Processing {len(font_paths)} fonts")

        total = len(font_paths) * len(characters)
        with tqdm(total=total, desc="Generating images") as pbar:
            for font_path in font_paths:
                for char in characters:
                    try:
                        img = create_character_image(char, font_path)
                        if img is not None:
                            self.images.append(img)
                            # 폰트 경로와 문자를 함께 저장
                            self.font_chars.append({
                                'font_path': font_path,
                                'char': char
                            })
                    except Exception as e:
                        print(f"Error processing {font_path}, {char}: {e}")
                    pbar.update(1)

        print(f"Successfully generated {len(self.images)} images")
        if len(self.images) == 0:
            raise ValueError("No valid images were generated")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        # 단일 튜플 반환
        return img, self.font_chars[idx]


def save_results(font_chars, cluster_labels, output_file):
    """클러스터링 결과를 저장"""
    results = {}

    print(f"Processing {len(font_chars)} font_chars and {len(cluster_labels)} cluster labels")
    print(f"Sample font_chars: {font_chars[:2]}")  # 데이터 구조 확인

    try:
        for i, font_char in enumerate(font_chars):
            if i >= len(cluster_labels):
                print(f"Warning: More font_chars than cluster labels")
                break

            # 데이터 구조에 따라 적절히 처리
            try:
                # 튜플인 경우
                if isinstance(font_char, tuple) and len(font_char) == 2:
                    font_path, char = font_char
                # 리스트인 경우
                elif isinstance(font_char, list) and len(font_char) == 2:
                    font_path, char = font_char
                # 다른 구조인 경우
                else:
                    print(f"Unexpected font_char format at index {i}: {font_char}")
                    continue

                font_name = os.path.basename(font_path)
                key = f"{font_name}_{char}"
                results[key] = int(cluster_labels[i])

                # 처음 5개 결과 출력
                if i < 5:
                    print(f"Sample result {i}: {key} -> Cluster {cluster_labels[i]}")

            except Exception as e:
                print(f"Error processing font_char at index {i}: {font_char}")
                print(f"Error details: {e}")
                continue

    except Exception as e:
        print(f"Error in save_results: {e}")
        print(f"Current font_char type: {type(font_chars)}")
        print(f"Current font_char structure: {font_chars[:5]}")
        raise

    # 결과 저장
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} results to {output_file}")
    return results


def extract_features(dataset, batch_size=32, device='mps'):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    features = []
    font_chars = []

    print("Starting feature extraction...")
    with torch.no_grad():
        for batch, meta in tqdm(dataloader, desc="Extracting features"):
            if batch.shape[0] == 0:
                continue

            batch = batch.to(device)
            try:
                output = model(batch)
                features.append(output.cpu().numpy())

                # meta는 딕셔너리로, font_path와 char가 리스트로 들어있음
                font_chars.append({
                    'font_path': meta['font_path'][0],  # 리스트의 첫 번째 항목
                    'char': meta['char'][0]  # 리스트의 첫 번째 항목
                })

            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch shape: {batch.shape}")
                print(f"Meta structure: {meta}")
                continue

    if not features:
        raise ValueError("No features were extracted")

    features = np.concatenate(features)
    print(f"Features shape: {features.shape}")
    print(f"Font chars count: {len(font_chars)}")
    print(f"Sample font_chars: {font_chars[:2]}")

    return features, font_chars


def cluster_styles(features, n_clusters=2048):
    """특징을 클러스터링"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)



def main():
    # M1/M2 Mac에서 MPS 디바이스 사용
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    # 설정
    font_folders = [
        '/Users/doheyonkim/data/fontbox/ttfs/fonts_all_ko',
        '/Users/doheyonkim/data/fontbox/ttfs/fonts_all_en'
    ]

    # 폴더 존재 확인
    for folder in font_folders:
        if not os.path.exists(folder):
            raise ValueError(f"Font folder not found: {folder}")

    # 한글 및 영문 문자 설정
    english_chars = list('abjwEFYZ')
    characters = english_chars

    try:
        # 1. 데이터셋 생성
        print("Creating dataset...")
        dataset = FontDataset(font_folders, characters)
        print(f"Dataset size: {len(dataset)}")

        # 2. 특징 추출
        print("Extracting features...")
        features, font_chars = extract_features(dataset, device=device)
        print(f"Extracted features shape: {features.shape}")

        # 3. 최적의 클러스터 수 찾기
        print("Finding optimal number of clusters...")
        optimal_k_elbow, optimal_k_silhouette = find_optimal_clusters(
            features,
            max_clusters=min(100, len(dataset) // 2),
            step=5
        )

        # 3. 클러스터링
        print("Clustering...")
        # 최종 클러스터 수 결정 (엘보우 메소드 사용)
        n_clusters = optimal_k_elbow if optimal_k_elbow else min(10, len(dataset))
        cluster_labels = cluster_styles(features, n_clusters=n_clusters)

        # 4. 결과 저장
        output_file = 'style_clusters.pkl'
        save_results(font_chars, cluster_labels, output_file)
        print(f"Results saved to {output_file}")

        # 5. 결과 분석
        print("\nAnalyzing clustering results...")
        analyze_clusters(output_file)

        # 6. 대화형 클러스터 뷰어 실행
        print("\nLaunching interactive cluster viewer...")
        interactive_cluster_viewer(output_file)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
