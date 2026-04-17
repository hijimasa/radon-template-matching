# radon-template-matching

ラドン変換のサイノグラム空間NCC-HFによる回転不変テンプレートマッチング

Rotation-invariant template matching via NCC-HF (Normalized Cross-Correlation in High-pass Filtered sinogram space)

![実行結果](./figs/result.png)

## Algorithm / アルゴリズム

### NCC-HF: サイノグラム空間のローカル正規化相関

ラドン変換のサイノグラム上で、ハイパスフィルタ適用済みのテンプレートコアと画像行のローカルNCC (Normalized Cross-Correlation) を最大化することで、テンプレートの回転角度と位置を同時に検出する。

**原理**: 画像上の点 (dx, dy) はサイノグラム上で正弦波 `offset(theta) = dx*cos(theta) - dy*sin(theta)` を描く。各角度候補 alpha について、テンプレートのサイノグラムコアを画像サイノグラム行に沿ってスライドさせ、HPFバンドでのNCCプロファイルを計算する。正弦波パス上のNCC平均値が最大となる (alpha, dx, dy) が検出結果となる。

**NCC-HFの利点**: 従来のHFエネルギー最小化はサイノグラム全長の残差エネルギーを評価するため、背景のベースラインに支配されやすい。NCC-HFはコア領域のみのローカル正規化相関を使うため、ベースラインの影響を正規化で吸収し、自然画像背景・黒背景の両方で安定して動作する。

### Processing Flow / 処理フロー

```
1. Adaptive contrast normalization (if contrast_ratio < 0.6)
   適応的コントラスト正規化

2. Radon transform -> float32 sinogram (360 angles)
   ラドン変換 -> float32サイノグラム (画像・テンプレートキャンバス各1回)

3. Precompute NCC-HF data
   事前計算: HPF適用済みFFT・累積和・コア統計量

4. For each candidate angle alpha:
   各候補角度 alpha に対して:
   a. Batch cross-correlation via IFFT  (バッチ相互相関)
   b. Local NCC profile from running stats (ローカルNCC算出)
   c. Hierarchical position search (step=4 -> 2 -> 1)
      階層的位置探索 (正弦波パスに沿ったNCC平均の最大化)

5. Coarse angle search (step=3 deg, 120 candidates)
   -> Top-5 candidates refined at 1 deg resolution
   粗角度探索 -> 上位5候補の近傍で1度刻み精密化

6. Result: (angle, dx, dy) with NCC-HF score
   結果: 角度・位置・スコア (画像空間NCC精密化は不要)
```

### Key Innovation / 手法の特徴

| 課題 | HFエネルギー最小化 | NCC-HF |
|---|---|---|
| 指標 | 残差の二乗和 | ローカル正規化相関 |
| 背景ベースライン | 支配される | 正規化で吸収 |
| 角度検出 (自然画像 8%) | rank 338/360 | **rank 1/360** |
| 角度検出 (黒背景 25%) | 未検証 | **100% (<=5 deg)** |
| NCC精密化 | 必要 | **不要** |

## Benchmark Results / ベンチマーク結果

### COIL-20 (20 grayscale objects, 128x128)

テスト方法: 各オブジェクトの0度ポーズ画像をテンプレートとし、**2D画像回転** (`cv2.warpAffine`) で10角度 (0-170 deg) に回転した画像を2倍サイズの黒背景上に配置。この2D回転角度を検出する。3D姿勢変化の認識ではない。

| Condition | Method | Mean Error | <=5 deg | <=10 deg |
|---|---|---|---|---|
| Clean | **NCC-HF** | **0.0 deg** | **100%** | **100%** |
| | Hough Voting | 0.8 deg | 97% | 100% |
| Gaussian sigma=25 | **NCC-HF** | **0.1 deg** | **100%** | **100%** |
| | Hough Voting | 0.7 deg | 97% | 100% |
| Brightness +50 | **NCC-HF** | **0.0 deg** | **100%** | **100%** |
| | Hough Voting | 0.8 deg | 97% | 100% |
| Contrast 0.5x | **NCC-HF** | **0.0 deg** | **100%** | **100%** |
| | Hough Voting | 1.1 deg | 96% | 98% |

### Natural Image (640x543, template 170x163, area ratio 8%)

犬の鼻のクロップをテンプレートとし、元画像中の30度回転配置を検出するテスト。

| Method | Angle | dx | dy | Time (Python) |
|---|---|---|---|---|
| Ground truth | 30 deg | 38 | 55 | - |
| **NCC-HF** | **30 deg** | **37** | **57** | **3.6s** |
| Brute-force 2D NCC | 30 deg | 38 | 55 | 2.2s |
| HF energy (old) | 検出不能 (rank 338) | - | - | 75s |

### Speed: NCC-HF vs Brute-force 2D NCC (C++)

合成テスト (128x128 template, 256x256 image):

| Method | Time | Angle Error |
|---|---|---|
| **NCC-HF** | **185 ms** | 0 deg |
| Brute-force 2D NCC | 313 ms | 0 deg |
| Hough Voting | 57 ms | 0 deg |

自然画像テスト (1/4 scale):

| Method | Time | Angle Error |
|---|---|---|
| **NCC-HF (C++)** | **1747 ms** | 0 deg |
| Brute-force 2D NCC (C++) | 2174 ms | 0 deg |

## How to Use / 使い方

### Python

```bash
pip install -r requirements.txt
python radon_template_matching.py
```

### C++ (with benchmark)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
./build/RadonTemplateMatching
```

### Evaluation / 評価

```bash
# COIL-20 benchmark with NCC-HF (default, auto-downloads dataset)
python evaluate_coil20.py

# COIL-20 benchmark with Hough Voting
python evaluate_coil20.py --method hough
```

## File Structure / ファイル構成

```
radon_template_matching.py   ... Python implementation (NCC-HF + Hough voting)
radon_template_matching.cpp  ... C++ implementation (OpenMP parallelized)
radon_template_matching.hpp  ... C++ header
main.cpp                     ... C++ benchmark
evaluate_coil20.py           ... COIL-20 benchmark (--method ncchf|hough)
evaluate_datasets.py         ... Multi-dataset benchmark (Hough voting)
docs/
  hf_energy_minimization_study.md  ... HFエネルギー最小化の検討記録
  hough_voting.md                  ... Hough投票手法の詳細 (付録)
```

## Technical Notes / 技術メモ

### HPFカットオフ比率

NCC-HFのHPFカットオフは `cutoff_ratio=1/16` (DCおよび最低6%の超低周波を除去) がデフォルト。この設定は黒背景 (COIL-20) と自然画像背景の両方で安定して動作する。

- カットオフが高すぎる (1/8以上): 黒背景で判別力が不足し、0度付近で大外れが発生
- カットオフが低すぎる (1/32以下): 背景の低周波成分が残り、自然画像での判別力が低下する可能性

### テンプレートキャンバスの充填値

テンプレートをサイノグラム変換する際、キャンバスの背景は画像の `corner_pixels_mean` (辺縁画素平均) で充填する。これにより画像とテンプレートのサイノグラム境界外基準が統一され、残差にベースライン差が生じない。

## References / 参照

- COIL-20 dataset: Columbia University CAVE Lab
- MPEG-7 CE-Shape-1: Temple University DABI Lab
