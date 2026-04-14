# 付録: ハフ投票によるサイノグラムテンプレートマッチング

NCC-HF手法の前身として開発された手法。テンプレートが画像の主要コンテンツ（面積比25%以上）である場合に高速に動作する。

## アルゴリズム

### 原理

画像上の1点 (dx, dy) はラドン変換のサイノグラム上で正弦波 `offset(theta) = dx*cos(theta) + dy*sin(theta)` を描く。テンプレートのサイノグラムコア（投影本体部分のみ）を画像のサイノグラム行と1D NCC (matchTemplate) でマッチングし、各一致結果から回転角度 `alpha = (theta_image - theta_template) % 180` にハフ投票する。

### 処理フロー

```
1. 適応的コントラスト正規化 (contrast_ratio < 0.6 の場合)
2. ラドン変換 → float32サイノグラム (360角度)
3. サイノグラムコア抽出 (端部填充を除去)
4. ハフ投票: 360×180 = 64,800ペアの1D NCC
   → alpha = (theta_img - theta_tmpl) % 180 に投票
5. アキュムレータのピーク → 回転角度
6. 正弦波フィッティング → (dx, dy) 推定
```

### 手法の特徴

FFT振幅やPOC (位相限定相関) が各サイノグラム行を独立に処理するのに対し、ハフ投票は全行を正弦波モデルで拘束する。これにより単一行のマッチング失敗に対してロバストな角度検出が可能。

## ベンチマーク結果

### COIL-20 (面積比25%, 黒背景)

| Condition | Mean Error | <=5 deg | <=10 deg |
|---|---|---|---|
| Clean | 0.8 deg | 97% | 100% |
| Gaussian sigma=25 | 0.7 deg | 97% | 100% |
| Brightness +50 | 0.8 deg | 97% | 100% |
| Contrast 0.5x | 1.1 deg | 96% | 98% |

### 速度 (C++, OpenMP)

| Template Size | Brute-force 2D NCC | Hough Voting | Speedup |
|---|---|---|---|
| 64x64 | 80 ms | 27 ms | 3.0x |
| 128x128 | 326 ms | 58 ms | 5.7x |
| 200x200 | 1,206 ms | 70 ms | 17.3x |
| 256x256 | 1,925 ms | 106 ms | 18.1x |

## 制約

- **面積比依存**: テンプレートが画像の25%以上を占める場合に最適。自然画像背景で面積比8%程度では投票が散逸し、角度検出に失敗する
- **180度曖昧性**: ハフ投票は180度周期で角度を検出する。曖昧性の解消には追加の検証ステップが必要
- サイノグラムは画像全体の線積分であるため、背景の影響を原理的に排除できない

## 関連ファイル

- `radon_template_matching.py`: `detectAngleHough()`
- `radon_template_matching.cpp`: `detectAngleHough()`
- `evaluate_coil20.py --method hough`: COIL-20でのベンチマーク
- `docs/hf_energy_minimization_study.md`: HFエネルギー最小化の検討記録（ハフ投票→HFエネルギー→NCC-HFへの発展過程）
