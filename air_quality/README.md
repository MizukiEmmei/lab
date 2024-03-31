my_moduleとulcaを兄弟ディレクトリから読み込む方法がわからないのでとりあえず複製を同じディレクトリに置いておく

# MultiDimReduction
ULCAによる比較指標を用いたMultiDimReduction

# 実行方法
`bokeh serve --show app_multi_ulca.py`

# 注意事項
クラスタを選択する際に,他の色を塗りつぶしてしまうとheatmapのところでエラー発生.(灰色がなくなるように色を塗っていくなど)

# 仕様
p1でクラスタ選択 : p1~p5更新
p6でクラスタ選択 : p6~p10更新

p1で次元削減するクラスタを選択 : p1,p2更新 p3~p5初期化
p6で次元削減するクラスタを選択 : p6,p7更新 p8~p10初期化

p1でULCA : p1で選択したクラスタを比較するULCA
p6でULCA : p6で選択したクラスタを比較するULCA