# 当前尾实体预测数据

训练和独立评估默认只读取本目录，不依赖外部 RED-GNN 路径。

| CLI 数据集 | 目录 | facts / train / valid / test |
|---|---|---|
| FB15k-237 | FB15k-237 | 204087 / 68028 / 17535 / 20466 |
| nell | nell | 112258 / 37420 / 543 / 2818 |
| umls | umls | 4006 / 1321 / 569 / 633 |

前三个目录的 entities.txt、relations.txt、facts.txt、train.txt、valid.txt、test.txt 从 `RED-GNN-main/Static/transductive/data/` 的对应目录逐字节复制，无重编号或重新划分。`data_manifest.json` 记录来源、行数和 SHA-256。旧 FMS 原始副本保留在项目根目录的 `FMS-main/Entity_Prediction/data/`。


归纳式数据目录（fb237_v1–v4_ind、nell_v1–v4_ind）已删除，当前仅保留传导式运行所需数据与原始 Kinship。文本三元组顺序为 head relation tail，词表每行一个名称。

原始 `kinship/` 保留，当前入口不运行该数据集。
