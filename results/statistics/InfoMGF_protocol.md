# InfoMGF-LA实验设置

- 发表来源：Z. Shen、S. Wang、Z. Kang，Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning，NeurIPS 2024。
- 官方实现commit：27962685731a7cde96fb839c3fcf1b45e6cfca2f；使用LA（learned augmentation）分支。
- 数据：8张真实图与320张240节点仿真图。原始节点、行序、层、二值邻接、K和评价掩码保持一致。
- 特征：各图统一采用无标签单位阵输入X=I；沿用作者的r=2聚合及图学习器。
- 固定设置：100个原生epoch，每epoch一次表示学习Adam更新和一次增广生成器Adam更新；最终checkpoint参与评价。隐藏128维，嵌入及投影64维，2层GCN，k=15，dropout及edge dropout均为0.5。表示学习率0.01，生成器学习率0.001，增广正则系数0.01。
- 生成器temperature固定为1（论文Table6），采样bias为0.0001（作者类默认值）。显式传参纠正发布driver的位置参数错配；作者的数值类和损失保持不变。CPU设备分配和对角乘法改写已完成原有值与梯度等价性核验。
- 每图10个预定训练seed，公式931100000+1000*目录图序号+repeat；repeat为0至9。两位执行者分别承担0至4和5至9，没有重叠。
- 每次训练固定10个KMeans seed（0至9），n_init=10、max_iter=300、tol=1e-4，分数在训练内先平均；不按ARI挑选seed、epoch或KMeans。
- 每张真实图在10次训练间求均值与样本标准差。仿真每图先平均10次训练，再对每条件10张独立图汇总。
- 评价沿用原标签掩码及节点—层口径，ARI、算术NMI、全局Hungarian匹配ACC。
- 3280次正式训练及服务器新进程权重重放记录均已核对，实际预测与嵌入hash一致。收到的小包没有权重，未新增本地权重重放。原OOM及资源修复记录保留。

本稿新增范围为上述8张真实图与320张仿真图；大规模MNIST/arXiv章节沿用已有SCML/DGFM3实验。
