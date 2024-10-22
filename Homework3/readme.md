# 凝聚层次聚类（Agglomerative Clustering）

## 简介

本项目包含一个凝聚层次聚类算法的完整实现代码。使用了欧氏距离（Euclidean）计算数据点之间的距离和平均链接（Average Linkage）策略来计算聚类间的距离，并以此为基础进行聚类的合并。其中考虑了时间复杂度和空间复杂度的优化，通过有效的数据结构和算法逻辑，减少了不必要的计算。


## 思考过程

1. 在聚类之前，需要一个距离矩阵来记录数据点之间的距离。
2. 因为是平均链接策略，为了合并距离最近的两个聚类，需要遍历所有聚类，计算并比较每对聚类之间的平均距离。
3. 需要一个聚类索引来分辨聚类，每当合并一个新的聚类时，需要添加索引。
4. 需要一个链接矩阵来记录聚类合并的历史信息，包括参与合并的两个聚类的索引，合并的两个聚类之间的距离，合并后新聚类中的样本点总数。
5. 可以使用最小堆，从而减少每次查找最近聚类对的时间复杂度。

## 执行步骤

1. 导入必要的库，确保所有环境要求满足。
2. 初始化距离矩阵和链接矩阵的函数。使用pdist计算X中所有样本点对的欧氏距离，squareform将距离数组转换成距离矩阵。初始化链接矩阵，其将存储聚类合并的历史信息。
3. 查找距离最近的两个聚类的函数。每一次迭代从最小堆中取出距离最小的元素（堆顶元素），取出的元素包含了两个聚类的索引和它们之间的距离，即当前最近的聚类对。
4. 执行层次聚类的函数。基于输入数据初始化距离矩阵和链接矩阵，并为每个样本点创建一个唯一的聚类索引。在每次迭代中，查找并合并距离最近的两个聚类，更新剩余聚类与这个新聚类之间的距离，并更新最小堆，更新链接矩阵和聚类索引。由于每次合并后聚类数量减少，因此需要从堆中删除涉及被合并聚类的距离信息，并将新计算的距离信息添加到堆中。当所有样本点的最终聚类标签生成后，算法完成。
5. 算法执行完毕后，输出轮廓系数以评估聚类效果，并显示聚类结果的树状图。
