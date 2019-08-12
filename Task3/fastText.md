### fastText

fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：

1、fastText在保持**高精度**的情况下加快了训练速度和测试速度

2、fastText**不需要预训练好的词向量**，fastText会自己训练词向量

3、fastText两个重要的优化：Hierarchical Softmax、N-gram



#### fastText模型架构

fastText模型架构和word2vec中的CBOW很相似，不同之处是**fastText预测标签**而**CBOW预测中间词**，模型架构类似但是模型任务不同。

word2vec将上下文关系转化为多分类任务，进而训练逻辑回归模型，这里的类别数量|V|词库大小。通常的文本数据中，词库少则数万，多则百万，在训练中直接训练多分类逻辑回归并不现实。word2vec提供两种针对大规模多分类问题的优化手段，negative sampling和hierarchical softmax。

**fastText最大的特点是模型简单，只有一层的隐层以及输出层**，因此训练速度非常快，在普通的CPU上可以实现分钟级别的训练，比深度模型的训练要快几个数量级。



参考文献

https://blog.csdn.net/feilong_csdn/article/details/88655927

https://www.jianshu.com/p/b1468218caf6









