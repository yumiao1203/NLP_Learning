#### BERT

BERT模型进一步增加词向量模型泛化能力，充分描述字符级、词级、句子级甚至句间关系特征。

BERT的三个亮点Masked LM、transformer、sentence-level。

Masked Language Model（完形填空的意思）

除了MLM，引入next sentence prediction 任务联合训练**pair级别**的特征表示

#####bert的模型架构

bert uses a bidirectional Transformer

##### Pre-training 任务

不用传统的左向右或者右向左的LM来预训练 BERT，采用两个新的无监督的预测任务来实现预训练。

###### 1）任务一：Masked LM

15%的比例，在每个sequence中，随机的mask掉 WordPiece中的tokens；同时也只predict被mask掉的位置上的words

这个方法有2个缺点downsides:
 A：因为[MASK]字符在fine-tuning阶段根本不会存在，所以在pre-training阶段用[MASK]代替原来的word参与训练，会到只pre-training和fine-tuning阶段mismatch。
 解决办法是：在整体15%的随机中，有80%用[MASK]替换选中的word，10%用一个随机的word代替选中的word，剩下10%的比例保持选中的word不变

B：因为在每一轮训练（each batch）只有15%的tokens会被预测，pre-training阶段的模型收敛就需要迭代更多的steps（相比于单向的模型，预测所有的word被预测）

###### 2）任务二：Next Sentence Prediction

理解2个text sentence之间的relationship，这个relation是不能直接用LM模型学习到的。

构建有一个2分类任务binarized next sentence prediction task，训练数据可以从任何一个单语种的语料库中生成：对于AB两个sentence，50%的概率B就是实际在A后面的sentence，另外50%的概率B是从语料中随机选择的一个sentence。

最终这个任务的Accuracy达到97%~98%



##### Pre-training过程

数据：BooksCorpus（800M words） + English Wikipedia（2500M words）

Wikipedia的数据只取text段落，忽略列表、表格等

从语料中，抽样2个spans作为sentence，第一个sentence作为A Embeding，第二个作为B Embeding；50%的概率B是实际的sentence，另外50% B是随机sentence.

抽样时保证A+B的长度<512 tokens

LM masking：在用wordpiece 分词（tokenization）后，均匀按15%的比例（rate）mask掉其中的tokens。没有任何特殊的操作。

batchsize = 256 sequence（256*512tokens = 128,000 tokens/batch）

steps=1,000,000 （40 epochs / 3.3 billion word corpus）

Adam lr=1e-4, beta1=0.9 beta2=0.999

L2 weight decay = 0.01

lr adj step = 10,000 steps, linear decay

dropout p=0.1 all layers

activation = **gelu** （不是relu）

Loss = mean masked LM Likelihood + mean next sentence prediction likelihood



##### 什么是Transformer

Transformer只用encoder-decoder和attention机制，最大的优点是高效地并行化。

在encoder中

1、Input 经过 embedding 后，要做 positional encodings

2、然后是 Multi-head attention

3、再经过 position-wise Feed Forward

4、每个子层之间有**残差连接**。

在 Decoder 中，

1、如上图所示，也有 positional encodings，Multi-head attention 和 FFN，子层之间也要做残差连接，

2、但比 encoder 多了一个 **Masked Multi-head attention**，

3、最后要经过 Linear 和 softmax 输出概率。



##### GELU激活函数（Gaussian error linear units，高斯误差线性单元）

$$GELU(x) = xP(X<x)=x\phi(x)$$

这里$\phi(x)$是正态分布的概率函数，可以使用标准的正态分布$N(0,1)$



在激活函数领域，大家公式的鄙视链应该是：Gelu > Elus > Relu > Sigmoid 

sigmoid容易饱和，Elus与Relu缺乏随机因素





参考

http://jalammar.github.io/illustrated-transformer/

https://www.csdn.net/link?target_url=https%3A%2F%2Farxiv.org%2Fabs%2F1606.08415&id=86510622&token=5394c716249b3659899b5ff51b54e4fb

https://www.cnblogs.com/rucwxb/p/10277217.html