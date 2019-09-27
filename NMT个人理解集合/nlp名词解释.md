## token

即句子中独立的单词，也称标记（token），包括标点

在数据中保存为某单词 <a href="https://www.codecogs.com/eqnedit.php?latex=v" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v" title="v" /></a> 在词典 <a href="https://www.codecogs.com/eqnedit.php?latex=dict" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dict" title="dict" /></a> 中的位置 <a href="https://www.codecogs.com/eqnedit.php?latex=dict[v]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dict[v]" title="dict[v]" /></a> ，类型为 <a href="https://www.codecogs.com/eqnedit.php?latex=int" target="_blank"><img src="https://latex.codecogs.com/gif.latex?int" title="int" /></a> 

## tokenization

标记化（tokenization）把句子分解成独立的token的过程。

假设有两句话：

s1：My name is Anny.

s2：I am a student.

那么把这两句话放在一起，经过token以后（假设参照某一个dictionary），可能会生成这样一个矩阵：

[[49, 31, 44,  2, 82, 43],
  [27, 68,  1, 75, 32, 55]]

其中，每个数字都是该位置的单词在dictionary中的位置，不同的dictionary中的位置不同。

## embedding

在embedding时，会同时传入一个字典dictionary，embedding函数会按照dictionary中的位置，扩展dim维度

假设一组句子token的维度是（255，23），那么经过embedding，生成的embeded token的维度就变成

（255，23，dim）。

其实，上面这组句子token的维度也可以看成是（255，23，1），这样可以视为对于255*23个单词，每个单词都由1个特征来描述。embedding的好处就是，在embedding之后，维度可以解释成255\*23个单词，每个词都由原先的一个特征描述，扩展成了dim个特征。这样，就可以从dim个特征去描述一个单词，使描述更加全面。

## dropout

也称“弃权”，是在训练的过程中，将部分神经元的权重或输出**随机**置零，使整个神经网络的神经元之间的依赖性降低，以降低 *过拟合* 的风险

参考链接：https://blog.csdn.net/program_developer/article/details/80737724

## padding

因为在训练的过程中，无论句子长短，都是按照统一的规格，比如288*14的维度来运行。那么如果一个句子的长度只有10，剩下的4个位置将使用0来填充。这个填充的过程就叫做padding。

## mask

mask过程是用一个只有0和1的矩阵M，和另一个矩阵V按元素相乘，那么M中元素为1的位置被保留，元素为0的位置就被置0，这样使得矩阵V中只有部分数值被利用，这个过程就被称作mask，矩阵M也被成为mask矩阵。

如果一个训练数据经过了上面的padding过程，那么可以把mask矩阵中，填充了0的部分置为0，其他部分置为1，这样就能区分哪些是真正的数据，哪些是填充数据。

使用mask矩阵是为了**让那些被mask掉的tensor不会被更新**。考虑一个tensor T的size(a,b)，同样大小的mask矩阵M，相乘后，在反向回传的时候在T对应mask为0的地方，0的梯度仍为0。因此不会被更新。

使用mask的场景：https://blog.csdn.net/weixin_37947156/article/details/83147294

## corpus

语料库。如中英平行语料LDC等等。

## ground truth

在机器翻译里，假设已经对齐的语料库中s1对应t1，那么这里t1就被称为**ground truth**。

在transformer里，ground truth作为六层Decoder结构的第一个decoder layer的输入。

## greedy search（贪心搜索）

在神经机器翻译RNN结构的decoder部分，我们知道单词是以时间步t一个一个推测出的。其实，在推测单词之前，会出现很多单词的候选项（candidate），而greedy search则是逐词选择概率最大的词。![nlp1](C:\Users\dragon\Desktop\mds\nlp1.png)

图片来源https://blog.csdn.net/qq_16234613/article/details/83012046

## beam search（集束搜索）

如果说greedy search是寻找单词级最优解，那么beam search则是寻找短语级最优解（窗口最优解）。

参考链接：https://blog.csdn.net/qq_16234613/article/details/83012046

## 标准化（scale）

**标准化（scale）改变数据的范围（range）**

将数据转为为特定范围的数据，比如（0，1）或者（0，100）

## 正则化（normalization） 

**正则化（normalization）改变数据的分布（distribution）**

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;\frac{x&space;-&space;\mathrm{E}[x]}{&space;\sqrt{\mathrm{Var}[x]&space;&plus;&space;\epsilon}}&space;*&space;\gamma&space;&plus;&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\frac{x&space;-&space;\mathrm{E}[x]}{&space;\sqrt{\mathrm{Var}[x]&space;&plus;&space;\epsilon}}&space;*&space;\gamma&space;&plus;&space;\beta" title="y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta" /></a> （pytorch里的正则化公式）

Normalization的目的就在于把你的数据转化为一个正态分布，从而进行下游的数据分析

```latex

```



