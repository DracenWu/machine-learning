# fairseq transformer训练中的一些问题

------

这两天看fairseq transformer的代码，并在服务器用transformer跑实验。今天遇到一些问题，和师兄进行了一些交流，记录下来。

另一篇梳理nlp中的一些英文名词的还在写，整理好再发布。

## transformer中的数据的流向和形式的变化？

1. 在训练前，数据基本都是以 <a href="https://www.codecogs.com/eqnedit.php?latex=batch\_size*src\_len" target="_blank"><img src="https://latex.codecogs.com/gif.latex?batch\_size*src\_len" title="batch\_size*src\_len" /></a> 的形式传入，即 维度为（句子数，单词数） 的一个矩阵。

   假设现在有3个句子：

   *s1: I am a student.*

   *s2: I like play basketball.*

   *s3: I have a dog.*

   那么，在输入之前，src数据就是这样的形式：

   |  I   |  am  |  a   |  student   |  .   |
   | :--: | :--: | :--: | :--------: | :--: |
   |  I   | like | play | basketball |  .   |
   |  I   | have |  a   |    dog     |  .   |

   也就是按一句一行，每一行都有src_len个单词。不过真正传入的不是单词，而是单词的token，即在词典中的位置。

   

2. 在传入模型之前，会对数据进行embedding操作。embedding简单来说就是对每个token进行以dim为维度的扩展。在《attention is all you need》中，dim即d_model = 512。如果把源数据视为一个长方形，那么经过embedding后，数据就变成了长方体，它的维度为 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 。

3. 接下来，经过positional embedding后（维度没变化），还需要进行一次transpose(0，1)的操作。**可能是为了后面multihead attention，但具体还没看到**，之后就进六层encode layer，然后输出。这中间没有变化。

4. 之后进入decoder的过程和encoder一样，只不过decoder的输入包括两部分：prev_output_token和encoder_out，而prev_output_token是tgt数据，下面会介绍。

5. 在经过decoder的6层layer之后，维度为  <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 。之后数据会经过decoder::output_layer()，映射成 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;len(dictionary)" title="\small batch\_size * src\_len * dim" /></a> 的维度。

6. 查dictionary，把token转变成单词，最后经过loss等，之后进入下轮数据循环。

## transformer::decoder部分运行完，是怎样变成一个个词的？

从transformer::decoder出来后，数据经过decoder::output_layer()，由原先的 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 映射成了 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;len(dictionary)" title="\small batch\_size * src\_len * dim" /></a> 。这样，对于每个词都能在词典dictionary中找到对应的位置，进而确定是哪个单词。

## transformer训练时，source和target是如何传入的？

假设有一对src和tgt，即src_token和tgt_token，那么，src_token将会传入encoder，然后经过encoder部分后，变成encoder_out 。而tgt_token则传入decoder的prev_output_tokens参数，进行计算。

由于transformer的并行性，所以会把一组句子以src_token和tgt_token的形式传入，这里也把这组数据称作一个batch，句子个数即batch_size。

## 在整个模型训练好之后，解码（predict）是个怎样的过程？

训练时是以维度为 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 传入，在解码时，首先还是以 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 的维度传入encoder，因为encoder的功能只是编码，所以仍然可以利用transformer的并行性。而decoder却不能像训练时一样一次输出一整个batch的单词，而要每次只输出一个单词。所以在传入decoder时，会按src_len循环，以 <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;batch\_size&space;*&space;src\_len&space;*&space;dim" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;batch\_size&space;*&space;1&space;*&space;dim" title="\small batch\_size * src\_len * dim" /></a> 的形式传入，这样才能使得decoder每次输出一个单词。**实际上是在第i步输出第1到第batch_size句的第i个词**。最终得出所有句子。
