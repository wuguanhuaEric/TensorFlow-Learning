# -*- coding;UTF -8 -*-

"""
神经网络模型相关
RNN-LSTM 循环神经网络
"""

# 神经网络的模型

import tensorflow as tf

# 神经网络的模型
class Model(object):
    # 构造函数
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout = 0.5, init_scale = 0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size =  input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        with tf.device("/cpu:0"):
            # 创建词向量 (Word Embedding), Embedding 表示 Dense Vector(密集向量)
            # 词向量本质上是一种单词聚类 (Clustering) 的方法
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            # embedding_lookup 返回词向量
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)
        # 如果是训练时并且dropout 率小于 1， 使输入经过一个 Dropout 层
        # Dropout 防止过拟合
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # 状态 (state)的存储和提取
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入
        # 一个是前一时刻 LSTM 的输出 h(t - 1)
        # 一个是前一时刻的单元状态 C(t - 1)
        # C 和 h 是用于构建之后的 tf.contrib.rnn.LSTMStateTuple
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size,
                self.hidden_size])
        # 每一层的状态
        state_per_layer_list = tf.unstack(self.init_state, axis = 0)

        # 初始的状态（包含前一时刻 LSTM 的输出 h(t - 1) 和前一时刻的单元状态 C(t - 1)）,用于
        # 之后的 dynamic_rnn
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[index][0],
                                state_per_layer_list[index][1]) for index in range(num_layers)])

        # 创建一个 LSTM 层，其中的神经元数目是 hidden_size 个（默认 650个）
        cell = tf.contrib.rnn.LSTMCell(hidden_size)

        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = dropout)

        # 如果 LSTM 的层数大于 1，则总是创建 num layers 个 LSTM 层
        # 并将所有的 LSTM 层包装进 MultiRNNCell 这样的序列化层架模型中
        # state_is_tuple = True 表示接受 LSTMStateTuple 形式的输入状态
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range (num_layers)],
                    state_is_tuple = True)

        # dynamic_rnn （动态 RNN）可以让不同迭代传入的 Batch 可以是长度不同的数据
        # 但同一次迭代中一个 Batch 内部的所有数据长度仍然是固定的
        # dynamic_rnn 能更好处理padding (补零)的情况，节约计算资源
        # 返回两个变量：
        # 第一个是一个 Batch 里在时间维度 (默认是 35) 上展开的所有 LSTM 单元的输出， 形状
        # 默认为 [20, 35, 650], 之后会经过扁平层处理
        # 第二个是最终的 state(状态)， 包含当前时刻 LSTM 的输出 h(t) 和当前时刻的单元状态 C(t)
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state = rnn_tuple_state)

        # 扁平化处理，改变输出形状为 (batch_size * num_steps, hidden_size), 形状默认为 [700, 650]
        output = tf.reshape(output, [-1, hidden_size]) # -1 表示自动推倒维度大小

        # Softmax 的权重(Weight)
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))

        # Softmax 的偏置 (Bias)
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale ))

        # logits 是 Logistic Regression （用于分类）模型（线性方程 ：y = W * x + b）计算的结果（分值）
        # 这个 logits (分值)之后会用 Softmax 来转成百分比概率
        # output 是输入 (x) , softmax_w 是权重（W），softmax_b 是偏置 （b）

        # 返回 W * x + b 结果
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # 将 logits 转化为三维的 Tensor,为了sequence loss 的计算
        # 形状默认为 [20, 35, 10000]

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # 计算 logits 的序列的交叉熵 (Cross-Entropy) 的损失 (loss)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, # 形状默认为 [20, 35, 10000]
            self.input_obj.targets, # 期望输出， 形状为 [20, 35]
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps = False,
            average_across_batch = True
        )

        # 更新代价 (cost)
        self.cost = tf.reduce_sum(loss)

        # Softmax 算出来的概率
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size])) # logits 形状为 [700, 10000]

        # 取最大概率的那个值作为预测
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)

        # 预测值和真实值(目标)对比
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))

        # 计算预测的精度
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 如果是测试，则直接退出
        if not is_training:
            return
        # 学习率。 trainable = False 表示 “不可被训练”
        self.learning_rate = tf.Variable(0.0, trainable=False)

        # 返回所有可被训练 (trainable = True, 如果不设定 trainable = False, 默认的
        # Variable 都是可以被训练的)

        tvars = tf.trainable_variables()

        # tf.clip_by_global_norm (实现 Gradient Clipping (梯度裁剪) 为了防止梯度爆炸)
        # tf.gradient 计算self.cost 对于 tvars 的梯度 (求导)， 返回一个梯度的列表
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        # 优化器用 GradientDescentOptimizer (梯度下降优化器)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # apply_gradients (应用梯度)将之前用 (Gradient Clipping) 梯度裁剪过的梯度 应用到
        # 可被训练的变量上去，做梯度下降
        # apply_gradient 其实是 minimize 方法里面的第二步，第一步是计算梯度
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step()
        )

        # 用于更新学习率
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    # 更新学习率
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict = {self.new_lr: lr_value})
























