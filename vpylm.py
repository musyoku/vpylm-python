# -*- coding: utf-8 -*-
import numpy as np
import vpylm
import dataset

# データの読み込み
line_list, n_vocab, n_data = dataset.load("alice")

# 前回推定したn-gramオーダー
prev_order_list = []
for i, line in enumerate(line_list):
	prev_order = []
	for j in xrange(len(line)):
		prev_order.append(-1)
	prev_order_list.append(prev_order)

model = vpylm.vpylm()
model.set_g0(1.0 / n_vocab)
model.load("aiueo")

max_epoch = 100
train_per_epoch = n_data
seed = 0
np.random.seed(seed)

for epoch in xrange(1, max_epoch + 1):
	
	for train in xrange(train_per_epoch):
		index = np.random.randint(0, n_data)
		line = line_list[index]
		prev_order = prev_order_list[index]
		new_order = model.train(line, prev_order)
		prev_order_list[index] = new_order[:]

	model.sample_hyperparameters()
