# -*- coding: utf-8 -*-
import numpy as np
import math
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

max_epoch = 100
seed = 0
np.random.seed(seed)
indices = np.arange(n_data)

for epoch in xrange(1, max_epoch + 1):

	np.random.shuffle(indices)
	for train_step in xrange(n_data):
		index = indices[train_step]
		line = line_list[index]
		prev_order = prev_order_list[index]
		new_order = model.perform_gibbs_sampling(line, prev_order)
		prev_order_list[index] = new_order[:]

	model.sample_hyperparameters()
	print model.get_max_depth()
	print model.get_num_child_nodes()
	print model.get_num_customers()
	print model.get_discount_parameters()
	print model.get_strength_parameters()

	# パープレキシティを計算
	sum_log_Pw = 0
	for index in xrange(n_data):
		line = line_list[index]
		sum_log_Pw += model.compute_log_Pw(line) / len(line)
	vpylm_ppl = math.exp(-sum_log_Pw / n_data);
	print vpylm_ppl
