# -*- coding: utf-8 -*-
import numpy as np
import math, sys, time, pickle, os
import vpylm
import dataset

def generate_words(model):
	sentence = [dataset.word_to_id("<bos>")]
	for i in xrange(100):
		next_id = model.sample_next_word(sentence)
		if next_id == 0:
			break
		sentence.append(next_id)
	sentence.append(dataset.word_to_id("<eos>"))

	str = ""
	for i in xrange(1, len(sentence) - 1):
		word = dataset.id_to_word(sentence[i])
		str += word + " "
	print str

def train(model):
	# データの読み込み
	line_list, n_vocab, n_data = dataset.load("alice", split_by="word", include_whitespace=False)

	# 前回推定したn-gramオーダー
	if os.path.exists("prev_orders.dump"):
		with open("prev_orders.dump", "rb") as f:
			prev_order_list = pickle.load(f)
	else:
		prev_order_list = []
		for i, line in enumerate(line_list):
			prev_order = []
			for j in xrange(len(line)):
				prev_order.append(-1)
			prev_order_list.append(prev_order)

	model.set_g0(1.0 / n_vocab)

	max_epoch = 50
	seed = 0
	np.random.seed(seed)
	indices = np.arange(n_data)

	for epoch in xrange(1, max_epoch + 1):

		print "Epoch {}/{}".format(epoch, max_epoch)
		np.random.shuffle(indices)
		start_time = time.time()
		for train_step in xrange(n_data):
			index = indices[train_step]
			line = line_list[index]
			prev_order = prev_order_list[index]
			new_order = model.perform_gibbs_sampling(line, prev_order)
			prev_order_list[index] = new_order[:]

			if train_step % (n_data // 200) == 0 or train_step == n_data - 1:
				show_progress(train_step, n_data)

		model.sample_hyperparameters()

		# パープレキシティを計算
		sum_log_Pw = 0
		for index in xrange(n_data):
			line = line_list[index]
			sum_log_Pw += model.compute_log_Pw(line) / len(line)
		vpylm_ppl = math.exp(-sum_log_Pw / n_data);

		lines_per_sec = n_data / float(time.time() - start_time)
		print "{:.2f} lines / sec - {:.2f} ppl - {} depth - {} nodes - {} customers".format(lines_per_sec, vpylm_ppl, model.get_max_depth(), model.get_num_child_nodes(), model.get_num_customers())
		# print model.get_discount_parameters()
		# print model.get_strength_parameters()
	model.save()
	with open("prev_orders.dump", "wb") as f:
		pickle.dump(prev_order_list, f)

def show_progress(step, total):
	progress = step / float(total - 1)
	barWidth = 70;
	str = "["
	pos = barWidth * progress;
	for i in xrange(barWidth):
		if i < pos:
			str += "="
		elif i == pos:
			str += ">"
		else:
			str += " "
	ret = "\r"
	if step == total - 1:
		ret = "\n"
	sys.stdout.write("{}] {}%{}".format(str, int(progress * 100.0), ret))
	sys.stdout.flush()

if __name__ == "__main__":
	model = vpylm.vpylm()
	file_exists = model.load()
	if file_exists:
		print "{} depth - {} nodes - {} customers".format(model.get_max_depth(), model.get_num_child_nodes(), model.get_num_customers())
		
	train(model)
	for n in xrange(100):
		generate_words(model)