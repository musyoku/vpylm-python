# -*- coding: utf-8 -*-
import numpy as np
import math, sys, time, pickle, os
import hpylm
import dataset

ngram = 3
model_filename = "model/python_hpylm.model"
trainer_filename = "model/python_hpylm.trainer"
model = hpylm.hpylm(ngram)
file_exists = model.load(model_filename)
if file_exists:
	print "HPYLMを読み込みました. {} depth - {} nodes - {} customers".format(model.get_max_depth(), model.get_num_nodes(), model.get_num_customers())

# データの読み込み
split_by = "word"
lines, n_vocab, n_data = dataset.load("alice", split_by=split_by, include_whitespace=False, bos_padding=ngram)

# 文章生成
def generate_words():
	eos_id = dataset.word_to_id("<eos>")
	bos_id = dataset.word_to_id("<bos>")
	context_token_ids = []
	for i in xrange(ngram):
		context_token_ids.append(dataset.word_to_id("<bos>"))
	for i in xrange(100):
		next_id = model.sample_next_token(context_token_ids, eos_id)
		if next_id == eos_id:
			break
		context_token_ids.append(next_id)
	context_token_ids.append(dataset.word_to_id("<eos>"))

	str = ""
	for i in xrange(1, len(context_token_ids) - 1):
		token_id = context_token_ids[i]
		if token_id == bos_id:
			continue
		word = dataset.id_to_word(token_id)
		str += word + (" " if split_by == "word" else "")
	print str

# n-gramオーダーのデータでの分布を可視化
def visualize_ngram_occurrences():
	counts = model.get_node_count_for_each_depth()
	max_count = max(counts)
	for depth, count in enumerate(counts):
		ngram = depth + 1
		print ngram, "#" * int(math.ceil(count / float(max_count) * 30)), count

# VPYLMの学習
def train():
	if os.path.exists(trainer_filename):
		with open(trainer_filename, "rb") as f:
			is_first_addition = pickle.load(f)
	else:
		is_first_addition = []
		for i, line in enumerate(lines):
			prev_order = []
			for j in xrange(len(line)):
				prev_order.append(-1)
			is_first_addition.append(True)

	model.set_g0(1.0 / n_vocab)

	max_epoch = 100
	seed = 0
	np.random.seed(seed)
	indices = np.arange(n_data)

	for epoch in xrange(1, max_epoch + 1):
		np.random.shuffle(indices)
		start_time = time.time()
		for train_step in xrange(n_data):
			index = indices[train_step]
			line = lines[index]
			new_order = model.perform_gibbs_sampling(line, is_first_addition[index])
			is_first_addition[index] = False

			if train_step % (n_data // 200) == 0 or train_step == n_data - 1:
				show_progress(train_step, n_data)

		model.sample_hyperparameters()

		# パープレキシティを計算
		sum_log_Pw = 0
		for index in xrange(n_data):
			line = lines[index]
			sum_log_Pw += model.log_Pw(line) / len(line)
		vpylm_ppl = math.exp(-sum_log_Pw / n_data);

		lines_per_sec = n_data / float(time.time() - start_time)
		print "Epoch {} / {} - {:.2f} lps - {:.2f} ppl - {} depth - {} nodes - {} customers".format(epoch, max_epoch, lines_per_sec, vpylm_ppl, model.get_max_depth(), model.get_num_nodes(), model.get_num_customers())
		# print model.get_discount_parameters()
		# print model.get_strength_parameters()

		if epoch % 100 == 0:
			model.save(model_filename)
			with open(trainer_filename, "wb") as f:
				pickle.dump(is_first_addition, f)

	model.save(model_filename)
	with open(trainer_filename, "wb") as f:
		pickle.dump(is_first_addition, f)

def show_progress(step, total):
	progress = step / float(total - 1)
	barWidth = 30;
	str = "["
	pos = int(barWidth * progress);
	for i in xrange(barWidth):
		if i < pos:
			str += "="
		elif i == pos:
			str += ">"
		else:
			str += " "
	sys.stdout.write("{}] {}%\r".format(str, int(progress * 100.0)))
	sys.stdout.flush()

def main():
	# train()
	for n in xrange(100):
		generate_words()
	visualize_ngram_occurrences()

if __name__ == "__main__":
	main()

