# -*- coding: utf-8 -*-
import numpy as np
import math, sys, time, pickle, os
import vpylm
import dataset

# データの読み込み
split_by = "word"
lines, n_vocab, n_data = dataset.load("alice", split_by=split_by, include_whitespace=False)

model_filename = "model/python_vpylm.model"
trainer_filename = "model/python_vpylm.trainer"
model = vpylm.vpylm()
file_exists = model.load(model_filename)
if file_exists:
	print "VPYLMを読み込みました."
	print "{} depth - {} nodes - {} customers".format(model.get_max_depth(), model.get_num_nodes(), model.get_num_customers())
	print "d:", model.get_discount_parameters()
	print "theta:", model.get_strength_parameters()

# 文章生成
def generate_words():
	eos_id = dataset.word_to_id("<eos>")
	bos_id = dataset.word_to_id("<bos>")
	context_token_ids = [bos_id]
	for i in xrange(100):
		next_id = model.sample_next_token(context_token_ids, eos_id)
		if next_id == eos_id:
			break
		context_token_ids.append(next_id)
	context_token_ids.append(eos_id)

	str = ""
	for i in xrange(1, len(context_token_ids) - 1):
		token_id = context_token_ids[i]
		if token_id == bos_id:
			continue
		word = dataset.id_to_word(token_id)
		str += word + (" " if split_by == "word" else "")
	print str

# 単語が生成されたn-gramオーダーをサンプリングして表示する
def visualize_orders():
	indices = np.arange(n_data)
	np.random.shuffle(indices)
	for i in xrange(50):
		index = indices[i]
		line = lines[index]
		orders = model.sample_orders(line)
		sentence_str = ""
		order_str = ""
		for i, id in enumerate(line[1:-1]):
			word = dataset.id_to_word(id)
			if split_by == "word":
				sentence_str += word + " "
				order_str += " " * (len(word) // 2) + str(orders[i + 1]) + " " * (len(word) - len(word) // 2)
			elif split_by == "char" or split_by == "character":
				sentence_str += word
				order_str += str(orders[i + 1])
		print sentence_str
		print order_str
		print "\n"

# n-gramオーダーのデータでの分布を可視化
def visualize_ngram_occurrences():
	print "推定された各オーダーのデータ全体に対する割合"
	counts = model.count_tokens_of_each_depth()
	max_count = max(counts)
	for depth, count in enumerate(counts):
		ngram = depth + 1
		print "{:2d}-gram".format(ngram), "#" * int(math.ceil(count / float(max_count) * 30)), count

def enumerate_phrases():
	phrases = model.enumerate_phrases_at_depth(6)
	for phrase in phrases:
		print dataset.ids_to_sentence(phrase)

# VPYLMの学習
def train():
	# 前回推定したn-gramオーダー
	if os.path.exists(trainer_filename):
		with open(trainer_filename, "rb") as f:
			prev_order_list = pickle.load(f)
	else:
		prev_order_list = []
		for i, line in enumerate(lines):
			prev_order = []
			for j in xrange(len(line)):
				prev_order.append(-1)
			prev_order_list.append(prev_order)

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
			prev_order = prev_order_list[index]
			new_order = model.perform_gibbs_sampling(line, prev_order)
			prev_order_list[index] = new_order[:]

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
				pickle.dump(prev_order_list, f)

	model.save(model_filename)
	with open(trainer_filename, "wb") as f:
		pickle.dump(prev_order_list, f)

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
	train()
	for n in xrange(100):
		generate_words()
	visualize_orders()
	visualize_ngram_occurrences()

if __name__ == "__main__":
	main()

