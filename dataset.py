# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

vocab = {}
inv_vocab = {}

def load(dir):
	fs = os.listdir(dir)
	print len(fs), "ファイルを読み込み中 ..."
	data = []
	vocab["<eos>"] = 0
	vocab["<bos>"] = 1
	for fn in fs:
		if fn[-4:] == ".txt":
			liens = codecs.open("%s/%s" % (dir, fn), "r", "utf_8")	# BOMありならutf_8_sig　そうでないならutf_8
			for line in liens:
				line = line.replace("\n", "")
				line = line.replace("\r", "")
				ids = []
				for i in xrange(len(line)):
					word = line[i]
					if word not in vocab:
						vocab[word] = len(vocab)
						inv_vocab[vocab[word]] = word
					ids.append(vocab[word])
				ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]
				data.append(ids)
	n_vocab = len(vocab)
	n_data = len(data)
	print "文字種:", n_vocab
	print "行数:", n_data
	return data, n_vocab, n_data

def id_to_word(id):
	return inv_vocab[id]

def word_to_id(word):
	return vocab[word]