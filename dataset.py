# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

vocab = {}
inv_vocab = {}

def load(dir, split_by="word", include_whitespace=False, bos_padding=1):
	fs = os.listdir(dir)
	print len(fs), "ファイルを読み込み中 ..."
	data = []
	vocab["<eos>"] = 1
	vocab["<bos>"] = 2
	vocab["<unk>"] = 3
	bos_id = vocab["<bos>"]
	for (id, word) in enumerate(vocab):
		inv_vocab[id] = word

	for fn in fs:
		if fn[-4:] == ".txt":
			liens = codecs.open("%s/%s" % (dir, fn), "r", "utf_8")	# BOMありならutf_8_sig　そうでないならutf_8
			for line in liens:
				line = line.replace("\n", "")
				line = line.replace("\r", "")
				ids = []
				for i in xrange(bos_padding):
					ids.append(bos_id)

				# 文字n-gramの場合
				if split_by == "char" or split_by == "character":
					for i in xrange(len(line)):
						word = line[i]
						if include_whitespace == False and word == " ":
							continue
						if word not in vocab:
							vocab[word] = len(vocab)
							inv_vocab[vocab[word]] = word
						ids.append(vocab[word])

				# 単語n-gramの場合
				elif split_by == "word":
					words = line.split(" ")
					for word in words:
						if word not in vocab:
							vocab[word] = len(vocab)
							inv_vocab[vocab[word]] = word
						ids.append(vocab[word])
						if include_whitespace:
							if " " not in vocab:
								vocab[" "] = len(vocab)
								inv_vocab[vocab[" "]] = " "
							ids.append(vocab[" "])

				else:
					raise NotImplementedError()

				ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]
				data.append(ids)
	n_vocab = len(vocab)
	n_data = len(data)
	print "文字種:", n_vocab
	print "行数:", n_data
	return data, n_vocab, n_data


def ids_to_sentence(ids, spacer=" "):
	sentence = ""
	for id in ids:
		sentence += id_to_word(id) + spacer
	return sentence

def id_to_word(id):
	return inv_vocab[id]

def word_to_id(word):
	return vocab[word]