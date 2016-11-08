#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <map>
#include <unordered_map> 
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <stdio.h>
#include <wchar.h>
#include <locale>
#include "vpylm/c_printf.h"
#include "vpylm/node.h"
#include "vpylm/hpylm.h"
#include "vpylm/vocab.h"

using namespace std;

vector<wstring> split(const wstring &s, char delim){
    vector<wstring> elems;
    wstring item;
    for(char ch: s){
        if (ch == delim){
            if (!item.empty())
                elems.push_back(item);
            item.clear();
        }
        else{
            item += ch;
        }
    }
    if (!item.empty())
        elems.push_back(item);
    return elems;
}

// スペースで分割する
Vocab* load_words_in_textfile(string &filename, vector<vector<id>> &dataset, int ngram){
	wifstream ifs(filename.c_str());
	wstring str;
	if (ifs.fail()){
		c_printf("[R]%s", "エラー");
		c_printf("[n] %s%s\n", filename.c_str(), "を開けません.");
		return NULL;
	}
	Vocab* vocab = new Vocab();
	id bos_id = vocab->add_string(L"<bos>");
	id eos_id = vocab->add_string(L"<eos>");
	id unk_id = vocab->add_string(L"<unk>");
	while (getline(ifs, str) && !str.empty()){
		vector<wstring> words = split(str, ' ');
		vector<id> token_ids;
		// HPYLMでは深さが固定なので先頭にダミーを挿入する
		for(int i = 0;i < ngram;i++){
			token_ids.push_back(bos_id);
		}
		for(auto word: words){
			if(word.size() == 0){
				continue;
			}
			id token_id = vocab->add_string(word);
			token_ids.push_back(token_id);
		}
		token_ids.push_back(eos_id);
		if(token_ids.size() > 0){
			dataset.push_back(token_ids);
		}
	}
	cout << filename << "を読み込みました（" << dataset.size() << "行）" << endl;
	return vocab;
}

// 文字n-gramの学習
Vocab* load_characters_in_textfile(string &filename, vector<vector<id>> &dataset, int ngram){
	wifstream ifs(filename.c_str());
	wstring str;
	if (ifs.fail()){
		c_printf("[R]%s", "エラー");
		c_printf("[n] %s%s\n", filename.c_str(), "を開けません.");
		return NULL;
	}
	Vocab* vocab = new Vocab();
	id bos_id = vocab->add_string(L"<bos>");
	id eos_id = vocab->add_string(L"<eos>");
	id unk_id = vocab->add_string(L"<unk>");
	while (getline(ifs, str) && !str.empty()){
		vector<id> token_ids;
		// HPYLMでは深さが固定なので先頭にダミーを挿入する
		for(int i = 0;i < ngram;i++){
			token_ids.push_back(bos_id);
		}
		if(str.size() == 0){
			continue;
		}
		for(int i = 0;i < str.size();i++){
			id token_id = vocab->add_string(wstring(str.begin() + i, str.begin() + i + 1));
			token_ids.push_back(token_id);
		}
		token_ids.push_back(eos_id);
		if(token_ids.size() > 0){
			dataset.push_back(token_ids);
		}
	}
	cout << filename << "を読み込みました（" << dataset.size() << "行）" << endl;
	return vocab;
}

void show_progress(int step, int total){
	double progress = step / (double)(total - 1);
	int barWidth = 30;

	cout << "[";
	int pos = barWidth * progress;
	for(int i = 0; i < barWidth; ++i){
		if (i < pos) cout << "=";
		else if (i == pos) cout << ">";
		else cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " %\r";
	cout.flush();
}

void generate_words(Vocab* vocab, vector<vector<id>> &dataset, wstring spacer){
	string hpylm_filename = "model/hpylm.model";
	string vocab_filename = "model/hpylm.vocab";

	HPYLM* hpylm = new HPYLM();
	hpylm->load(hpylm_filename);
	vocab->load(vocab_filename);

	int num_sample = 50;
	int max_length = 400;
	id bos_id = vocab->string_to_token_id(L"<bos>");
	id eos_id = vocab->string_to_token_id(L"<eos>");
	vector<id> token_ids;
	for(int s = 0;s < num_sample;s++){
		token_ids.clear();
		for(int i = 0;i < hpylm->ngram();i++){
			token_ids.push_back(bos_id);
		}
		for(int i = 0;i < max_length;i++){
			id token_id = hpylm->sample_next_token(token_ids, eos_id);
			token_ids.push_back(token_id);
			if(token_id == eos_id){
				break;
			}
		}
		for(auto token_id: token_ids){
			if(token_id == bos_id){
				continue;
			}
			if(token_id == eos_id){
				continue;
			}
			wstring word = vocab->token_id_to_string(token_id);
			wcout << word << spacer;
		}
		cout << endl;
	}
}

void train(Vocab* vocab, vector<vector<id>> &dataset, int ngram){
	string hpylm_filename = "model/hpylm.model";
	string vocab_filename = "model/hpylm.vocab";
	string trainer_filename = "model/hpylm.trainer";

	vector<int> rand_indices;
	vector<bool> is_first_addition;
	for(int i = 0;i < dataset.size();i++){
		rand_indices.push_back(i);
		is_first_addition.push_back(true);
	}

	cout << ngram << "-gram HPYLMを初期化しています ..." << endl;
	HPYLM* hpylm = new HPYLM(ngram);
	int num_chars = vocab->num_tokens();
	hpylm->set_g0(1.0 / num_chars);
	cout << "g0 <- " << 1.0 / num_chars << endl;

	hpylm->load(hpylm_filename);
	vocab->load(vocab_filename);
	std::ifstream ifs(trainer_filename);
	if(ifs.good()){
		boost::archive::binary_iarchive iarchive(ifs);
		iarchive >> is_first_addition;
	}

	int max_epoch = 100;
	int num_data = dataset.size();

	cout << "HPYLMを学習しています ..." << endl;
	for(int epoch = 1;epoch <= max_epoch;epoch++){
		// printf("Epoch %d / %d", epoch, max_epoch);
		auto start_time = chrono::system_clock::now();
		random_shuffle(rand_indices.begin(), rand_indices.end());

		for(int step = 0;step < num_data;step++){
			show_progress(step, num_data);
			int data_index = rand_indices[step];
			vector<id> token_ids = dataset[data_index];

			for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
				if(is_first_addition[data_index] == false){
					hpylm->remove_customer_at_timestep(token_ids, token_t_index);
				}
				hpylm->add_customer_at_timestep(token_ids, token_t_index);
			}
			is_first_addition[data_index] = false;
		}

		hpylm->sample_hyperparams();

		auto end_time = chrono::system_clock::now();
		auto duration = end_time - start_time;
		auto msec = chrono::duration_cast<chrono::milliseconds>(duration).count();

		// パープレキシティ
		double ppl = 0;
		for(int step = 0;step < num_data;step++){
			vector<id> token_ids = dataset[step];
			double log_p = hpylm->log2_Pw(token_ids) / token_ids.size();
			ppl += log_p;
		}
		ppl = exp(-ppl / num_data);
		printf("Epoch %d / %d - %.1f fps - %.3f ppl\n", epoch, max_epoch, (double)num_data / msec * 1000.0, ppl);
		if(epoch % 10 == 0){
			hpylm->save(hpylm_filename);
			vocab->save(vocab_filename);
			std::ofstream ofs(trainer_filename);
			boost::archive::binary_oarchive oarchive(ofs);
			oarchive << is_first_addition;
		}
	}

	hpylm->save(hpylm_filename);
	vocab->save(vocab_filename);
	std::ofstream ofs(trainer_filename);
	boost::archive::binary_oarchive oarchive(ofs);
	oarchive << is_first_addition;

	// <!-- デバッグ用
	//客を全て削除した時に客数が本当に0になるかを確認する場合
	// for(int step = 0;step < num_data;step++){
	// 	int data_index = rand_indices[step];
	// 	vector<id> token_ids = dataset[data_index];
	// 	for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
	// 		hpylm->remove_customer_at_timestep(token_ids, token_t_index);
	// 	}
	// }
	//  -->

	cout << hpylm->get_max_depth() << endl;
	cout << hpylm->get_num_nodes() << endl;
	cout << hpylm->get_num_customers() << endl;
	cout << hpylm->get_sum_stop_counts() << endl;
	cout << hpylm->get_sum_pass_counts() << endl;
}

int main(int argc, char *argv[]){
	// 日本語周り
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	ios_base::sync_with_stdio(false);
	locale default_loc("ja_JP.UTF-8");
	locale::global(default_loc);
	locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
	wcout.imbue(ctype_default);
	wcin.imbue(ctype_default);

	string text_filename;
	int ngram = 3;
	cout << "num args = " << argc << endl;
	if(argc % 2 != 1){
		c_printf("[R]%s", "エラー");
		c_printf("[n] %s\n", "テキストファイルを指定してください. -t example.txt");
		exit(1);
	}else{
		for(int i = 0; i < argc; i++){
			cout << i << "-th args = " << argv[i] << endl; 
			if (string(argv[i]) == "-t" || string(argv[i]) == "--text"){
				if(i + 1 >= argc){
					c_printf("[R]%s", "エラー");
					cout << "不正なコマンドライン引数です. " << string(argv[i]) << endl;
					exit(1);
				}
				text_filename = string(argv[i + 1]);
			}
			else if (string(argv[i]) == "-n" || string(argv[i]) == "--ngram"){
				if(i + 1 >= argc){
					c_printf("[R]%s", "エラー");
					cout << "不正なコマンドライン引数です. " << string(argv[i]) << endl;
					exit(1);
				}
				ngram = atoi(argv[i + 1]);
			}
		}
	}
	vector<vector<id>> dataset;
	Vocab* vocab = load_characters_in_textfile(text_filename, dataset, ngram);
	// train(vocab, dataset, ngram);
	generate_words(vocab, dataset, L"");
	return 0;
}
