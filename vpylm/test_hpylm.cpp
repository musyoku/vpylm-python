#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <map>
#include <unordered_map> 
#include <stdio.h>
#include <wchar.h>
#include <locale>
#include "c_printf.h"
#include "node.h"
#include "hpylm.h"
#include "vocab.h"

using namespace std;

int main(int argc, char *argv[]){
	// 日本語周り
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	ios_base::sync_with_stdio(false);
	locale default_loc("ja_JP.UTF-8");
	locale::global(default_loc);
	locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
	wcout.imbue(ctype_default);
	wcin.imbue(ctype_default);
	vector<wstring> dataset;

	int ngram = 2;
	HPYLM* hpylm = new HPYLM(ngram);
	vector<id> token_ids;
	for(int i = 0;i < 5000;i++){
		token_ids.push_back(i);
	}
	int max_epoch = 50;
	for(int epoch = 0;epoch < max_epoch;epoch++){
		for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
			hpylm->add_customer_at_timestep(token_ids, token_t_index);
		}
	}
	c_printf("[n]%d\n", hpylm->get_max_depth(false));
	c_printf("[n]%d\n", hpylm->get_num_nodes());
	c_printf("[n]%d\n", hpylm->get_num_customers());
	for(int epoch = 0;epoch < max_epoch;epoch++){
		for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
			hpylm->remove_customer_at_timestep(token_ids, token_t_index);
		}
	}
	hpylm->sample_hyperparams();
	c_printf("[n]%d\n", hpylm->get_max_depth(false));
	c_printf("[n]%d\n", hpylm->get_num_nodes());
	c_printf("[n]%d\n", hpylm->get_num_customers());
}
