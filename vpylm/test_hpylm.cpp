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
	hpylm->set_g0(1.0 / 10.0);
	vector<id> token_ids;
	for(int i = 0;i < 5000;i++){
		token_ids.push_back(i % 10);
	}
	int max_epoch = 50;
	for(int epoch = 0;epoch < max_epoch;epoch++){
		for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
			hpylm->add_customer_at_timestep(token_ids, token_t_index);
		}
	}
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
	hpylm->save("./");
	hpylm->load("./");
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
	for(int epoch = 0;epoch < max_epoch;epoch++){
		for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
			hpylm->remove_customer_at_timestep(token_ids, token_t_index);
		}
	}
	hpylm->sample_hyperparams();
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
}
