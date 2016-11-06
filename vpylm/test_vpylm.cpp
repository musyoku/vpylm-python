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
#include "vpylm.h"
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

	VPYLM* vpylm = new VPYLM();
	vpylm->set_g0(1.0 / 10.0);
	vector<id> token_ids;
	for(int i = 0;i < 5000;i++){
		token_ids.push_back(i % 10);
	}
	vector<int> prev_orders(5000, -1);
	int max_epoch = 500;
	for(int epoch = 0;epoch < max_epoch;epoch++){
		for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
			int prev_order_t = prev_orders[token_t_index];
			if(prev_order_t != -1){
				vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_order_t);
			}
			int order_t = vpylm->sample_order_at_timestep(token_ids, token_t_index);
			vpylm->add_customer_at_timestep(token_ids, token_t_index, order_t);
			prev_orders[token_t_index] = order_t;
		}
	}
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
	vpylm->save("./");
	vpylm->load("./");
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
	for(int epoch = 0;epoch < 1;epoch++){
		for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
			int prev_order_t = prev_orders[token_t_index];
			if(prev_order_t != -1){
				vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_order_t);
			}
		}
	}
	vpylm->sample_hyperparams();
	printf("depth: %d\n", vpylm->get_max_depth(false));
	printf("# of nodes: %d\n", vpylm->get_num_nodes());
	printf("# of customers: %d\n", vpylm->get_num_customers());
	printf("# of tables: %d\n", vpylm->get_num_tables());
	printf("stop count: %d\n", vpylm->get_sum_stop_counts());
	printf("pass count: %d\n", vpylm->get_sum_pass_counts());
}
