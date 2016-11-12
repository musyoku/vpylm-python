#include <iostream>
#include <string>
#include <unordered_map> 
#include <boost/format.hpp>
#include <boost/python.hpp>
#include "core/c_printf.h"
#include "core/node.h"
#include "core/vpylm.h"
#include "core/vocab.h"

using namespace boost;

template<class T>
python::list list_from_vector(vector<T> &vec){  
	 python::list list;
	 typename vector<T>::const_iterator it;

	 for(it = vec.begin(); it != vec.end(); ++it)   {
		  list.append(*it);
	 }
	 return list;
}
// Pythonラッパー
class PyVPYLM{
private:
	VPYLM* vpylm;

public:
	PyVPYLM(){
		vpylm = new VPYLM();
		c_printf("[*]%s\n", "VPYLMを初期化しています ...");
	}
	// 基底分布 i.e. 単語（文字）0-gram確率
	// 1 / 単語数（文字数）でよい
	void set_g0(double g0){
		vpylm->_g0 = g0;
		c_printf("[*]%s\n", (boost::format("G0 <- %lf") % g0).str().c_str());
	}
	bool save(string filename){
		c_printf("[*]%s\n", "VPYLMを保存しています ...");
		return vpylm->save(filename);
	}
	bool load(string filename){
		c_printf("[*]%s\n", "VPYLMを読み込んでいます ...");
		return vpylm->load(filename);
	}
	python::list perform_gibbs_sampling(python::list &sentence, python::list &prev_orders){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0;i < len;i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		if(python::len(prev_orders) != token_ids.size()){
			c_printf("[r]%s [*]%s\n", "エラー:", "prev_ordersとword_idsの長さが違います.");
		}
		for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
			int order_t = python::extract<int>(prev_orders[token_t_index]);
			if(order_t != -1){
				vpylm->remove_customer_at_timestep(token_ids, token_t_index, order_t);
			}
		}				
		vector<int> new_order;
		for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
			int order_t = vpylm->sample_order_at_timestep(token_ids, token_t_index);
			vpylm->add_customer_at_timestep(token_ids, token_t_index, order_t);
			new_order.push_back(order_t);
		}
		return list_from_vector(new_order);
	}
	int get_max_depth(){
		return vpylm->get_max_depth();
	}
	int get_num_nodes(){
		return vpylm->get_num_nodes();
	}
	int get_num_customers(){
		return vpylm->get_num_customers();
	}
	python::list count_tokens_of_each_depth(){
		unordered_map<int, int> counts_by_depth;
		vpylm->count_tokens_of_each_depth(counts_by_depth);

		// ソート
		std::map<int, int> sorted_counts_by_depth(counts_by_depth.begin(), counts_by_depth.end());

		std::vector<int> counts;
		for(auto it = sorted_counts_by_depth.begin(); it != sorted_counts_by_depth.end(); ++it){
			counts.push_back(it->second);
		}
		return list_from_vector(counts);
	}
	python::list get_discount_parameters(){
		return list_from_vector(vpylm->_d_m);
	}
	python::list get_strength_parameters(){
		return list_from_vector(vpylm->_theta_m);
	}
	void sample_hyperparameters(){
		vpylm->sample_hyperparams();
	}
	python::list sample_orders(python::list &sentence){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0;i < len;i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		vector<int> new_order;
		for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
			int order_t = vpylm->sample_order_at_timestep(token_ids, token_t_index);
			vpylm->add_customer_at_timestep(token_ids, token_t_index, order_t);
			new_order.push_back(order_t);
		}
		return list_from_vector(new_order);
	}
	id sample_next_token(python::list &sentence, id eos_id){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		return vpylm->sample_next_token(token_ids, eos_id);
	}
	double log_Pw(python::list &sentence){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		return vpylm->log_Pw(token_ids);
	}
	python::list enumerate_phrases_at_depth(int depth){
		vector<vector<id>> phrase_vectors;
		vpylm->enumerate_phrases_at_depth(depth, phrase_vectors);
		python::list phrases;
		for(int i = 0;i < phrase_vectors.size();i++){
			phrases.append(list_from_vector(phrase_vectors[i]));
		}
		return phrases;
	}
};

BOOST_PYTHON_MODULE(vpylm){
	python::class_<PyVPYLM>("vpylm")
	.def("set_g0", &PyVPYLM::set_g0)
	.def("perform_gibbs_sampling", &PyVPYLM::perform_gibbs_sampling)
	.def("get_max_depth", &PyVPYLM::get_max_depth)
	.def("get_num_nodes", &PyVPYLM::get_num_nodes)
	.def("get_num_customers", &PyVPYLM::get_num_customers)
	.def("get_discount_parameters", &PyVPYLM::get_discount_parameters)
	.def("get_strength_parameters", &PyVPYLM::get_strength_parameters)
	.def("sample_hyperparameters", &PyVPYLM::sample_hyperparameters)
	.def("log_Pw", &PyVPYLM::log_Pw)
	.def("sample_next_token", &PyVPYLM::sample_next_token)
	.def("sample_orders", &PyVPYLM::sample_orders)
	.def("count_tokens_of_each_depth", &PyVPYLM::count_tokens_of_each_depth)
	.def("enumerate_phrases_at_depth", &PyVPYLM::enumerate_phrases_at_depth)
	.def("save", &PyVPYLM::save)
	.def("load", &PyVPYLM::load);
}