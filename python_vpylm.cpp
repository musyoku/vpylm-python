#include <iostream>
#include <string>
#include <unordered_map> 
#include <boost/python.hpp>
#include "vpylm/c_printf.h"
#include "vpylm/node.h"
#include "vpylm/vpylm.h"
#include "vpylm/vocab.h"

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
		c_printf("[n]%s", " VPYLMを初期化しています ...\n");
	}

	// 基底分布 i.e. 単語（文字）0-gram確率
	// 1 / 単語数（文字数）でよい
	void set_g0(double g0){
		this->vpylm->_g0 = g0;
		c_printf("[n]%s%f\n", " G0 <- ", g0);
	}

	bool save(){
		c_printf("[n]%s", " VPYLMを保存しています ...\n");
		return vpylm->save();
	}

	bool load(){
		c_printf("[n]%s", " VPYLMを読み込んでいます ...\n");
		return vpylm->load();
	}

	python::list perform_gibbs_sampling(python::list &sentence, python::list &prev_orders){
		std::vector<id> word_ids;
		int len = python::len(sentence);
		for(int i = 0;i < len;i++) {
			word_ids.push_back(python::extract<id>(sentence[i]));
		}

		if(python::len(prev_orders) != word_ids.size()){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " prev_ordersとword_idsの長さが違います\n");
		}

		for(int w_t_i = 0;w_t_i < word_ids.size();w_t_i++){
			int n_t = python::extract<int>(prev_orders[w_t_i]);
			if(n_t != -1){
				bool success = vpylm->remove(word_ids, w_t_i, n_t);
				if(success == false){
					c_printf("[R]%s", "エラー");
					c_printf("[n]%s", " 客を除去できませんでした\n");
				}
			}
		}				

		vector<int> new_order;
		for(int w_t_i = 0;w_t_i < word_ids.size();w_t_i++){
			int n_t = vpylm->sampleOrder(word_ids, w_t_i);
			vpylm->add(word_ids, w_t_i, n_t);
			new_order.push_back(n_t);
		}

		return list_from_vector(new_order);
	}

	int get_max_depth(){
		return vpylm->maxDepth();
	}

	int get_num_child_nodes(){
		return vpylm->numChildNodes();
	}

	int get_num_customers(){
		return vpylm->numCustomers();
	}

	python::list get_node_count_for_each_depth(){
		unordered_map<id, int> map;
		vpylm->countNodeForEachDepth(map);

		std::vector<int> counts;
		std::map<int, int> ordered(map.begin(), map.end());
		
		// 0-gram
		counts.push_back(0);

		for(auto it = ordered.begin(); it != ordered.end(); ++it){
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
		vpylm->sampleHyperParams();
	}

	python::list sample_orders(python::list &sentence){
		std::vector<id> word_ids;
		int len = python::len(sentence);
		for(int i = 0;i < len;i++) {
			word_ids.push_back(python::extract<id>(sentence[i]));
		}

		vector<int> new_order;
		for(int w_t_i = 0;w_t_i < word_ids.size();w_t_i++){
			int n_t = vpylm->sampleOrder(word_ids, w_t_i);
			vpylm->add(word_ids, w_t_i, n_t);
			new_order.push_back(n_t);
		}

		return list_from_vector(new_order);
	}

	id sample_next_word(python::list &sentence){
		std::vector<id> word_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			word_ids.push_back(python::extract<id>(sentence[i]));
		}
		return vpylm->sampleNextWord(word_ids);
	}

	double compute_log_Pw(python::list &sentence){
		std::vector<id> word_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			word_ids.push_back(python::extract<id>(sentence[i]));
		}
		return vpylm->log_Pw(word_ids);
	}
};

BOOST_PYTHON_MODULE(vpylm){
	python::class_<PyVPYLM>("vpylm")
	.def("set_g0", &PyVPYLM::set_g0)
	.def("perform_gibbs_sampling", &PyVPYLM::perform_gibbs_sampling)
	.def("get_max_depth", &PyVPYLM::get_max_depth)
	.def("get_num_child_nodes", &PyVPYLM::get_num_child_nodes)
	.def("get_num_customers", &PyVPYLM::get_num_customers)
	.def("get_discount_parameters", &PyVPYLM::get_discount_parameters)
	.def("get_strength_parameters", &PyVPYLM::get_strength_parameters)
	.def("sample_hyperparameters", &PyVPYLM::sample_hyperparameters)
	.def("compute_log_Pw", &PyVPYLM::compute_log_Pw)
	.def("sample_next_word", &PyVPYLM::sample_next_word)
	.def("sample_orders", &PyVPYLM::sample_orders)
	.def("get_node_count_for_each_depth", &PyVPYLM::get_node_count_for_each_depth)
	.def("save", &PyVPYLM::save)
	.def("load", &PyVPYLM::load);
}