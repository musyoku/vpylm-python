#include <iostream>
#include <string>
#include <unordered_map> 
#include <boost/python.hpp>
#include "vpylm/c_printf.h"
#include "vpylm/node.h"
#include "vpylm/hpylm.h"
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
class PyHPYLM{
private:
	HPYLM* hpylm;

public:
	PyHPYLM(int ngram){
		hpylm = new HPYLM(ngram);
		c_printf("[n]%d-gram %s", ngram, "HPYLMを初期化しています ...\n");
	}
	// 基底分布 i.e. 単語（文字）0-gram確率
	// 1 / 単語数（文字数）でよい
	void set_g0(double g0){
		hpylm->_g0 = g0;
		c_printf("[n]%s%f\n", "G0 <- ", g0);
	}
	bool save(){
		c_printf("[n]%s", "HPYLMを保存しています ...\n");
		return hpylm->save();
	}
	bool load(){
		c_printf("[n]%s", "HPYLMを読み込んでいます ...\n");
		return hpylm->load();
	}
	void perform_gibbs_sampling(python::list &sentence, bool first_addition = false){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0;i < len;i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		for(int token_t_index = hpylm->ngram() - 1;token_t_index < token_ids.size();token_t_index++){
			if(first_addition == false){
				hpylm->remove_customer_at_timestep(token_ids, token_t_index);
			}
			hpylm->add_customer_at_timestep(token_ids, token_t_index);
		}
	}
	int get_max_depth(){
		return hpylm->get_max_depth();
	}
	int get_num_nodes(){
		return hpylm->get_num_nodes();
	}
	int get_num_customers(){
		return hpylm->get_num_customers();
	}
	python::list get_node_count_for_each_depth(){
		unordered_map<id, int> map;
		hpylm->count_node_of_each_depth(map);
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
		return list_from_vector(hpylm->_d_m);
	}
	python::list get_strength_parameters(){
		return list_from_vector(hpylm->_theta_m);
	}
	void sample_hyperparameters(){
		hpylm->sample_hyperparams();
	}
	id sample_next_token(python::list &sentence, id eos_id){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		return hpylm->sample_next_token(token_ids, eos_id);
	}
	double log_Pw(python::list &sentence){
		std::vector<id> token_ids;
		int len = python::len(sentence);
		for(int i = 0; i<len; i++) {
			token_ids.push_back(python::extract<id>(sentence[i]));
		}
		return hpylm->log_Pw(token_ids);
	}
};

BOOST_PYTHON_MODULE(hpylm){
	python::class_<PyHPYLM>("hpylm", python::init<int>())
	.def("set_g0", &PyHPYLM::set_g0)
	.def("perform_gibbs_sampling", &PyHPYLM::perform_gibbs_sampling)
	.def("get_max_depth", &PyHPYLM::get_max_depth)
	.def("get_num_nodes", &PyHPYLM::get_num_nodes)
	.def("get_num_customers", &PyHPYLM::get_num_customers)
	.def("get_discount_parameters", &PyHPYLM::get_discount_parameters)
	.def("get_strength_parameters", &PyHPYLM::get_strength_parameters)
	.def("sample_hyperparameters", &PyHPYLM::sample_hyperparameters)
	.def("log_Pw", &PyHPYLM::log_Pw)
	.def("sample_next_token", &PyHPYLM::sample_next_token)
	.def("get_node_count_for_each_depth", &PyHPYLM::get_node_count_for_each_depth)
	.def("save", &PyHPYLM::save)
	.def("load", &PyHPYLM::load);
}