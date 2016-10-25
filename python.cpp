#include <iostream>
#include <string>
#include <boost/python.hpp>
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
		this->vpylm = new VPYLM();
	}

	// 基底分布 i.e. 単語（文字）ゼログラム確率
	// 1 / 単語数（文字数）でよい
	void set_g0(double g0){
		this->vpylm->_g0 = g0;
	}

	void load(string filename){
		cout << filename << endl;
	}

	python::list train(python::list &sentence, python::list &prev_order){
		std::vector<id> word_ids;
		int len = python::len(sentence);
		for(int i=0; i<len; i++) {
			word_ids.push_back(python::extract<id>(sentence[i]));
		}

		for(int w_t_i = 0;w_t_i < word_ids.size();w_t_i++){
			int n_t = python::extract<int>(prev_order[w_t_i]);
			if(n_t > 0){
				bool success = vpylm->remove(word_ids, w_t_i, n_t);
				if(success == false){
					printf("\x1b[41;97m");
					printf("WARNING");
					printf("\x1b[49;39m");
					printf(" Failed to remove a customer from VPYLM.\n");
				}
			}
		}				

		vector<int> new_order;
		for(int w_t_i = 0;w_t_i < word_ids.size();w_t_i++){
			int n_t = vpylm->sampleOrder(word_ids, w_t_i);
			cout << n_t << endl;
			vpylm->add(word_ids, w_t_i, n_t);
			new_order.push_back(n_t);
		}

		python::list new_order_list = list_from_vector(new_order);
		return new_order_list;
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
};

BOOST_PYTHON_MODULE(vpylm){
	python::class_<PyVPYLM>("vpylm")
	.def("set_g0", &PyVPYLM::set_g0)
	.def("train", &PyVPYLM::train)
	.def("get_max_depth", &PyVPYLM::get_max_depth)
	.def("get_num_child_nodes", &PyVPYLM::get_num_child_nodes)
	.def("get_num_customers", &PyVPYLM::get_num_customers)
	.def("load", &PyVPYLM::load);
}