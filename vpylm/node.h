#ifndef _node_
#define _node_
#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include "c_printf.h"
#include "sampler.h"
#include "const.h"
#include "vocab.h"

using namespace std;

class Node{
private:
	// 新しいテーブルを生成しそこに客を追加
	bool add_customer_to_empty_arrangement(id token_id, double parent_Pw, vector<double> &d_m, vector<double> &theta_m){
		if(_arrangement.find(token_id) == _arrangement.end()){
			vector<int> tables = {1};
			_arrangement[token_id] = tables;
			_num_customers++;
			_num_tables++;
			if(_parent != NULL){
				// 親に代理客を送る
				_parent->add_customer(token_id, parent_Pw, d_m, theta_m, false);
			}
			return true;
		}
		c_printf("[R]%s", "エラー");
		c_printf("[n]%s", " 客を追加できません. _arrangement.find(token_id) == _arrangement.end()\n");
		return false;
	}
	// 客をテーブルに追加
	bool add_customer_to_table(id token_id, int table_k, double parent_Pw, vector<double> &d_m, vector<double> &theta_m){
		if(_arrangement.find(token_id) == _arrangement.end()){
			return add_customer_to_empty_arrangement(token_id, parent_Pw, d_m, theta_m);
		}
		if(table_k < _arrangement[token_id].size()){
			_arrangement[token_id][table_k]++;
			_num_customers++;
			return true;
		}
		c_printf("[R]%s", "エラー");
		c_printf("[n]%s", " 客を追加できません. table_k < _arrangement[token_id].size()\n");
		return false;
	}
	bool add_customer_to_new_table(id token_id, double parent_Pw, vector<double> &d_m, vector<double> &theta_m){
		_arrangement[token_id].push_back(1);
		_num_tables++;
		_num_customers++;
		if(_parent != NULL){
			_parent->add_customer(token_id, parent_Pw, d_m, theta_m, false);
		}
		return true;
	}
	bool remove_customer_from_table(id token_id, int table_k){
		if(_arrangement.find(token_id) == _arrangement.end()){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 客を除去できません. _arrangement.find(token_id) == _arrangement.end()\n");
			return false;
		}
		if(table_k < _arrangement[token_id].size()){
			vector<int> &tables = _arrangement[token_id];
			tables[table_k]--;
			_num_customers--;
			if(tables[table_k] < 0){
				c_printf("[R]%s", "エラー");
				c_printf("[n]%s", " 客の管理に不具合があります. tables[table_k] < 0\n");
				return false;
			}
			if(tables[table_k] == 0){
				if(_parent != NULL){
					_parent->remove_customer(token_id, false);
				}
				tables.erase(tables.begin() + table_k);
				_num_tables--;
				if(tables.size() == 0){
					_arrangement.erase(token_id);
				}
			}
			return true;
		}
		c_printf("[R]%s", "エラー");
		c_printf("[n]%s", " 客を除去できません. table_k < _arrangement[token_id].size()\n");
		return false;
	}
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		static_cast<void>(version); // No use
		archive & _children;
		archive & _arrangement;
		archive & _num_tables;
		archive & _num_customers;
		archive & _parent;
		archive & _stop_count;
		archive & _pass_count;
		archive & _token_id;
		archive & _depth;
		archive & _identifier;
		archive & _auto_increment;
	}
public:
	static id _auto_increment;						// identifier用 VPYLMとは無関係
	unordered_map<id, Node*> _children;				// 子の文脈木
	unordered_map<id, vector<int> > _arrangement;	// 客の配置 vector<int>のk番目の要素がテーブルkの客数を表す
	int _num_tables;								// 総テーブル数
	int _num_customers;								// 客の総数
	Node* _parent;									// 親ノード
	int _stop_count;								// 停止回数
	int _pass_count;								// 通過回数
	id _token_id;									// 単語ID　文字ID
	int _depth;										// ノードの深さ　rootが0
	id _identifier;									// 識別用　特別な意味は無い VPYLMとは無関係

	Node(id token_id = 0){
		_num_tables = 0;
		_num_customers = 0;
		_stop_count = 0;
		_pass_count = 0;
		_identifier = _auto_increment;
		_auto_increment++;
		_token_id = token_id;
		_parent = NULL;
	}
	bool parent_exists(){
		return !(_parent == NULL);
	}
	bool child_exists(int id){
		return !(_children.find(id) == _children.end());
	}
	bool need_to_remove_from_parent(){
		if(_children.size() == 0 and _arrangement.size() == 0){
			return true;
		}
		return false;
	}
	int get_num_tables_serving_word(id token_id){
		if(_arrangement.find(token_id) == _arrangement.end()){
			return 0;
		}
		return _arrangement[token_id].size();
	}
	int get_num_customers_eating_word(id token_id){
		if(_arrangement.find(token_id) == _arrangement.end()){
			return 0;
		}
		vector<int> &tables = _arrangement[token_id];
		int sum = 0;
		for(int i = 0;i < tables.size();i++){
			sum += tables[i];
		}
		return sum;
	}
	Node* find_child_node(id token_id, bool generate_if_not_exist = false){
		auto itr = _children.find(token_id);
		if (itr != _children.end()) {
			return itr->second;
		}
		if(generate_if_not_exist == false){
			return NULL;
		}
		Node* child = new Node(token_id);
		child->_parent = this;
		child->_depth = _depth + 1;
		_children[token_id] = child;
		return child;
	}

	void add_customer(id token_id, double parent_Pw, vector<double> &d_m, vector<double> &theta_m, bool update_n = true){
		init_hyperparameters_at_depth_if_needed(_depth, d_m, theta_m);
		double d_u = d_m[_depth];
		double theta_u = theta_m[_depth];

		if(_arrangement.find(token_id) == _arrangement.end()){
			add_customer_to_empty_arrangement(token_id, parent_Pw, d_m, theta_m);
			if(update_n == true){
				increment_stop_count();
			}
		}else{
			vector<int> &tables = _arrangement[token_id];
			double rand_max = 0.0;
			for(int k = 0;k < tables.size();k++){
				rand_max += std::max(0.0, tables[k] - d_u);
			}
			double t_u = (double)_num_tables;
			rand_max += (theta_u + d_u * t_u) * parent_Pw;

			uniform_real_distribution<double> rand(0, rand_max);
			double r = rand(Sampler::mt);

			double sum = 0.0;
			for(int k = 0;k < tables.size();k++){
				sum += std::max(0.0, tables[k] - d_u);
				if(r <= sum){
					if(!add_customer_to_table(token_id, k, parent_Pw, d_m, theta_m)){
						printf("\x1b[41;97m");
						printf("WARNING");
						printf("\x1b[49;39m");
						printf(" Failed to add a customer\n");
					}
					if(update_n){
						increment_stop_count();
					}
					return;
				}
			}

			add_customer_to_new_table(token_id, parent_Pw, d_m, theta_m);
			if(update_n){
				increment_stop_count();
			}
		}
	}

	bool remove_customer(id token_id, bool update_n = true){
		if(_arrangement.find(token_id) == _arrangement.end()){
			return false;
		}

		vector<int> &tables = _arrangement[token_id];
		double max = 0.0;
		for(int k = 0;k < tables.size();k++){
			max += tables[k];
		}
		
		uniform_real_distribution<double> rand(0, max);
		double r = rand(Sampler::mt);

		double sum = 0.0;
		for(int k = 0;k < tables.size();k++){
			sum += tables[k];
			if(r <= sum){
				if(!remove_customer_from_table(token_id, k)){
					printf("\x1b[41;97m");
					printf("WARNING");
					printf("\x1b[49;39m");
					printf(" Failed to remove a customer.\n");
				}
				if(update_n == true){
					decrement_stop_count();
				}
				return true;
			}
		}
		if(!remove_customer_from_table(token_id, tables.size() - 1)){
			printf("\x1b[41;97m");
			printf("WARNING");
			printf("\x1b[49;39m");
			printf(" Failed to remove a customer.\n");
		}
		if(update_n == true){
			decrement_stop_count();
		}
		return true;
	}

	double Pw(id token_id, double g0, vector<double> &d_m, vector<double> &theta_m){
		init_hyperparameters_at_depth_if_needed(_depth, d_m, theta_m);

		double d_u = d_m[_depth];
		double theta_u = theta_m[_depth];
		double t_u = (double)_num_tables;
		double c_u = (double)_num_customers;
		double mult = (theta_u + d_u * t_u) / (theta_u + c_u);
		if(_arrangement.find(token_id) == _arrangement.end()){
			if(_parent != NULL){
				return _parent->Pw(token_id, g0, d_m, theta_m) * mult;
			}
			return g0 * mult;
		}
		if(_parent != NULL){
			vector<int> &tables = _arrangement[token_id];
			double c_uw = std::accumulate(tables.begin(), tables.end(), 0);
			double t_uw = tables.size();

			double first_coeff = (c_uw - d_u * t_uw) / (theta_u + c_u);
			if(first_coeff < 0){
				first_coeff = 0;
			}
			double second_coeff = (theta_u + d_u * t_u) / (theta_u + c_u);

			double parent_p = _parent->Pw(token_id, g0, d_m, theta_m);
			double p = first_coeff + second_coeff * parent_p;
			return p;
		}

		vector<int> &tables = _arrangement[token_id];
		double c_uw = std::accumulate(tables.begin(), tables.end(), 0);
		double t_uw = tables.size();

		double first_coeff = (c_uw - d_u * t_uw) / (theta_u + c_u);
		if(first_coeff < 0){
			first_coeff = 0;
		}
		double second_coeff = (theta_u + d_u * t_u) / (theta_u + c_u);

		double p = first_coeff + second_coeff * g0;
		return p;

	}
	double stop_probability(double beta_stop, double beta_pass){
		double p = (_stop_count + beta_stop) / (_stop_count + _pass_count + beta_stop + beta_pass);
		if(_parent != NULL){
			p *= _parent->pass_probability(beta_stop, beta_pass);
		}
		return p;
	}
	double pass_probability(double beta_stop, double beta_pass){
		double p = (_pass_count + beta_pass) / (_stop_count + _pass_count + beta_stop + beta_pass);
		if(_parent != NULL){
			p *= _parent->pass_probability(beta_stop, beta_pass);
		}
		return p;
	}
	void increment_stop_count(){
		_stop_count++;
		if(_parent != NULL){
			_parent->increment_pass_count();
		}
	}
	void decrement_stop_count(){
		_stop_count--;
		if(_stop_count < 0){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 停止回数の管理に不具合があります. _stop_count < 0\n");
		}
		if(_parent != NULL){
			_parent->decrement_passC_count();
		}
	}
	void increment_pass_count(){
		_pass_count++;
		if(_parent != NULL){
			_parent->increment_pass_count();
		}
	}
	void decrement_passC_count(){
		_pass_count--;
		if(_pass_count < 0){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 通過回数の管理に不具合があります. _pass_count < 0\n");
		}
		if(_parent != NULL){
			_parent->decrement_passC_count();
		}
	}
	void delete_child_node(id token_id){
		Node* child = find_child_node(token_id);
		if(child){
			_children.erase(token_id);
			delete child;
		}
		if(_children.size() == 0 && _arrangement.size() == 0){
			if(_parent != NULL){
				_parent->delete_child_node(this->_token_id);
			}
		}
	}
	int get_max_depth(int base){
		int max_depth = base;
		for(auto elem: _children){
			int depth = elem.second->get_max_depth(base + 1);
			if(depth > max_depth){
				max_depth = depth;
			}
		}
		return max_depth;
	}

	int get_num_child_nodes(){
		int num = _children.size();
		for(auto elem: _children){
			num += elem.second->get_num_child_nodes();
		}
		return num;
	}

	int get_num_customers(){
		int num = 0;
		for(auto elem: _arrangement){
			vector<int> customers = elem.second;
			for(int i = 0;i < customers.size();i++){
				num += customers[i];
			}
		}
		if(num != _num_customers){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 客の管理に不具合があります. num != _num_customers\n");
		}
		for(auto elem: _children){
			num += elem.second->get_num_customers();
		}
		return num;
	}

	int _sum_pass_counts(){
		int sum = _pass_count;
		for(auto elem: _children){
			sum += elem.second->_sum_pass_counts();
		}
		return sum;
	}

	int _sum_stop_counts(){
		int sum = _stop_count;
		for(auto elem: _children){
			sum += elem.second->_sum_pass_counts();
		}
		return sum;
	}

	void set_active_tokens(unordered_map<id, bool> &flags){
		for(auto elem: _arrangement){
			id token_id = elem.first;
			flags[token_id] = true;
		}
		for(auto elem: _children){
			elem.second->set_active_tokens(flags);
		}
	}

	void count_node_of_each_depth(unordered_map<id, int> &map){
		for(auto elem: _arrangement){
			id token_id = elem.first;
			map[_depth + 1] += 1;
		}
		for(auto elem: _children){
			elem.second->count_node_of_each_depth(map);
		}
	}

	// dとθの推定用
	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	double auxiliary_log_x_u(double theta_u){
		if(_num_customers >= 2){
			double x_u = Sampler::beta(theta_u + 1, _num_customers - 1);
			return log(x_u + 1e-8);
		}
		return 0;
	}
	double auxiliary_y_ui(double d_u, double theta_u){
		if(_num_tables >= 2){
			double sum_y_ui = 0;
			for(int i = 1;i <= _num_tables - 1;i++){
				double denominator = theta_u + d_u * (double)i;
				if(denominator == 0){
					c_printf("[R]%s", "エラー");
					c_printf("[n]%s", " 0除算. denominator == 0\n");
					continue;
				}
				sum_y_ui += Sampler::bernoulli(theta_u / denominator);;
			}
			return sum_y_ui;
		}
		return 0;
	}
	double auxiliary_1_y_ui(double d_u, double theta_u){
		if(_num_tables >= 2){
			double sum_1_y_ui = 0;
			for(int i = 1;i <= _num_tables - 1;i++){
				double denominator = theta_u + d_u * (double)i;
				if(denominator == 0){
					c_printf("[R]%s", "エラー");
					c_printf("[n]%s", " 0除算. denominator == 0\n");
					continue;
				}
				sum_1_y_ui += 1.0 - Sampler::bernoulli(theta_u / denominator);
			}
			return sum_1_y_ui;
		}
		return 0;
	}
	double auxiliary_1_z_uwkj(double d_u){
		double sum_z_uwkj = 0;
		// c_u..
		for(auto elem: _arrangement){
			// c_uw.
			vector<int> &tables = elem.second;
			for(int k = 0;k < tables.size();k++){
				// c_uwk
				int c_uwk = tables[k];
				if(c_uwk >= 2){
					for(int j = 1;j <= c_uwk - 1;j++){
						if(j - d_u == 0){
							c_printf("[R]%s", "エラー");
							c_printf("[n]%s", " 0除算. j - d_u == 0\n");
							continue;
						}
						sum_z_uwkj += 1 - Sampler::bernoulli((j - 1) / (j - d_u));
					}
				}
			}
		}
		return sum_z_uwkj;
	}

	void init_hyperparameters_at_depth_if_needed(int depth, vector<double> &d_m, vector<double> &theta_m){
		if(depth >= d_m.size()){
			while(d_m.size() <= depth){
				d_m.push_back(PYLM_INITIAL_D);
			}
			while(theta_m.size() <= depth){
				theta_m.push_back(PYLM_INITIAL_THETA);
			}
		}
	}

	friend ostream& operator<<(ostream& os, const Node& node){
		os << "[Node." << node._identifier << ":id." << node._token_id << ":depth." << node._depth << "]" << endl;
		os << "_num_tables: " << node._num_tables << ", _num_customers: " << node._num_customers << endl;
		os << "_stop_count: " << node._stop_count << ", _pass_count: " << node._pass_count << endl;
		os << "- _arrangement" << endl;
		for(auto elem: node._arrangement){
			os << "  [" << elem.first << "]" << endl;
			os << "    ";
			for(auto customers: elem.second){
				os << customers << ",";
			}
			os << endl;
		}
		os << endl;
		os << "- _children" << endl;
		os << "    ";
		for(auto elem: node._children){
			os << "  [Node." << elem.second->_identifier << ":id." << elem.first << "]" << ",";
		}
		os << endl;
		return os;
	}
};

id Node::_auto_increment = 1;

#endif