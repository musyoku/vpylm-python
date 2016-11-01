#ifndef _hpylm_
#define _hpylm_
#include <vector>
#include <random>
#include <unordered_map> 
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "c_printf.h"
#include "sampler.h"
#include "node.h"
#include "const.h"
#include "vocab.h"

class HPYLM{
private:
	friend class boost::serialization::access;
	template <class Archive>
	// モデルの保存
	void serialize(Archive& archive, unsigned int version)
	{
		static_cast<void>(version); // No use
		archive & _root;
		archive & _max_depth;
		archive & _g0;
		archive & _d_m;
		archive & _theta_m;
		archive & _a_m;
		archive & _b_m;
		archive & _alpha_m;
		archive & _beta_m;
	}
public:
	Node* _root;				// 文脈木のルートノード
	int _max_depth;				// 最大の深さ
	int _bottom;				// VPYLMへ拡張時に使う
	double _g0;					// ゼログラム確率

	// 深さmのノードに関するパラメータ
	vector<double> _d_m;		// Pitman-Yor過程のディスカウント係数
	vector<double> _theta_m;	// Pitman-Yor過程の集中度

	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	vector<double> _a_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _b_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _alpha_m;	// ガンマ分布のパラメータ	θの推定用
	vector<double> _beta_m;		// ガンマ分布のパラメータ	θの推定用

	HPYLM(int ngram = 2){
		// 深さは0から始まることに注意
		// バイグラムなら最大深さは1
		_max_depth = ngram - 1;

		_root = new Node(0);
		_root->_depth = 0;	// ルートは深さ0

		for(int n = 0;n <= _max_depth;n++){
			_d_m.push_back(PYLM_INITIAL_D);	
			_theta_m.push_back(PYLM_INITIAL_THETA);
			_a_m.push_back(PYLM_INITIAL_A);	
			_b_m.push_back(PYLM_INITIAL_B);	
			_alpha_m.push_back(PYLM_INITIAL_ALPHA);
			_beta_m.push_back(PYLM_INITIAL_BETA);
		}
	}

	// 単語列のi番目の単語をモデルに追加
	void add(vector<id> &token_ids, int token_t_index){
		// HPYLMでは深さは固定
		if(token_t_index < _max_depth){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 客を追加できません. $token_t_index < $_max_depth\n");
			return;
		}

		id token_t = token_ids[token_t_index];
		
		// ノードをたどっていく
		// なければ作る
		Node* node = _root;
		for(int depth = 0;depth < _max_depth;depth++){
			id u_t = token_ids[token_t_index - depth - 1];
			Node* child = node->generateChildIfNeeded(u_t);
			if(child == NULL){
				c_printf("[R]%s", "エラー");
				c_printf("[n]%s", " レストランが存在しません. $child == NULL\n");
				return;
			}
			node = child;
		}

		// "客が親から生成される確率"と自らが持つ経験分布の混合分布から次の客の座るテーブルが決まる
		double parent_p_w = _g0;
		if(node->parentExists()){
			parent_p_w = node->_parent->Pw(token_t, _g0, _d_m, _theta_m);
		}
		node->addCustomer(token_t, parent_p_w, _d_m, _theta_m);
	}

	bool remove(vector<id> &token_ids, int w_t_i){
		// HPYLMでは深さは固定
		if(w_t_i < _max_depth){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 客を除去できません. $w_t_i < $_max_depth\n");
			return false;
		}

		id w_t = token_ids[w_t_i];

		Node* node = _root;
		for(int depth = 0;depth < _max_depth;depth++){
			id u_t = token_ids[w_t_i - depth - 1];
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				c_printf("[R]%s", "エラー");
				c_printf("[n]%s", " 客を除去できません. $child == NULL\n");
				return false;
			}
			node = child;
		}

		bool need_to_remove_from_parent = false;
		node->removeCustomer(w_t);

		if(node->parentExists() && node->needToRemoveFromParent()){
			// 客が一人もいなくなったらノードを削除する
			node->_parent->deleteChildWithId(node->_id);
		}
		return true;
	}

	double Pw_h(vector<id> &token_ids, vector<id> context_ids){
		double p = 1;
		for(int n = 0;n < token_ids.size();n++){
			p *= Pw_h(token_ids[n], context_ids);
			context_ids.push_back(token_ids[n]);
		}
		return p;
	}

	double Pw_h(id word_id, vector<id> &context_ids){
		// HPYLMでは深さは固定
		if(context_ids.size() < _max_depth){
			c_printf("[R]%s", "エラー");
			c_printf("[n]%s", " 単語確率を計算できません. $context_ids.size() < $_max_depth\n");
			return -1;
		}

		Node* node = _root;
		for(int n = 0;n < _max_depth;n++){
			id u_t = context_ids[context_ids.size() - n - 1];
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				c_printf("[R]%s", "エラー");
				c_printf("[n]%s", " 単語確率を計算できません. $child == NULL\n");
				return -1;
			}
			node = child;
		}

		return node->Pw(word_id, _g0, _d_m, _theta_m);
	}

	double Pw(id word_id){
		double p = _root->Pw(word_id, _g0, _d_m, _theta_m);
		return p;
	}

	double Pw(vector<id> &token_ids){
		if(token_ids.size() == 0){
			return 0;
		}
		double p = 1;
		vector<id> context_ids(token_ids.begin(), token_ids.begin() + _max_depth);
		for(int depth = _max_depth;depth < token_ids.size();depth++){
			id word_id = token_ids[depth];
			double _p = Pw_h(word_id, context_ids);
			p *= _p;
			context_ids.push_back(word_id);
		}
		return p;
	}
	double log_Pw(vector<id> &token_ids){
		if(token_ids.size() == 0){
			return 0;
		}
		double p = 0;
		vector<id> context_ids(token_ids.begin(), token_ids.begin() + _max_depth);
		for(int depth = _max_depth;depth < token_ids.size();depth++){
			id word_id = token_ids[depth];
			double _p = Pw_h(word_id, context_ids);
			p += log(_p + 1e-10);
			context_ids.push_back(word_id);
		}
		return p;
	}

	double log2_Pw(vector<id> &token_ids){
		if(token_ids.size() == 0){
			return 0;
		}
		double p = 0;
		vector<id> context_ids(token_ids.begin(), token_ids.begin() + _max_depth);
		for(int depth = _max_depth;depth < token_ids.size();depth++){
			id word_id = token_ids[depth];
			double _p = Pw_h(word_id, context_ids);
			p += log2(_p + 1e-10);
			context_ids.push_back(word_id);
		}
		return p;
	}

	id sampleNextWord(vector<id> &context_ids, id eos_id){
		Node* node = _root;
		int depth = context_ids.size() < _max_depth ? context_ids.size() : _max_depth;

		for(int n = 0;n < depth;n++){
			id u_t = context_ids[context_ids.size() - n - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->findChildWithId(u_t);
			if(child == NULL){
				break;
			}
			node = child;
		}

		vector<id> token_ids;
		vector<double> probs;
		double sum = 0;
		for(auto elem: node->_arrangement){
			id word_id = elem.first;
			double p = Pw_h(word_id, context_ids);
			if(p > 0){
				token_ids.push_back(word_id);
				probs.push_back(p);
				sum += p;
			}
		}
		if(token_ids.size() == 0){
			return eos_id;
		}
		if(sum == 0){
			return eos_id;
		}
		double ratio = 1.0 / sum;
		double r = Sampler::uniform(0, 1);
		sum = 0;
		id sampled_word_id = token_ids.back();
		for(int i = 0;i < token_ids.size();i++){
			sum += probs[i] * ratio;
			if(sum > r){
				sampled_word_id = token_ids[i];
				break;
			}
		}
		return sampled_word_id;
	}


	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	void sumAuxiliaryVariablesRecursively(Node* node, vector<double> &sum_log_x_u_m, vector<double> &sum_y_ui_m, vector<double> &sum_1_y_ui_m, vector<double> &sum_1_z_uwkj_m){
		for(auto elem: node->_children){
			Node* child = elem.second;
			int depth = child->_depth;

			if(depth > _bottom){
				_bottom = depth;
			}
			if(depth >= _d_m.size()){
				while(_d_m.size() <= depth){
					_d_m.push_back(PYLM_INITIAL_D);
				}
			}
			if(depth >= _theta_m.size()){
				while(_theta_m.size() <= depth){
					_theta_m.push_back(PYLM_INITIAL_THETA);
				}
			}

			double d = _d_m[depth];
			double theta = _theta_m[depth];
			sum_log_x_u_m[depth] += child->auxiliary_log_x_u(theta);	// log(x_u)
			sum_y_ui_m[depth] += child->auxiliary_y_ui(d, theta);		// y_ui
			sum_1_y_ui_m[depth] += child->auxiliary_1_y_ui(d, theta);	// 1 - y_ui
			sum_1_z_uwkj_m[depth] += child->auxiliary_1_z_uwkj(d);		// 1 - z_uwkj

			sumAuxiliaryVariablesRecursively(child, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);
		}
	}

	// dとθの推定
	void sampleHyperParams(){
		unordered_map<int, vector<Node*> > nodes_by_depth;
		int max_depth = _d_m.size() - 1;

		// 親ノードの深さが0であることに注意
		vector<double> sum_log_x_u_m(max_depth + 1, 0.0);
		vector<double> sum_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_z_uwkj_m(max_depth + 1, 0.0);

		// _root
		sum_log_x_u_m[0] = _root->auxiliary_log_x_u(_theta_m[0]);			// log(x_u)
		sum_y_ui_m[0] = _root->auxiliary_y_ui(_d_m[0], _theta_m[0]);		// y_ui
		sum_1_y_ui_m[0] = _root->auxiliary_1_y_ui(_d_m[0], _theta_m[0]);	// 1 - y_ui
		sum_1_z_uwkj_m[0] = _root->auxiliary_1_z_uwkj(_d_m[0]);				// 1 - z_uwkj


		// それ以外
		_bottom = 0;
		sumAuxiliaryVariablesRecursively(_root, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);

		for(int u = 0;u <= _bottom;u++){

			if(u >= _a_m.size()){
				while(_a_m.size() <= u){
					_a_m.push_back(PYLM_INITIAL_A);
				}
			}
			if(u >= _b_m.size()){
				while(_b_m.size() <= u){
					_b_m.push_back(PYLM_INITIAL_B);
				}
			}
			if(u >= _alpha_m.size()){
				while(_alpha_m.size() <= u){
					_alpha_m.push_back(PYLM_INITIAL_ALPHA);
				}
			}
			if(u >= _beta_m.size()){
				while(_beta_m.size() <= u){
					_beta_m.push_back(PYLM_INITIAL_BETA);
				}
			}
			
			_d_m[u] = Sampler::beta(_a_m[u] + sum_1_y_ui_m[u], _b_m[u] + sum_1_z_uwkj_m[u]);
			_theta_m[u] = Sampler::gamma(_alpha_m[u] + sum_y_ui_m[u], 1 / (_beta_m[u] - sum_log_x_u_m[u]));
		}

		int num_remove = _d_m.size() - _bottom;
		for(int n = 0;n < num_remove;n++){
			_d_m.pop_back();
			_theta_m.pop_back();
			_a_m.pop_back();
			_b_m.pop_back();
			_alpha_m.pop_back();
			_beta_m.pop_back();
		}
	}

	int maxDepth(){
		return _d_m.size() - 1;
	}

	int numChildNodes(){
		return _root->numChildNodes();
	}

	int numCustomers(){
		return _root->numCustomers();
	}

	void setActiveKeys(unordered_map<id, bool> &keys){
		_root->setActiveKeys(keys);
	}

	void countNodeForEachDepth(unordered_map<id, int> &map){
		_root->countNodeForEachDepth(map);
	}

	void save(string dir = "model/"){
		string filename = dir + "hpylm.model";
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << static_cast<const HPYLM&>(*this);
		cout << "saved to " << filename << endl;
		cout << "	num_customers: " << numCustomers() << endl;
		cout << "	num_nodes: " << numChildNodes() << endl;
		cout << "	max_depth: " << maxDepth() << endl;
	}

	void load(string dir = "model/"){
		string filename = dir + "hpylm.model";
		std::ifstream ifs(filename);
		if(ifs.good()){
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> *this;
		}
	}
};

#endif