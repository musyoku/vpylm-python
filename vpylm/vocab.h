#ifndef _vocab_
#define _vocab_
#include <stdlib.h>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;
using id = unsigned long long;

class Vocab{
private:
	// wstring -> characters -> word
	unordered_map<wstring, id> _token_id_by_wstring;
	unordered_map<id, wstring> _wstring_by_token_id;
	unordered_map<id, vector<id> > _word_dict;
	map<vector<id>, id> _word_dict_inv;
	vector<id> _reusable_ids;
	vector<id> _null_word;
	id _auto_increment;
	id _unk_id;
	id _bos_id;
	id _eos_id;
	
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		static_cast<void>(version); // No use
		archive & _token_id_by_wstring;
		archive & _wstring_by_token_id;
		archive & _word_dict;
		archive & _word_dict_inv;
		archive & _reusable_ids;
		archive & _auto_increment;
	}

public:
	Vocab(){
		_bos_id = 0;
		_eos_id = 1;
		_wstring_by_token_id[_bos_id] = L'^';
		_wstring_by_token_id[_eos_id] = L'$';
		_auto_increment = _eos_id;
		_null_word = {0};
	}
	id generateId(){
		if(_reusable_ids.size() == 0){
			_auto_increment++;
			return _auto_increment;
		}
		id used_id = _reusable_ids.back();
		_reusable_ids.pop_back();
		return used_id;
	}
	void addCharacter(wstring ch){
		if(_token_id_by_wstring.find(ch) == _token_id_by_wstring.end()){
			id gen_id = generateId();
			_token_id_by_wstring[ch] = gen_id;
			_wstring_by_token_id[gen_id] = ch;
			if(_word_dict.find(gen_id) == _word_dict.end()){
				_word_dict[gen_id] = {gen_id};
			}
		}
	}
	id char2id(wstring ch){
		auto itr = _token_id_by_wstring.find(ch);
		if(itr == _token_id_by_wstring.end()){
			id gen_id = generateId();
			_token_id_by_wstring[ch] = gen_id;
			_wstring_by_token_id[gen_id] = ch;
			return gen_id;
		}
		return itr->second;
	}
	wstring id2char(id id){
		if(_wstring_by_token_id.find(id) == _wstring_by_token_id.end()){
			return L' ';
		}
		return _wstring_by_token_id[id];
	}
	id characters2word(vector<id> &char_ids){
		if(char_ids.size() == 1){
			return char_ids[0];
		}
		auto itr = _word_dict_inv.find(char_ids);
		if(itr == _word_dict_inv.end()){
			id gen_id = generateId();
			_word_dict_inv[char_ids] = gen_id;
			// cout << "word vocab generated " << gen_id << endl;
			_word_dict[gen_id] = char_ids;
			return gen_id;
		}
		return itr->second;
	}
	vector<id> &word2characters(id word_id){
		auto itr = _word_dict.find(word_id); 
		if (itr == _word_dict.end()) {
			return _null_word;
		}
		return itr->second;
	}
	wstring word2string(id word_id){
		if(_word_dict.find(word_id) == _word_dict.end()){
			if(_wstring_by_token_id.find(word_id) == _wstring_by_token_id.end()){
				return L"";
			}
			wstring str;
			str += id2char(word_id);
			return str;
		}
		vector<id> &char_ids = _word_dict[word_id];
		return characters2string(char_ids);
	}
	wstring characters2string(vector<id> &char_ids){
		wstring str;
		for(int i = 0;i < char_ids.size();i++){
			str += id2char(char_ids[i]);
		}
		return str;
	}
	void string2characters(wstring str, vector<id> &ids){
		ids.clear();
		for(int i = 0;i < str.length();i++){
			ids.push_back(char2id(str[i]));
		}
	}
	int numWords(){
		return _word_dict.size();
	}
	int numCharacters(){
		return _token_id_by_wstring.size();
	}
	id eosId(){
		return _eos_id;
	}
	id bosId(){
		return _bos_id;
	}
	id autoIncrement(){
		return _auto_increment;
	}
	void save(string dir = "model/"){
		string filename = dir + "vocab";
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << static_cast<const Vocab&>(*this);
		cout << "saved to " << filename << endl;
		cout << "	num_words: " << numWords() << endl;
		cout << "	num_characters: " << numCharacters() << endl;
		cout << "	auto_increment: " << _auto_increment << endl;
		cout << "	num_reusable_ids: " << _reusable_ids.size() << endl;
	}

	void load(string dir = "model/"){
		string filename = dir + "vocab";
		std::ifstream ifs(filename);
		if(ifs.good()){
			cout << "loading " << filename << endl;
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> *this;
			cout << "	num_words: " << numWords() << endl;
			cout << "	num_characters: " << numCharacters() << endl;
			cout << "	auto_increment: " << _auto_increment << endl;
			cout << "	num_reusable_ids: " << _reusable_ids.size() << endl;
		}
	}
	void dump(){
		for(auto kv : _wstring_by_token_id) {
			int key = kv.first;
			wstring charactor = kv.second;
			wcout << key << ": " << charactor << endl;
		} 
	}
	void clean(unordered_map<id, bool> &active_keys){
		unordered_map<id, vector<id> > new_word_dict;
		map<vector<id>, id> new_word_dict_inv;
		
		for(auto elem: _word_dict){
			id word_id = elem.first;
			if(active_keys.find(word_id) == active_keys.end()){
				if(_wstring_by_token_id.find(word_id) == _wstring_by_token_id.end()){
					_reusable_ids.push_back(word_id);
				}
			}else{
				vector<id> dict = _word_dict[word_id];
				new_word_dict[word_id] = dict;
				new_word_dict_inv[dict] = word_id;
			}
		}

		_word_dict.clear();
		_word_dict_inv.clear();
		_word_dict = new_word_dict;
		_word_dict_inv = new_word_dict_inv;
	}
};

#endif