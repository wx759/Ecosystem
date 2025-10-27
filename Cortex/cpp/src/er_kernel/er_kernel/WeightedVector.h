#pragma once

#include <vector>

using namespace std;

template<class weight_t, class value_t = char>
class WeightedVector {
public:
	using id_t = uint64_t;
	using pos_t = id_t;
	using w_pair = pair<weight_t, id_t>;

	static const id_t invalid_id = id_t(-1);

protected:
	vector<w_pair> W;
	vector<pos_t> P;
	vector<value_t> V;

	w_pair* _array_W();
	pos_t* _array_P();
	value_t* _array_V();

	void _swap(pos_t pos0, pos_t pos1);
	pos_t _move_up(pos_t pos);
	pos_t _move_down(pos_t pos, pos_t bound = invalid_id);
	pos_t _move_auto(pos_t pos, pos_t bound = invalid_id);

public:
	WeightedVector() {}

	~WeightedVector() {
		W.clear();
		P.clear();
		V.clear();
	}

	id_t size();
	value_t* value_array();

	void set_value(id_t id, const value_t& value);
	const value_t& get_value(id_t id);

	void set_weight(id_t id, const weight_t& weight);
	void set_all_weight(const weight_t& weight);
	const weight_t& get_weight(id_t id);

	id_t push_back(const weight_t& weight, const value_t& value);
	void remove_tail(id_t new_end);
	void exchange(id_t id0, id_t id1);

	id_t top_at();
	id_t top_n_at(id_t n, id_t* ids);
};

#define __INCLUDE_WEIGHTED_VECTOR_CPP
#include "WeightedVector.cpp"