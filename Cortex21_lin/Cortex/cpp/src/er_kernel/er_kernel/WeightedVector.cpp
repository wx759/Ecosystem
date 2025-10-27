#ifdef __INCLUDE_WEIGHTED_VECTOR_CPP

#define _L(pos)	(((pos) << 1) + 1)
#define _R(pos)	(((pos) << 1) + 2)
#define _F(pos) (((pos) - 1) >> 1)

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_array_W()->WeightedVector<weight_t, value_t>::w_pair*
{
	if (W.size() == 0) return 0;
	else return &(W[0]);
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_array_P()->WeightedVector<weight_t, value_t>::pos_t* {
	if (P.size() == 0) return 0;
	else return &(P[0]);
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_array_V()->value_t* {
	if (V.size() == 0) return 0;
	else return &(V[0]);
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
_swap(pos_t pos0, pos_t pos1) {
	w_pair wp0 = W[pos0];
	w_pair wp1 = W[pos1];
	W[pos0] = wp1;
	W[pos1] = wp0;
	P[wp0.second] = pos1;
	P[wp1.second] = pos0;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_move_up(pos_t pos)->WeightedVector<weight_t, value_t>::pos_t {
	while (pos > 0) {
		pos_t father = _F(pos);
		if (W[father].first >= W[pos].first) break;
		_swap(pos, father);
		pos = father;
	}
	return pos;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_move_down(pos_t pos, pos_t bound)->WeightedVector<weight_t, value_t>::pos_t {
	if (bound == invalid_id) bound = W.size() - 1;
	while (pos < bound) {
		pos_t left = _L(pos);
		if (left > bound) left = pos;
		pos_t right = _R(pos);
		if (right > bound) right = pos;
		if (W[pos].first >= W[left].first && W[pos].first >= W[right].first) break;
		if (W[left].first > W[right].first) {
			_swap(pos, left);
			pos = left;
		}
		else {
			_swap(pos, right);
			pos = right;
		}
	}
	return pos;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
_move_auto(pos_t pos, pos_t bound)->WeightedVector<weight_t, value_t>::pos_t {
	return _move_down(_move_up(pos), bound);
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
size()->WeightedVector<weight_t, value_t>::id_t {
	return V.size();
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
value_array()->value_t* {
	return _array_V();
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
set_value(id_t id, const value_t& value) {
	V[id] = value;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
get_value(id_t id)->const value_t& {
	return V[id];
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
set_weight(id_t id, const weight_t& weight) {
	pos_t pos = P[id];
	W[pos].first = weight;
	_move_auto(pos);
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
set_all_weight(const weight_t& weight) {
	const auto iter_end = W.end();
	for (auto iter = W.begin(); iter != iter_end; ++iter)
		iter->first = weight;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
get_weight(id_t id)->const weight_t& {
	return W[P[id]].first;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
push_back(const weight_t& weight, const value_t& value)->WeightedVector<weight_t, value_t>::id_t {
	id_t end = V.size();
	V.push_back(value);
	W.push_back(w_pair({ weight, end }));
	P.push_back(end);
	_move_up(end);
	return end;
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
remove_tail(id_t new_end) {
	pos_t* p_end = _array_P() + P.size();
	pos_t pos_last = W.size() - 1;
	for (pos_t* p = _array_P() + new_end; p != p_end; ++p) {
		pos_t pos = *p;
		_swap(pos, pos_last);
		_move_auto(pos, --pos_last);
	}
	W.erase(W.begin() + new_end, W.end());
	P.erase(P.begin() + new_end, P.end());
	V.erase(V.begin() + new_end, V.end());
}

template<class weight_t, class value_t = char>
void WeightedVector<weight_t, value_t>::
exchange(id_t id0, id_t id1) {
	pos_t pos0 = P[id0];
	pos_t pos1 = P[id1];
	P[id0] = pos1;
	P[id1] = pos0;
	W[pos0].second = id1;
	W[pos1].second = id0;
	value_t val_tmp = V[id0];
	V[id0] = V[id1];
	V[id1] = val_tmp;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
top_at()->WeightedVector<weight_t, value_t>::id_t {
	if (W.size() == 0) return invalid_id;
	return W[0].second;
}

template<class weight_t, class value_t = char>
auto WeightedVector<weight_t, value_t>::
top_n_at(id_t n, id_t* ids)->WeightedVector<weight_t, value_t>::id_t {
	WeightedVector<weight_t, pos_t> candidate_pos;
	candidate_pos.push_back(W[0].first, 0);
	id_t end = size();
	while (n--) {
		id_t i0 = candidate_pos.top_at();
		if (i0 == invalid_id) break;
		pos_t pos = candidate_pos.get_value(i0);
		pos_t left = _L(pos);
		pos_t right = _R(pos);
		if (left < end) {
			candidate_pos.set_value(i0, left);
			candidate_pos.set_weight(i0, W[left].first);
			if (right < end)
				candidate_pos.push_back(W[right].first, right);
		}
		else {
			id_t candidate_pos_last = candidate_pos.size() - 1;
			candidate_pos.exchange(i0, candidate_pos_last);
			candidate_pos.remove_tail(candidate_pos_last);
		}
		*ids = W[pos].second;
		++ids;
	}
	return n + 1;
}

#endif