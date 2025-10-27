// er_kernel.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "er_kernel.h"

#define DEFAULT_ALLOW_SHORT_SEQ	false
#define DEFAULT_PICK_LEN		1

PTR ex_rp_new(LH_REC_T max_record_num, SEQ_LEN_T pick_len, INT32 allow_short_seq) {
	EX_RP* pER = new EX_RP;
	pER->max_record_num = max_record_num;
	BATCH_MAKER* pBM = &(pER->batch_maker);
	if (allow_short_seq >= 0) pBM->allow_short_seq = allow_short_seq;
	else pBM->allow_short_seq = DEFAULT_ALLOW_SHORT_SEQ;
	if (pick_len > 0) pBM->pick_len = pick_len;
	else pBM->pick_len = DEFAULT_PICK_LEN;
	pER->remove_handle = -1;
	pBM->ps_table.push_back(new PS_UniRand());
	PTR ptrER = (PTR)pER;
	ex_rp_clear(ptrER);
	return ptrER;
}

void ex_rp_del(PTR ptrER) {
	EX_RP* pER = (EX_RP*)ptrER;
	delete pER;
}

void ex_rp_clear(PTR ptrER) {
	EX_RP* pER = (EX_RP*)ptrER;
	pER->head_handle = 0;
	pER->tail_handle = 0;
	pER->record_num = 0;
	pER->state_size = 0;
	if (pER->remove_handle >= 0) pER->remove_handle = 0;
	const auto iter_epi_end = pER->episodes.end();
	for (auto iter = pER->episodes.begin(); iter != iter_epi_end; ++iter) delete iter->second;
	pER->episodes.clear();
	BATCH_MAKER* pBM = &(pER->batch_maker);
	const auto iter_ps_end = pBM->ps_table.end();
	for (auto iter = pBM->ps_table.begin(); iter != iter_ps_end; ++iter) (*iter)->on_pick_table_remove_tail(0);
	pBM->pick_table.clear();
}

template <class T>
UINT64 SERIALIZE_VECTOR_SIZE(vector<T>& V) {
	return V.size() * sizeof(T) + sizeof(UINT64);
}

template <class T>
char* SERIALIZE_VECTOR(char* ptr, vector<T>& V) {
	UINT64 size = V.size();
	ptr = WRITE_BYTES(ptr, size, UINT64);
	ptr = WRITE_BYTES_N(ptr, AS_ARRAY(V), T, size);
	return ptr;
}

template <class T>
char* UNSERIALIZE_VECTOR(char* ptr, vector<T>& V) {
	UINT64 size;
	ptr = READ_BYTES(ptr, size, UINT64);
	V.insert(V.end(), ((T*)ptr), ((T*)ptr) + size);
	ptr += sizeof(T) * size;
	return ptr;
}

inline bool with_prioritized_pick_selector(EX_RP* pER) {
	return (pER->batch_maker.ps_table.size() > 1);
}

UINT64 ex_rp_serialize_size(PTR ptrER) {
	EX_RP* pER = (EX_RP*)ptrER;
	UINT64 size = sizeof(pER->head_handle)
				+ sizeof(pER->tail_handle)
				+ sizeof(pER->remove_handle)
				+ sizeof(pER->record_num)
				+ sizeof(pER->state_size)
				+ sizeof(LH_EPI_T);
	const auto iter_epi_end = pER->episodes.end();
	bool including_pick_table = with_prioritized_pick_selector(pER);
	for (auto iter = pER->episodes.begin(); iter != iter_epi_end; ++iter) {
		EPISODE* pEPI = iter->second;
		size += sizeof(LH_EPI_T) + sizeof(pEPI->record_num) 
			  + sizeof(pEPI->access_flag) + sizeof(pEPI->terminated);
		size += SERIALIZE_VECTOR_SIZE<ACTION_T>(pEPI->actions);
		size += SERIALIZE_VECTOR_SIZE<REWARD_T>(pEPI->rewards);
		size += SERIALIZE_VECTOR_SIZE<STATE_T>(pEPI->states);
		if (including_pick_table) size += SERIALIZE_VECTOR_SIZE<LH_REC_T>(pEPI->pick_index);
	}
	if (including_pick_table) {
		BATCH_MAKER* pBM = &(pER->batch_maker);
		size += sizeof(pBM->allow_short_seq) + sizeof(pBM->pick_len);
		size += SERIALIZE_VECTOR_SIZE<PICK>(pBM->pick_table);
		const auto iter_ps_end = pBM->ps_table.end();
		for (auto iter = pBM->ps_table.begin(); iter != iter_ps_end; ++iter)
			size += (*iter)->serialize_size();
	}
	return size;
}

void ex_rp_serialize(PTR ptrER, char* pSer) {
	EX_RP* pER = (EX_RP*)ptrER;
	LH_EPI_T epi_num = pER->episodes.size();
	pSer = WRITE_BYTES(pSer, pER->head_handle, LH_EPI_T);
	pSer = WRITE_BYTES(pSer, pER->tail_handle, LH_EPI_T);
	pSer = WRITE_BYTES(pSer, pER->remove_handle, LH_EPI_T);
	pSer = WRITE_BYTES(pSer, pER->record_num, LH_REC_T);
	pSer = WRITE_BYTES(pSer, pER->state_size, STATE_SIZE_T);
	pSer = WRITE_BYTES(pSer, epi_num, LH_EPI_T);
	bool including_pick_table = with_prioritized_pick_selector(pER);
	const auto iter_epi_end = pER->episodes.end();
	for (auto iter = pER->episodes.begin(); iter != iter_epi_end; ++iter) {
		LH_EPI_T hEPI = iter->first;
		EPISODE* pEPI = iter->second;
		pSer = WRITE_BYTES(pSer, hEPI, LH_EPI_T);
		pSer = WRITE_BYTES(pSer, pEPI->record_num, H_REC_T);
		pSer = WRITE_BYTES(pSer, pEPI->access_flag, UCHAR);
		pSer = WRITE_BYTES(pSer, pEPI->terminated, bool);
		pSer = SERIALIZE_VECTOR<ACTION_T>(pSer, pEPI->actions);
		pSer = SERIALIZE_VECTOR<REWARD_T>(pSer, pEPI->rewards);
		pSer = SERIALIZE_VECTOR<STATE_T>(pSer, pEPI->states);
		if (including_pick_table) pSer = SERIALIZE_VECTOR<LH_REC_T>(pSer, pEPI->pick_index);
	}
	if (including_pick_table) {
		BATCH_MAKER* pBM = &(pER->batch_maker);
		pSer = WRITE_BYTES(pSer, pBM->allow_short_seq, bool);
		pSer = WRITE_BYTES(pSer, pBM->pick_len, SEQ_LEN_T);
		pSer = SERIALIZE_VECTOR<PICK>(pSer, pBM->pick_table);
		const auto iter_ps_end = pBM->ps_table.end();
		for (auto iter = pBM->ps_table.begin(); iter != iter_ps_end; ++iter)
			pSer = (*iter)->serialize(pSer);
	}
}

inline void resolve_pick_link(EX_RP* pER) {
	PICK_TABLE* pPT = &(pER->batch_maker.pick_table);
	EPI_TABLE* pET = &(pER->episodes);
	const auto iter_pick_end = pPT->end();
	for (auto iter = pPT->begin(); iter != iter_pick_end; ++iter)
		iter->pEPI = pET->find(iter->hEPI)->second;
}

inline void rebuild_pick_table(EX_RP* pER) {
	BATCH_MAKER* pBM = &(pER->batch_maker);
	PICK_TABLE* pPT = &(pBM->pick_table);
	PickSelector* pPS0 = pBM->ps_table[0];
	pPS0->on_pick_table_remove_tail(0);
	pPT->clear();
	const auto iter_epi_end = pER->episodes.end();
	for (auto iter = pER->episodes.begin(); iter != iter_epi_end; ++iter) {
		LH_EPI_T hEPI = iter->first;
		EPISODE* pEPI = iter->second;
		PICK_INDEX* pPI = &(pEPI->pick_index);
		pPI->clear();
		if (pEPI->record_num == 0) continue;
		H_REC_T last_pick_pos = pEPI->record_num - pBM->pick_len;
		if (last_pick_pos < 0 && pBM->allow_short_seq) last_pick_pos = 0;
		for (H_REC_T i = 0; i <= last_pick_pos; ++i) {
			pPI->push_back(pPT->size());
			pPT->push_back(PICK({ hEPI, pEPI, i }));
			pPS0->on_pick_table_push_back();
		}
	}
}

void ex_rp_unserialize(PTR ptrER, char* pSer) {
	EX_RP* pER = (EX_RP*)ptrER;
	LH_EPI_T epi_num;
	const auto iter_epi_end = pER->episodes.end();
	for (auto iter = pER->episodes.begin(); iter != iter_epi_end; ++iter) delete iter->second;
	pER->episodes.clear();
	pSer = READ_BYTES(pSer, pER->head_handle, LH_EPI_T);
	pSer = READ_BYTES(pSer, pER->tail_handle, LH_EPI_T);
	pSer = READ_BYTES(pSer, pER->remove_handle, LH_EPI_T);
	pSer = READ_BYTES(pSer, pER->record_num, LH_REC_T);
	pSer = READ_BYTES(pSer, pER->state_size, STATE_SIZE_T);
	pSer = READ_BYTES(pSer, epi_num, LH_EPI_T);
	bool including_pick_table = with_prioritized_pick_selector(pER);
	for (LH_EPI_T i = 0; i < epi_num; ++i) {
		LH_EPI_T hEPI;
		pSer = READ_BYTES(pSer, hEPI, LH_EPI_T);
		EPISODE* pEPI = pER->episodes[hEPI] = new EPISODE();
		pSer = READ_BYTES(pSer, pEPI->record_num, H_REC_T);
		pSer = READ_BYTES(pSer, pEPI->access_flag, UCHAR);
		pSer = READ_BYTES(pSer, pEPI->terminated, bool);
		pSer = UNSERIALIZE_VECTOR<ACTION_T>(pSer, pEPI->actions);
		pSer = UNSERIALIZE_VECTOR<REWARD_T>(pSer, pEPI->rewards);
		pSer = UNSERIALIZE_VECTOR<STATE_T>(pSer, pEPI->states);
		if (including_pick_table) pSer = UNSERIALIZE_VECTOR<LH_REC_T>(pSer, pEPI->pick_index);
	}
	if (including_pick_table) {
		BATCH_MAKER* pBM = &(pER->batch_maker);
		pBM->pick_table.clear();
		pSer = READ_BYTES(pSer, pBM->allow_short_seq, bool);
		pSer = READ_BYTES(pSer, pBM->pick_len, SEQ_LEN_T);
		pSer = UNSERIALIZE_VECTOR<PICK>(pSer, pBM->pick_table);
		const auto iter_ps_end = pBM->ps_table.end();
		for (auto iter = pBM->ps_table.begin(); iter != iter_ps_end; ++iter)
			pSer = (*iter)->unserialize(pSer);
		resolve_pick_link(pER);
	}
	else rebuild_pick_table(pER);
}

void ex_rp_get_pick_policy(PTR ptrER, SEQ_LEN_T* ptr_pick_len, bool* p_allow_short_seq) {
	EX_RP* pER = (EX_RP*)ptrER;
	*p_allow_short_seq = pER->batch_maker.allow_short_seq;
	*ptr_pick_len = pER->batch_maker.pick_len;
}

LH_EPI_T ex_rp_new_episode(PTR ptrER) {
	EX_RP* pER = (EX_RP*)ptrER;
	LH_EPI_T hEPI = pER->head_handle;
	EPISODE* pEPI = pER->episodes[hEPI] = new EPISODE();
	pEPI->access_flag = 0;
	pEPI->record_num = 0;
	pEPI->terminated = false;
	++(pER->head_handle);
	return hEPI;
}

LH_REC_T eliminate_FIFO(EX_RP* pER) {
	return pER->tail_handle;
}

LH_REC_T eliminate_SecondChance(EX_RP* pER) {
	LH_EPI_T remove_handle = pER->remove_handle;
	if (remove_handle < pER->tail_handle
		|| remove_handle >= pER->head_handle)
		remove_handle = pER->tail_handle;
	EPI_TABLE* pET = &(pER->episodes);
	EPI_TABLE::iterator iter = pET->find(remove_handle);
	LH_EPI_T hEPI;
	while (1) {
		hEPI = iter->first;
		EPISODE* pEPI = iter->second;
		++iter;
		if (iter == pET->end()) iter = pET->begin();
		if (pEPI->access_flag) pEPI->access_flag = 0;
		else break;
	}
	pER->remove_handle = iter->first;
	return hEPI;
}

LH_EPI_T ex_rp_record(PTR ptrER, LH_EPI_T hEPI, STATE_T* state, ACTION_T action, REWARD_T reward,
					  STATE_T* final_state, STATE_SIZE_T state_size) {
	EX_RP* pER = (EX_RP*)ptrER;
	if (pER->state_size == 0) pER->state_size = state_size;
	else state_size = pER->state_size;
	EPI_TABLE* pET = &(pER->episodes);
	BATCH_MAKER* pBM = &(pER->batch_maker);
	PICK_TABLE* pPT = &(pBM->pick_table);
	PS_TABLE* pPST = &(pBM->ps_table);
	const auto iter_ps_end = pPST->end();
	SEQ_LEN_T pick_len = pBM->pick_len;
	auto iter_epi = pET->find(hEPI);
	if (iter_epi == pET->end()) {
		hEPI = ex_rp_new_episode(ptrER);
		iter_epi = pET->find(hEPI);
	}
	EPISODE* pEPI = iter_epi->second;
	PICK_INDEX* pPI = &(pEPI->pick_index);
	STATE_TABLE* pST = &(pEPI->states);
	pST->insert(pST->end(), state, state + state_size);
	pEPI->actions.push_back(action);
	pEPI->rewards.push_back(reward);
	if (final_state) {
		pST->insert(pST->end(), final_state, final_state + state_size);
		pEPI->terminated = true;
	}
	H_REC_T old_record_num = pEPI->record_num;
#pragma warning(push)
#pragma warning(disable: 4267)
	H_REC_T record_num = pEPI->record_num = pEPI->states.size() / state_size - 1;
#pragma warning(pop)
	pER->record_num += record_num - old_record_num;
	for (H_REC_T n = old_record_num + 1; n <= record_num; ++n) {
		H_REC_T pick_pos = n - pick_len;
		if (pick_pos > 0 || (!pBM->allow_short_seq && (pick_pos == 0))) {
			pPI->push_back(pPT->size());
			pPT->push_back(PICK({ hEPI, pEPI, pick_pos }));
			for (auto iter_ps = pPST->begin(); iter_ps != iter_ps_end; ++iter_ps)
				(*iter_ps)->on_pick_table_push_back();
		}
		else if (pBM->allow_short_seq && n == 1) {
			pPI->push_back(pPT->size());
			pPT->push_back(PICK({ hEPI, pEPI, 0 }));
			for (auto iter_ps = pPST->begin(); iter_ps != iter_ps_end; ++iter_ps)
				(*iter_ps)->on_pick_table_push_back();
		}
	}
	if (pER->max_record_num > 0) {
		while (pER->record_num > pER->max_record_num) {
			LH_REC_T(*eliminate_func)(EX_RP*) = eliminate_FIFO;
			if (pER->remove_handle >= 0) eliminate_func = eliminate_SecondChance;
			LH_EPI_T hEPI_remove = eliminate_func(pER);
			pEPI = pET->find(hEPI_remove)->second;
			pER->record_num -= pEPI->record_num;
			PICK* aPT = AS_ARRAY(*pPT);
			LH_REC_T hPICK_end = pPT->size();
			pPI = &(pEPI->pick_index);
			const auto iter_pi_end = pPI->end();
			for (auto iter = pPI->begin(); iter != iter_pi_end; ++iter) {
				LH_REC_T hPICK_remove = *iter;
				while (hPICK_remove < hPICK_end && aPT[--hPICK_end].hEPI == hEPI_remove);
				if (hPICK_remove >= hPICK_end) continue;
				PICK replace_pick = aPT[hPICK_end];
				replace_pick.pEPI->pick_index[replace_pick.pos] = hPICK_remove;
				aPT[hPICK_remove] = replace_pick;
				for (auto iter_ps = pPST->begin(); iter_ps != iter_ps_end; ++iter_ps)
					(*iter_ps)->on_pick_exchange(hPICK_end, hPICK_remove);
			}
			pPT->erase(pPT->begin() + hPICK_end, pPT->end());
			for (auto iter_ps = pPST->begin(); iter_ps != iter_ps_end; ++iter_ps)
				(*iter_ps)->on_pick_table_remove_tail(hPICK_end);
			delete pEPI;
			if (pER->tail_handle == hEPI_remove) {
				LH_EPI_T tail = pER->tail_handle;
				LH_EPI_T head = pER->head_handle;
				while (pET->find(++tail) == pET->end() && tail < head);
				pER->tail_handle = tail;
			}
			pET->erase(hEPI_remove);
		}
	}
	return hEPI;
}

inline PICK* select_picks(BATCH_MAKER* pBM, EPI_TABLE* pEPI, H_PS_T hPS, BATCH_SIZE_T batch_size) {
	PICK* aPT = AS_ARRAY(pBM->pick_table);
	PICK* picks = new PICK[batch_size];
	LH_REC_T* hPICKs = new LH_REC_T[batch_size];
	pBM->ps_table[hPS]->select(batch_size, hPICKs);
	PICK* iter_pick = picks;
	LH_REC_T* iter_pi = hPICKs;
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i) 
		*(iter_pick++) = aPT[*(iter_pi++)];
	delete[] hPICKs;
	return picks;
}

int ex_rp_get_random_batch(PTR ptrER, H_PS_T hPS, BATCH_SIZE_T batch_size, float valid_sample_rate,
						   LH_EPI_T* pick_epi, H_REC_T* pick_pos, STATE_T* state, ACTION_T* action, REWARD_T* reward,
						   STATE_T* state_, SEQ_LEN_T* seq_len, SEQ_LEN_T* seq_len_) {
	EX_RP* pER = (EX_RP*)ptrER;
	BATCH_MAKER* pBM = &(pER->batch_maker);
	PICK_TABLE* pPT = &(pBM->pick_table);
	if (pPT->size() <= batch_size * valid_sample_rate) return pPT->size();
	PICK* picks = select_picks(pBM, &(pER->episodes), hPS, batch_size);
	SEQ_LEN_T pick_len = pBM->pick_len;
	STATE_SIZE_T state_size = pER->state_size;
	UINT64 pick_state_size = pick_len * state_size;
	UINT64 sizeof_state = sizeof(STATE_T) * state_size;
	UINT64 sizeof_action = sizeof(ACTION_T);
	UINT64 sizeof_reward = sizeof(REWARD_T);
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i) {
		EPISODE* pEPI = picks[i].pEPI;
		H_REC_T pos = picks[i].pos;
		H_REC_T pick_end = pos + pick_len;
		bool touch_tail = (pick_end >= pEPI->record_num);
		if (touch_tail) pick_end = pEPI->record_num;
		SEQ_LEN_T len = pick_end - pos;
		if (pick_epi) pick_epi[i] = picks[i].hEPI;
		if (pick_pos) pick_pos[i] = pos;
		seq_len[i] = len;
		seq_len_[i] = (touch_tail && pEPI->terminated) ? len - 1 : len;
		STATE_T* addr = AS_ARRAY(pEPI->states) + state_size * pos;
		UINT64 size = sizeof_state * len;
		memcpy(state, addr, size);
		memcpy(state_, addr + state_size, size);
		memcpy(action, AS_ARRAY(pEPI->actions) + pos, sizeof_action * len);
		memcpy(reward, AS_ARRAY(pEPI->rewards) + pos, sizeof_reward * len);
		state += pick_state_size;
		state_ += pick_state_size;
		action += pick_len;
		reward += pick_len;
		pEPI->access_flag |= 1;
	}
	delete[] picks;
	return 1;
}

void ex_rp_set_elimination_policy(PTR ptrER, INT32 SecChanElimination) {
	EX_RP* pER = (EX_RP*)ptrER;
	if (SecChanElimination) pER->remove_handle = pER->tail_handle;
	else pER->remove_handle = -1;
}

void ex_rp_set_pick_priority(PTR ptrER, H_PS_T hPS, LH_EPI_T* pick_epi, H_REC_T* pick_pos,
							 PRIORITY_T* prio, BATCH_SIZE_T batch_size) {
	EX_RP* pER = (EX_RP*)ptrER;
	if (hPS == 0) return;
	EPI_TABLE* pET = &(pER->episodes);
	LH_REC_T* hPICK = new LH_REC_T[batch_size];
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i) {
		LH_EPI_T hEPI = pick_epi[i];
		if (hEPI == -1) {
			hPICK[i] = -1;
			continue;
		}
		auto iter_epi = pET->find(hEPI);
		if (iter_epi == pET->end()) {
			hPICK[i] = -1;
			pick_epi[i] = -1;
		}
		else {
			hPICK[i] = iter_epi->second->pick_index[pick_pos[i]];
		}
	}
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i) {
		if (hPICK[i] != -1) pER->batch_maker.ps_table[hPS]->set_priority(hPICK[i], prio[i]);
	}
	delete[] hPICK;
}

void ex_rp_reset_pick_priorities(PTR ptrER, H_PS_T hPS) {
	EX_RP* pER = (EX_RP*)ptrER;
	pER->batch_maker.ps_table[hPS]->reset_priorities();
}

void ex_rp_del_pick_selector(PTR ptrER, H_PS_T hPS) {
	EX_RP* pER = (EX_RP*)ptrER;
	delete pER->batch_maker.ps_table[hPS];
}


//以下代码由Anna篡改hh
void ex_rp_decoded_batch_actions(PTR ptrER, ACTION_T* encoded_values, ACTION_PY* actions, BATCH_SIZE_T batchsize, int picklen, int action_size) {
	// 注意：现在actions是一个一维数组，不再需要分配内存给每个batch的子数组
	// (*actions)已经被假定为足够大，能够容纳所有解码后的数据

	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < picklen; j++) {
			// 计算当前动作在一维数组中的起始索引
			int start_index = (i * picklen * action_size) + (j * action_size);
			// 解码每个encoded_value，并填充到actions数组的正确位置
			ex_rp_decoded_action(encoded_values[i * picklen + j], &(actions[start_index]), action_size);
		}
	}
}
/*

void ex_rp_decoded_batch_actions(PTR ptrER, ACTION_T** encoded_values, ACTION_PY*** actions, int batchsize, int picklen, int action_size) {
	// 为actions分配内存
	*actions = (float**)malloc(batchsize * sizeof(float*));
	for (int i = 0; i < batchsize; i++) {
		(*actions)[i] = (float*)malloc(picklen * action_size * sizeof(float));
		for (int j = 0; j < picklen; j++) {
			// 解码每个encoded_value，并填充到actions数组
			ex_rp_decoded_action(encoded_values[i][j], &((*actions)[i][j * action_size]), action_size);
		}
	}
}
*/
// 将action数组转换为一个int64整数，，前面4*12位为数值位(为了防止溢出，12位中后10位是存数据的),后四位为符号位
ACTION_T ex_rp_encoded_action(PTR ptrER, ACTION_PY* array, int action_size) {
	ACTION_T result = 0;
	ACTION_T sign_bits = 0;
	ACTION_T value_bits = 0;
	for (int i = 0; i < action_size; i++) {
		int sign_bit = array[i] < 0 ? 1 : 0;
		float positive_value = fabs(array[i]);

		// Shift the number left by 10 bits and convert to integer
		ACTION_T shifted_value;

		shifted_value = (ACTION_T)(positive_value * (1 << 10));

		// Append the sign bit to sign_bits
		sign_bits = (sign_bits << 1) | sign_bit;

		// Append the shifted value to value_bits
		value_bits = (value_bits << 12) | shifted_value;

	}

	// Combine the sign bits and value bits
	result = (value_bits << action_size) | sign_bits;

	return result;
}


void ex_rp_decoded_action(ACTION_T encoded_value, ACTION_PY* array, int action_size) {
	// 从编码中提取所有符号位
	ACTION_T signs = encoded_value & 0xF;
	encoded_value = encoded_value >> action_size;
	for (int i = 0; i < action_size; i++) {
		//去掉末尾的符号位
		// 计算每个数值的索引位置
		int index = (action_size - 1 - i) * 12;
		// 提取当前数值的符号位
		int sign_bit = (signs >> (action_size - i - 1)) & 1;
		// 提取数值位
		ACTION_T shifted_value = (encoded_value >> index) & 0xFFF; // 0xFFF 表示 10 位

		// 将提取的整数值转换回浮点数
		float value = (float)shifted_value / (1 << 10);
		// 根据符号位应用符号
		array[i] = sign_bit ? -value : value;
	}
	
}
