#pragma once

#include <vector>
#include <map>
using namespace std;

#include "common.h"
#include "PickSelector.h"

typedef vector<STATE_T>		STATE_TABLE;
typedef vector<ACTION_T>	ACTION_TABLE;
typedef vector<REWARD_T>	REWARD_TABLE;
typedef vector<LH_REC_T>	PICK_INDEX;

struct EPISODE {
	STATE_TABLE states;
	ACTION_TABLE actions;
	REWARD_TABLE rewards;
	PICK_INDEX pick_index;
	H_REC_T record_num;
	UCHAR access_flag;
	bool terminated;

	~EPISODE(){
		actions.clear();
		rewards.clear();
		states.clear();
		pick_index.clear();
	}
};

typedef map<LH_EPI_T, EPISODE*>	EPI_TABLE;

struct PICK {
	LH_EPI_T hEPI;
	EPISODE* pEPI;
	H_REC_T pos;
};

typedef vector<PICK> PICK_TABLE;

struct BATCH_MAKER {
	PICK_TABLE pick_table;
	PS_TABLE ps_table;
	SEQ_LEN_T pick_len;
	bool allow_short_seq;

	~BATCH_MAKER(){
		const auto iter_ps_end = ps_table.end();
		for (auto iter = ps_table.begin(); iter != iter_ps_end; ++iter) delete *iter;
		ps_table.clear();
		pick_table.clear();
	}
};

struct EX_RP {
	EPI_TABLE episodes;
	BATCH_MAKER batch_maker;
	LH_REC_T max_record_num;
	LH_REC_T record_num;
	LH_EPI_T head_handle;
	LH_EPI_T tail_handle;
	LH_EPI_T remove_handle;
	STATE_SIZE_T state_size;

	~EX_RP() {
		const EPI_TABLE::iterator iter_epi_end = episodes.end();
		for (EPI_TABLE::iterator iter = episodes.begin(); iter != iter_epi_end; ++iter) delete iter->second;
		episodes.clear();
	}
};

template<class T>
H_PS_T ex_rp_new_pick_selector(PTR ptrER) {
	EX_RP* pER = (EX_RP*)ptrER;
#pragma warning(push)
#pragma warning(disable: 4267)
	H_PS_T hPS = pER->batch_maker.ps_table.size();
#pragma warning(pop)
	pER->batch_maker.ps_table.push_back(new T());
	return hPS;
}

DLL_EXPORT PTR ex_rp_new(LH_REC_T max_record_num, SEQ_LEN_T pick_len, INT32 allow_short_seq);
DLL_EXPORT void ex_rp_del(PTR ptrER);
DLL_EXPORT void ex_rp_clear(PTR ptrER);

DLL_EXPORT UINT64 ex_rp_serialize_size(PTR ptrER);
DLL_EXPORT void ex_rp_serialize(PTR ptrER, char* pSer);
DLL_EXPORT void ex_rp_unserialize(PTR ptrER, char* pSer);

DLL_EXPORT void ex_rp_get_pick_policy(PTR ptrER, SEQ_LEN_T* p_pick_len, bool* p_allow_short_seq);

DLL_EXPORT LH_EPI_T ex_rp_new_episode(PTR ptrER);
DLL_EXPORT LH_EPI_T ex_rp_record(PTR ptrER, LH_EPI_T hEPI, STATE_T* state, ACTION_T action, REWARD_T reward,
									  STATE_T* final_state, STATE_SIZE_T state_size);
DLL_EXPORT INT32 ex_rp_get_random_batch(PTR ptrER, H_PS_T hPS, BATCH_SIZE_T batch_size, float valid_sample_rate,
										LH_EPI_T* pick_epi, H_REC_T* pick_pos, STATE_T* state, ACTION_T* action,
										REWARD_T* reward, STATE_T* state_, SEQ_LEN_T* seq_len, SEQ_LEN_T* seq_len_);

DLL_EXPORT void ex_rp_set_elimination_policy(PTR ptrER, INT32 SecChanElimination);

DLL_EXPORT void ex_rp_set_pick_priority(PTR ptrER, H_PS_T hPS, LH_EPI_T* pick_epi, H_REC_T* pick_pos,
										PRIORITY_T* prio, BATCH_SIZE_T batch_size);
DLL_EXPORT void ex_rp_reset_pick_priorities(PTR ptrER, H_PS_T hPS);

DLL_EXPORT void ex_rp_del_pick_selector(PTR ptrER, H_PS_T hPS);

// Anna´Û¸Ä
DLL_EXPORT ACTION_T ex_rp_encoded_action(PTR ptrER, ACTION_PY* array, int action_size);

DLL_EXPORT void ex_rp_decoded_batch_actions(PTR ptrER, ACTION_T* encoded_values, ACTION_PY* actions, BATCH_SIZE_T batchsize, int picklen, int action_size);

DLL_EXPORT void ex_rp_decoded_action(ACTION_T encoded_value, ACTION_PY* array, int action_size);




