#include "stdafx.h"
#include "PS_greedy_bin_heap.h"
#include "er_kernel.h"

PS_Greedy_Bin_Heap::PS_Greedy_Bin_Heap() {
}

PS_Greedy_Bin_Heap::~PS_Greedy_Bin_Heap() {
}

void PS_Greedy_Bin_Heap::select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs) {
	LH_REC_T pick_num = size();
	LH_REC_T* hPICKs_0 = 0;
	while (((LH_REC_T)batch_size) >= pick_num) {
		if (!hPICKs_0) {
			hPICKs_0 = new LH_REC_T[pick_num];
			for (LH_REC_T h = 0; h < pick_num; h++) hPICKs_0[h] = h;
		}
		memcpy(hPICKs, hPICKs_0, sizeof(LH_REC_T) * pick_num);
		hPICKs += pick_num;
#pragma warning(push)
#pragma warning(disable: 4244)
		batch_size -= pick_num;
#pragma warning(pop)
	}
	if (hPICKs_0) delete[] hPICKs_0;
	top_n_at(batch_size, hPICKs);
}

void PS_Greedy_Bin_Heap::set_priority(LH_REC_T hPICK, PRIORITY_T prio) {
	set_weight(hPICK, prio);
}

void PS_Greedy_Bin_Heap::reset_priorities() {
	set_all_weight(FLT_MAX);
}

void PS_Greedy_Bin_Heap::on_pick_table_push_back() {
	push_back(FLT_MAX, 0);
}

void PS_Greedy_Bin_Heap::on_pick_table_remove_tail(LH_REC_T new_end) {
	remove_tail(new_end);
}

void PS_Greedy_Bin_Heap::on_pick_exchange(LH_REC_T hPICK0, LH_REC_T hPICK1) {
	exchange(hPICK0, hPICK1);
}

UINT64 PS_Greedy_Bin_Heap::serialize_size() {
	LH_REC_T pick_num = size();
	return sizeof(pick_num) + sizeof(PRIORITY_T) * pick_num;
}

char* PS_Greedy_Bin_Heap::serialize(char* pSer) {
	LH_REC_T pick_num = size();
	pSer = WRITE_BYTES(pSer, pick_num, LH_REC_T);
	for (auto iter = W.begin(); iter != W.end(); ++iter)
		pSer = WRITE_BYTES(pSer, iter->first, PRIORITY_T);
	return pSer;
}

char* PS_Greedy_Bin_Heap::unserialize(char* pSer) {
	LH_REC_T pick_num;
	V.clear();
	P.clear();
	W.clear();
	pSer = READ_BYTES(pSer, pick_num, LH_REC_T);
	for (LH_REC_T hPICK = 0; hPICK < pick_num; ++hPICK) {
		PRIORITY_T prio;
		pSer = READ_BYTES(pSer, prio, PRIORITY_T);
		push_back(prio, 0);
	}
	return pSer;
}

H_PS_T ex_rp_new_pick_selector_greedy_bin_heap(PTR ptrER) {
	return ex_rp_new_pick_selector<PS_Greedy_Bin_Heap>(ptrER);
}