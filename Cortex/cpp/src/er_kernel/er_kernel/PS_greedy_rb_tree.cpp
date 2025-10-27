#include "stdafx.h"
#include "PS_greedy_rb_tree.h"
#include "er_kernel.h"

PS_Greedy_RB_Tree::PS_Greedy_RB_Tree() {

}

PS_Greedy_RB_Tree::~PS_Greedy_RB_Tree() {
	sort_tree.clear();
	iters.clear();
}

void PS_Greedy_RB_Tree::select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs) {
	const auto iter_end = sort_tree.rend();
	auto iter = sort_tree.rbegin();
	for (BATCH_SIZE_T i = 0; i < batch_size; ++i, ++iter) {
		if (iter == iter_end) iter = sort_tree.rbegin();
		hPICKs[i] = iter->second;
	}
}

void PS_Greedy_RB_Tree::set_priority(LH_REC_T hPICK, PRIORITY_T prio) {
	auto iter = iters[hPICK];
	iters[hPICK] = sort_tree.insert(sort_tree.begin(), SORT_TREE_NODE(prio, iter->second));
	sort_tree.erase(iter);
}

void PS_Greedy_RB_Tree::reset_priorities() {
	sort_tree.clear();
	LH_REC_T pick_num = iters.size();
	PRIORITY_T prio = FLT_MAX;
	for (LH_REC_T hPICK = 0; hPICK < pick_num; ++hPICK) {
		iters[hPICK] = sort_tree.insert(sort_tree.begin(), SORT_TREE_NODE(prio, hPICK));
	}
}

void PS_Greedy_RB_Tree::on_pick_table_push_back() {
	PRIORITY_T prio = FLT_MAX;
	LH_REC_T hPICK = iters.size();
	iters.push_back(sort_tree.insert(sort_tree.end(), SORT_TREE_NODE(prio, hPICK)));
}

void PS_Greedy_RB_Tree::on_pick_table_remove_tail(LH_REC_T new_end) {
	for (auto iter = iters.begin() + new_end; iter != iters.end(); ++iter)
		sort_tree.erase(*iter);
	iters.erase(iters.begin() + new_end, iters.end());
}

void PS_Greedy_RB_Tree::on_pick_exchange(LH_REC_T hPICK0, LH_REC_T hPICK1) {
	auto iter0 = iters[hPICK0];
	auto iter1 = iters[hPICK1];
	iters[hPICK0] = iter1;
	iters[hPICK1] = iter0;
	iter0->second = hPICK1;
	iter1->second = hPICK0;
}

UINT64 PS_Greedy_RB_Tree::serialize_size() {
	LH_REC_T pick_num = iters.size();
	return sizeof(pick_num) + sizeof(PRIORITY_T) * pick_num;
}

char* PS_Greedy_RB_Tree::serialize(char* pSer) {
	LH_REC_T pick_num = iters.size();
	pSer = WRITE_BYTES(pSer, pick_num, LH_REC_T);
	for (auto iter = iters.begin(); iter != iters.end(); ++iter)
		pSer = WRITE_BYTES(pSer, (*iter)->first, PRIORITY_T);
	return pSer;
}

char* PS_Greedy_RB_Tree::unserialize(char* pSer) {
	LH_REC_T pick_num;
	iters.clear();
	sort_tree.clear();
	pSer = READ_BYTES(pSer, pick_num, LH_REC_T);
	for (LH_REC_T hPICK = 0; hPICK < pick_num; ++hPICK) {
		PRIORITY_T prio;
		pSer = READ_BYTES(pSer, prio, PRIORITY_T);
		iters.push_back(sort_tree.insert(sort_tree.begin(), SORT_TREE_NODE(prio, hPICK)));
	}
	return pSer;
}

H_PS_T ex_rp_new_pick_selector_greedy_rb_tree(PTR ptrER) {
	return ex_rp_new_pick_selector<PS_Greedy_RB_Tree>(ptrER);
}