#pragma once

#include "PickSelector.h"

class PS_Greedy_RB_Tree : public PickSelector {
private:
	using SORT_TREE = multimap<PRIORITY_T, LH_REC_T>;
	using SORT_TREE_NODE = SORT_TREE::value_type;
	using SORT_TREE_ITER = SORT_TREE::iterator;

	SORT_TREE sort_tree;
	vector<SORT_TREE_ITER> iters;

public:
	PS_Greedy_RB_Tree();
	~PS_Greedy_RB_Tree();

	virtual void select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs);
	virtual void set_priority(LH_REC_T hPICK, PRIORITY_T prio);
	virtual void reset_priorities();
	virtual void on_pick_table_push_back();
	virtual void on_pick_table_remove_tail(LH_REC_T new_end);
	virtual void on_pick_exchange(LH_REC_T hPICK0, LH_REC_T hPICK1);
	virtual UINT64 serialize_size();
	virtual char* serialize(char* pSer);
	virtual char* unserialize(char* pSer);
};

DLL_EXPORT H_PS_T ex_rp_new_pick_selector_greedy_rb_tree(PTR ptrER);