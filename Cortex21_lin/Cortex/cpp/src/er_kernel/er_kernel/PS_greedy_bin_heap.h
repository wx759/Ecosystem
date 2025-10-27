#pragma once

#include "PickSelector.h"
#include "WeightedVector.h"

class PS_Greedy_Bin_Heap : public PickSelector, 
						   public WeightedVector<PRIORITY_T> {
public:
	PS_Greedy_Bin_Heap();
	~PS_Greedy_Bin_Heap();

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

DLL_EXPORT H_PS_T ex_rp_new_pick_selector_greedy_bin_heap(PTR ptrER);
