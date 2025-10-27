#pragma once

#include <vector>
#include <map>

using namespace std;

#include "common.h"

class PickSelector {
public:
	~PickSelector() {}

	virtual void select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs) = 0;
	virtual void set_priority(LH_REC_T hPICK, PRIORITY_T prio) = 0;
	virtual void reset_priorities() = 0;
	virtual void on_pick_table_push_back() = 0;
	virtual void on_pick_table_remove_tail(LH_REC_T new_end) = 0;
	virtual void on_pick_exchange(LH_REC_T hPICK0, LH_REC_T hPICK1) = 0;
	virtual UINT64 serialize_size() = 0;
	virtual char* serialize(char* pSer) = 0;
	virtual char* unserialize(char* pSer) = 0;
};

typedef vector<PickSelector*> PS_TABLE;

class PS_UniRand : public PickSelector {
private:
	LH_REC_T pick_num;

public:
	PS_UniRand();
	~PS_UniRand() {}

	virtual void select(BATCH_SIZE_T batch_size, LH_REC_T* hPICKs);
	virtual void set_priority(LH_REC_T hPICK, PRIORITY_T prio) {}
	virtual void reset_priorities() {}
	virtual void on_pick_table_push_back() { ++pick_num; }
	virtual void on_pick_table_remove_tail(LH_REC_T new_end) { pick_num = new_end; }
	virtual void on_pick_exchange(LH_REC_T hPICK0, LH_REC_T hPICK1) {}
	virtual UINT64 serialize_size() { return sizeof(LH_REC_T); }
	virtual char* serialize(char* pSer);
	virtual char* unserialize(char* pSer);
};

