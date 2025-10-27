#pragma once

#include <basetsd.h>

#define WRITE_BYTES(ptr, var_src, as_type)	(*((as_type*)(ptr)) = (var_src), ((char*)(ptr)) + sizeof(as_type))
#define WRITE_BYTES_N(ptr, ptr_src, as_type, n)	(memcpy((ptr), (ptr_src), sizeof(as_type) * (n)), ((char*)(ptr)) + sizeof(as_type) * (n))
#define READ_BYTES(ptr, var_dst, as_type)	((var_dst) = *((as_type*)(ptr)), ((char*)(ptr)) + sizeof(as_type))
#define READ_BYTES_N(ptr, ptr_dst, as_type, n)	(memcpy((ptr_dst), (ptr), sizeof(as_type) * (n)), ((char*)(ptr)) + sizeof(as_type) * (n))
#define AS_ARRAY(VECTOR)	(&(*((VECTOR).begin())))

#define DLL_EXPORT	extern "C" __declspec(dllexport)

typedef INT64	LH_EPI_T;
typedef UINT64	LH_REC_T;
typedef UINT64	PTR;
typedef INT32	H_REC_T;
typedef UINT32	BATCH_SIZE_T;
typedef UINT32	H_PS_T;
typedef UINT64	ACTION_T;
typedef UINT16	STATE_SIZE_T;
typedef float	STATE_T;
typedef float	REWARD_T;
typedef float	PRIORITY_T;
typedef H_REC_T	SEQ_LEN_T;

//Anna
typedef float ACTION_PY;


