// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XFULL_POSE_ACCEL_H
#define XFULL_POSE_ACCEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xfull_pose_accel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XFull_pose_accel_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XFull_pose_accel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XFull_pose_accel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XFull_pose_accel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XFull_pose_accel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XFull_pose_accel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XFull_pose_accel_Initialize(XFull_pose_accel *InstancePtr, u16 DeviceId);
XFull_pose_accel_Config* XFull_pose_accel_LookupConfig(u16 DeviceId);
int XFull_pose_accel_CfgInitialize(XFull_pose_accel *InstancePtr, XFull_pose_accel_Config *ConfigPtr);
#else
int XFull_pose_accel_Initialize(XFull_pose_accel *InstancePtr, const char* InstanceName);
int XFull_pose_accel_Release(XFull_pose_accel *InstancePtr);
#endif

void XFull_pose_accel_Start(XFull_pose_accel *InstancePtr);
u32 XFull_pose_accel_IsDone(XFull_pose_accel *InstancePtr);
u32 XFull_pose_accel_IsIdle(XFull_pose_accel *InstancePtr);
u32 XFull_pose_accel_IsReady(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_EnableAutoRestart(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_DisableAutoRestart(XFull_pose_accel *InstancePtr);

void XFull_pose_accel_Set_input_r(XFull_pose_accel *InstancePtr, u64 Data);
u64 XFull_pose_accel_Get_input_r(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_Set_weights(XFull_pose_accel *InstancePtr, u64 Data);
u64 XFull_pose_accel_Get_weights(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_Set_command(XFull_pose_accel *InstancePtr, u32 Data);
u32 XFull_pose_accel_Get_command(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_Set_final_pose(XFull_pose_accel *InstancePtr, u64 Data);
u64 XFull_pose_accel_Get_final_pose(XFull_pose_accel *InstancePtr);

void XFull_pose_accel_InterruptGlobalEnable(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_InterruptGlobalDisable(XFull_pose_accel *InstancePtr);
void XFull_pose_accel_InterruptEnable(XFull_pose_accel *InstancePtr, u32 Mask);
void XFull_pose_accel_InterruptDisable(XFull_pose_accel *InstancePtr, u32 Mask);
void XFull_pose_accel_InterruptClear(XFull_pose_accel *InstancePtr, u32 Mask);
u32 XFull_pose_accel_InterruptGetEnabled(XFull_pose_accel *InstancePtr);
u32 XFull_pose_accel_InterruptGetStatus(XFull_pose_accel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
