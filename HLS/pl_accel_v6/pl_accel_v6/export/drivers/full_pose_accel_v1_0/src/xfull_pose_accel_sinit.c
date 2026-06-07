// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xfull_pose_accel.h"

extern XFull_pose_accel_Config XFull_pose_accel_ConfigTable[];

XFull_pose_accel_Config *XFull_pose_accel_LookupConfig(u16 DeviceId) {
	XFull_pose_accel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XFULL_POSE_ACCEL_NUM_INSTANCES; Index++) {
		if (XFull_pose_accel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XFull_pose_accel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XFull_pose_accel_Initialize(XFull_pose_accel *InstancePtr, u16 DeviceId) {
	XFull_pose_accel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XFull_pose_accel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XFull_pose_accel_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

