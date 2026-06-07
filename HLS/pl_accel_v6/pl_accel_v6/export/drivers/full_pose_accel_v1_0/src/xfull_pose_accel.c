// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xfull_pose_accel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XFull_pose_accel_CfgInitialize(XFull_pose_accel *InstancePtr, XFull_pose_accel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XFull_pose_accel_Start(XFull_pose_accel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XFull_pose_accel_IsDone(XFull_pose_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XFull_pose_accel_IsIdle(XFull_pose_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XFull_pose_accel_IsReady(XFull_pose_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XFull_pose_accel_EnableAutoRestart(XFull_pose_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XFull_pose_accel_DisableAutoRestart(XFull_pose_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XFull_pose_accel_Set_input_r(XFull_pose_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XFull_pose_accel_Get_input_r(XFull_pose_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XFull_pose_accel_Set_weights(XFull_pose_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XFull_pose_accel_Get_weights(XFull_pose_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XFull_pose_accel_Set_command(XFull_pose_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_COMMAND_DATA, Data);
}

u32 XFull_pose_accel_Get_command(XFull_pose_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_COMMAND_DATA);
    return Data;
}

void XFull_pose_accel_Set_final_pose(XFull_pose_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_FINAL_POSE_DATA, (u32)(Data));
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_FINAL_POSE_DATA + 4, (u32)(Data >> 32));
}

u64 XFull_pose_accel_Get_final_pose(XFull_pose_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_FINAL_POSE_DATA);
    Data += (u64)XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_FINAL_POSE_DATA + 4) << 32;
    return Data;
}

void XFull_pose_accel_InterruptGlobalEnable(XFull_pose_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_GIE, 1);
}

void XFull_pose_accel_InterruptGlobalDisable(XFull_pose_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_GIE, 0);
}

void XFull_pose_accel_InterruptEnable(XFull_pose_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_IER);
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_IER, Register | Mask);
}

void XFull_pose_accel_InterruptDisable(XFull_pose_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_IER);
    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XFull_pose_accel_InterruptClear(XFull_pose_accel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFull_pose_accel_WriteReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_ISR, Mask);
}

u32 XFull_pose_accel_InterruptGetEnabled(XFull_pose_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_IER);
}

u32 XFull_pose_accel_InterruptGetStatus(XFull_pose_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XFull_pose_accel_ReadReg(InstancePtr->Control_BaseAddress, XFULL_POSE_ACCEL_CONTROL_ADDR_ISR);
}

