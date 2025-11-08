//
// pch.h
//

#pragma once

// Include Windows headers first
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // Prevent Windows min/max macros from conflicting with std::min/max and LibTorch
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

// Undefine Windows macros that conflict with our code
#ifdef ERROR
#undef ERROR
#endif

#ifdef FATAL
#undef FATAL
#endif

#include "gtest/gtest.h"
