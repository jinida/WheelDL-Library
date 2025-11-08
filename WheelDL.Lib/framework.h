#pragma once

#define WIN32_LEAN_AND_MEAN             // 거의 사용되지 않는 내용을 Windows 헤더에서 제외합니다.
#define NOMINMAX                        // Prevent Windows min/max macros from conflicting with std::min/max
// Windows 헤더 파일
#include <windows.h>

#ifdef FATAL
#undef FATAL
#endif
