#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define NOMINMAX                        // Prevent Windows min/max macros from conflicting with std::min/max
// Windows Header Files
#include <windows.h>

#ifdef FATAL
#undef FATAL
#endif
