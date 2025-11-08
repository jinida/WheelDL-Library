#pragma once

// DLL export/import macros
// Only use dllexport/dllimport if building as DLL
#ifdef WHEELDLLIB_EXPORTS
    #define WHEELDL_API __declspec(dllexport)
#elif defined(_WINDLL)
    #define WHEELDL_API __declspec(dllimport)
#else
    // Static library - no export needed
    #define WHEELDL_API
#endif
