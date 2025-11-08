# WheelLib - Deep Learning Library for Computer Vision

A comprehensive C++ deep learning library built on PyTorch C++ (LibTorch) for computer vision tasks.

## Project Structure

```
WheelLib/
├── WheelDL.Lib/                # Main library
│   ├── Config/                 # Configuration management
│   ├── Data/                   # Data loading and processing
│   │   ├── Augmentation/       # Data augmentation
│   │   ├── Cache/              # Caching mechanisms
│   │   ├── Common/             # Common data structures
│   │   ├── Dataset/            # Dataset implementations
│   │   ├── Transforms/         # Image transformations
│   │   └── Utils/              # Data utilities
│   ├── Model/                  # Neural network models
│   │   ├── Builder/            # Model construction
│   │   │   └── Factory/        # Factory patterns for modules
│   │   ├── Loss/               # Loss functions
│   │   ├── Modules/            # Network modules (Conv, Attention, etc.)
│   │   ├── Task/               # Task-specific models
│   │   └── Utils/              # Model utilities
│   └── Utils/                  # General utilities
│       ├── Common/             # Common utilities
│       ├── Error/              # Error handling
│       ├── Logger/             # Logging system
│       ├── Memory/             # Memory management
│       ├── Path/               # Path validation
│       ├── Profiler/           # Performance profiling
│       └── ThreadPool/         # Multi-threading support
├── WheelDL.Lib.Tests/          # Unit tests
└── 3rdparty/                   # Third-party libraries
    └── yaml-cpp-src/           # YAML parser
```

## Features

### Supported Computer Vision Tasks
- **Classification**: Image classification models
- **Object Detection**: Object detection with bounding boxes
- **Segmentation**: Semantic and instance segmentation
- **OBB Detection**: Oriented bounding box detection
- **Anomaly Detection**: Anomaly detection models

### Key Components

#### Data Layer
- Efficient data loading with caching mechanisms
- Comprehensive data augmentation pipeline
- Support for multiple dataset formats
- Batch collation and preprocessing

#### Model Layer
- Modern architecture modules (YOLO-style blocks, Transformers, Attention mechanisms)
- 50+ pre-built neural network modules
- YAML-based model configuration
- Task-specific heads for different CV tasks

#### Training Infrastructure
- Multi-GPU support via CUDA
- Dynamic batch sizing based on GPU memory
- Memory-efficient training with gradient accumulation
- Comprehensive logging and profiling

## Technologies & Frameworks

### Core Dependencies
- **C++ Standard**: C++17
- **Build System**: MSBuild / Visual Studio 2022
- **Platform**: Windows (x64)

### Deep Learning Framework
- **PyTorch C++ (LibTorch)**: Version compatible with CUDA 12.4
- **CUDA**: 12.4 (for GPU acceleration)
- **cuDNN**: For optimized deep learning primitives

### Computer Vision
- **OpenCV**: 4.11.0 (for image processing)

### Other Libraries
- **yaml-cpp**: For YAML configuration parsing
- **spdlog**: Fast C++ logging library
- **Google Test**: For unit testing

## Architecture Highlights

### Memory Management
- Smart pointer usage throughout (unique_ptr, shared_ptr)
- RAII pattern for resource management
- GPU memory pooling and guards
- Dynamic batch sizing based on available memory

### Performance Optimizations
- Multi-threaded data loading
- Efficient tensor operations
- Memory pooling for frequent allocations
- Profiling infrastructure for performance analysis

### Error Handling
- Comprehensive exception hierarchy
- Detailed error messages with context
- Validation at module boundaries
- NaN/Inf detection in loss computation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Licenses

This project uses the following open-source libraries:

- **PyTorch/LibTorch**: BSD-style license
- **OpenCV**: Apache 2.0 License
- **yaml-cpp**: MIT License
- **spdlog**: MIT License
- **Google Test**: BSD 3-Clause License

Please refer to each library's respective license for detailed terms and conditions.

## Project Status

**Current Version**: MVP (Minimum Viable Product)
**Development Stage**: Active Development

## Contributing

This project is currently in early development. Contribution guidelines will be added in future releases.

## Contact

For questions or feedback, please open an issue on GitHub.
