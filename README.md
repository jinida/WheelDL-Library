# WheelLib - Deep Learning Library for Computer Vision

A comprehensive C++ deep learning library built on PyTorch C++ (LibTorch) for computer vision tasks.

## Features

### Supported Computer Vision Tasks

- **Classification**: Image classification with modern CNN architectures
- **Object Detection**: YOLO-style object detection with bounding boxes
- **Segmentation**: Semantic and instance segmentation
- **OBB Detection**: Oriented bounding box detection for rotated objects
- **Anomaly Detection**: Industrial anomaly detection with multiple methods
  - **EfficientAD**: Fast anomaly detection with teacher-student framework
  - **PatchCore**: Training-free memory bank approach with k-center greedy sampling

### Model Layer

#### Architecture Components
- **50+ Pre-built Modules**: Conv, DWConv, Focus, SPPF, C2f, C3k2, etc.
- **Attention Mechanisms**: PSA, CBAM, Spatial Attention
- **Transformer Blocks**: Vision transformers and hybrid architectures
- **Task-Specific Heads**: Detect, Classify, Segment, OBB, Anomaly

#### Model Configuration
- **YAML-based Configuration**: Define models declaratively
- **Factory Pattern**: Runtime model instantiation
- **Flexible Architecture**: Mix and match modules easily

### Data Layer

- **Efficient Data Loading**: Multi-threaded with caching
- **Dataset Support**:
  - Classification (ImageNet-style)
  - Detection (YOLO format)
  - Segmentation (mask annotations)
  - OBB (oriented bounding boxes)
  - Anomaly Detection (MVTec AD format)
- **Data Augmentation**:
  - Geometric: Resize, Crop, Flip, Rotate, Perspective
  - Color: HSV, ColorJitter, CLAHE
  - Advanced: Mosaic, MixUp, CopyPaste

### Training Infrastructure

- **Multi-GPU Support**: CUDA 12.4 acceleration
- **Dynamic Batch Sizing**: Automatic GPU memory management
- **Loss Functions**:
  - Classification: CrossEntropy, BCE
  - Detection: DFL, CIOU, GIoU
  - Segmentation: BCE+Dice
  - Anomaly: EfficientAD, PatchCore
- **Memory Optimization**: Gradient accumulation, mixed precision support

## Technologies & Frameworks

### Core Dependencies
- **C++ Standard**: C++17
- **Build System**: MSBuild / Visual Studio 2022
- **Platform**: Windows (x64)

### Deep Learning Framework
- **PyTorch C++ (LibTorch)**: Version compatible with CUDA 12.4
- **CUDA**: 12.4 (for GPU acceleration)
- **cuDNN**: Optimized deep learning primitives

### Computer Vision
- **OpenCV**: 4.11.0 (image processing and I/O)

### Other Libraries
- **yaml-cpp**: YAML configuration parsing
- **spdlog**: Fast C++ logging library
- **Google Test**: Unit testing framework

## Project Structure

```
WheelLib/
├── WheelDL.Lib/                # Main library
│   ├── Config/                 # Configuration management
│   ├── Data/                   # Data loading and processing
│   │   ├── Augmentation/       # Data augmentation
│   │   ├── Cache/              # Caching mechanisms
│   │   ├── Dataset/            # Dataset implementations
│   │   ├── Transforms/         # Image transformations
│   │   └── Utils/              # Data utilities
│   ├── Model/                  # Neural network models
│   │   ├── Builder/            # Model construction
│   │   │   └── Factory/        # Factory patterns for modules
│   │   ├── Loss/               # Loss functions
│   │   ├── Modules/            # Network modules
│   │   │   ├── Block.cpp       # Basic building blocks
│   │   │   ├── Conv.cpp        # Convolution layers
│   │   │   ├── Head.cpp        # Task-specific heads
│   │   │   ├── Model.cpp       # Anomaly detection models
│   │   │   └── Transformer.cpp # Transformer modules
│   │   ├── Task/               # Task-specific models
│   │   │   ├── ClassificationModel
│   │   │   ├── DetectionModel
│   │   │   ├── SegmentationModel
│   │   │   └── AnomalyModel
│   │   └── Utils/              # Model utilities
│   └── Utils/                  # General utilities
│       ├── Logger/             # Logging system (spdlog)
│       ├── Memory/             # Memory management
│       ├── Profiler/           # Performance profiling
│       └── ThreadPool/         # Multi-threading support
├── WheelDL.Lib.Tests/          # Unit and integration tests
└── 3rdparty/                   # Third-party libraries
    └── yaml-cpp-src/           # YAML parser
```

## Architecture Highlights

### Memory Management
- Smart pointer usage (unique_ptr, shared_ptr)
- RAII pattern for resource management
- GPU memory pooling and guards
- Dynamic batch sizing based on available memory
- 5GB memory limit for PatchCore memory bank building

### Performance Optimizations
- Multi-threaded data loading
- Efficient tensor operations
- Memory pooling for frequent allocations
- Profiling infrastructure for bottleneck analysis
- Batch processing for k-center greedy algorithm

### Error Handling
- Comprehensive exception hierarchy
- Detailed error messages with context
- Validation at module boundaries
- NaN/Inf detection in loss computation

### Design Patterns
- **Factory Pattern**: Dynamic module instantiation
- **Strategy Pattern**: Interchangeable loss functions
- **Template Method**: Dataset loading workflows
- **RAII**: Automatic resource management

## Example: Anomaly Detection Model Configuration

### EfficientAD
```yaml
head:
  - [-1, 1, Anomaly, [EfficientAD, 384, true]]  # model_type, out_channels, small
```

### PatchCore
```yaml
head:
  - [-1, 1, Anomaly, [PatchCore, 9, 25600]]  # model_type, num_neighbors, max_memory_patches
```

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

### Recent Updates
- Added comprehensive anomaly detection support
- Implemented EfficientAD and PatchCore models
- Enhanced factory pattern for runtime model creation
- Improved memory management for large-scale feature extraction

## Contributing

This project is currently in early development. Contribution guidelines will be added in future releases.

## Contact

For questions or feedback, please open an issue on GitHub.
