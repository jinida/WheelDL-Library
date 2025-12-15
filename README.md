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

- **Core Engine**: Unified training/validation/prediction pipeline
  - Task-specific Trainers, Validators, Predictors
  - TaskManager for async task management
  - Checkpoint save/load with metadata
- **Multi-GPU Support**: CUDA 12.4 acceleration
- **Optimizer Support**: SGD, Adam, AdamW with factory pattern
- **LR Schedulers**: Cosine Annealing, Linear warmup
- **Training Utilities**: EarlyStopping, Model EMA
- **Loss Functions**:
  - Classification: CrossEntropy, BCE
  - Detection: DFL, CIOU, GIoU
  - Segmentation: BCE+Dice
  - Anomaly: EfficientAD, PatchCore
- **Memory Optimization**: GPU memory pooling, gradient accumulation

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
WheelDL-Library/
├── WheelDL.Lib/                    # Main library
│   ├── Config/                     # Configuration management
│   │   ├── Configuration           # Main config class
│   │   ├── JsonParser              # JSON parsing
│   │   └── YamlParser              # YAML parsing
│   ├── Core/                       # Core training/inference engine
│   │   ├── Callback/               # Async callback system
│   │   ├── Engine/                 # Base trainer/validator/predictor
│   │   ├── Manager/                # Task management & context
│   │   ├── Predictor/              # Task-specific predictors
│   │   ├── Trainer/                # Task-specific trainers
│   │   ├── Utils/                  # Checkpoint utilities
│   │   └── Validator/              # Task-specific validators
│   ├── Data/                       # Data loading and processing
│   │   ├── Augmentation/           # Mosaic, MixUp augmentation
│   │   ├── Cache/                  # RAM/Disk caching
│   │   ├── Common/                 # Annotation handling
│   │   ├── Dataset/                # Dataset implementations
│   │   ├── Transforms/             # Color/Geometric transforms
│   │   └── Utils/                  # Image I/O utilities
│   ├── Model/                      # Neural network models
│   │   ├── Builder/                # Model construction
│   │   │   └── Factory/            # Module factories (Conv, Block, Attention, etc.)
│   │   ├── Loss/                   # Loss functions per task
│   │   ├── Modules/                # Network modules (50+ types)
│   │   ├── Task/                   # Task-specific model wrappers
│   │   └── Utils/                  # IoU, NMS, TaskAlignedAssigner
│   ├── Optimizer/                  # Optimization utilities
│   │   ├── EarlyStopping/          # Early stopping callback
│   │   ├── EMA/                    # Exponential moving average
│   │   ├── Scheduler/              # LR schedulers (Cosine, Linear)
│   │   └── OptimizerFactory        # Optimizer creation (SGD, Adam, AdamW)
│   └── Utils/                      # General utilities
│       ├── Common/                 # Types, Random, Timer, Constants
│       ├── Error/                  # Exception handling
│       ├── Export/                 # Image/Metrics exporters
│       ├── Logger/                 # Logging system (spdlog)
│       ├── Memory/                 # GPU memory management
│       ├── Path/                   # Path validation
│       ├── Profiler/               # Performance profiling
│       ├── ThreadPool/             # Multi-threading support
│       └── Workspace/              # Workspace management
├── WheelDL.Lib.Tests/              # Unit tests
│   └── Unit/                       # Component-based test structure
│       ├── Config/                 # Configuration tests
│       ├── Data/                   # Data pipeline tests
│       ├── Model/                  # Model & module tests
│       ├── Optimizer/              # Optimizer tests
│       └── Utils/                  # Utility tests
└── 3rdparty/                       # Third-party libraries
```

## Architecture Highlights

### Memory Management
- Smart pointer usage (unique_ptr, shared_ptr)
- RAII pattern for resource management
- GPU memory pooling and guards
- GPU memory monitoring and management
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
