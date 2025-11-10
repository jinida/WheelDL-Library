#include "pch.h"
#include "ColorTransforms.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <algorithm>
#include <mutex>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// CPU feature detection
namespace
{
    bool hasAVX2()
    {
        static std::once_flag initFlag;
        static bool hasAVX2Support = false;

        std::call_once(initFlag, []() {
#ifdef _MSC_VER
            int cpuInfo[4];
            __cpuid(cpuInfo, 0);
            int nIds = cpuInfo[0];

            if (nIds >= 7) {
                __cpuidex(cpuInfo, 7, 0);
                hasAVX2Support = (cpuInfo[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
            } else {
                hasAVX2Support = false;
            }
#else
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                hasAVX2Support = (ebx & (1 << 5)) != 0;
            } else {
                hasAVX2Support = false;
            }
#endif
        });

        return hasAVX2Support;
    }
}

#if defined(_MSC_VER) || defined(__AVX2__)
#include <immintrin.h>

// AVX2-optimized ToTensor: uint8 -> float with optional BGR->RGB
static void toTensorAVX2(uint8_t* srcData, float* dstData, int totalPixels, bool bgrToRgb)
{
    const float scale = 1.0f / 255.0f;
    __m256 scale_vec = _mm256_set1_ps(scale);

    int i = 0;
    const int simd_end = (totalPixels / 8) * 8;

    // Branch prediction optimization: check bgrToRgb once outside the loop
    // This avoids redundant SIMD operations when BGR→RGB conversion is needed
    if (bgrToRgb)
    {
        // BGR→RGB path: Use scalar code for all pixels
        // AVX2 BGR→RGB requires complex cross-vector shuffles, so scalar is more efficient
        for (i = 0; i < totalPixels; ++i)
        {
            int idx = i * 3;
            dstData[idx]     = srcData[idx + 2] * scale;  // R
            dstData[idx + 1] = srcData[idx + 1] * scale;  // G
            dstData[idx + 2] = srcData[idx]     * scale;  // B
        }
    }
    else
    {
        // BGR path: Use SIMD for bulk processing
        for (; i < simd_end; i += 8)
        {
            int src_idx = i * 3;
            int dst_idx = i * 3;

            // Load 24 uint8 values (8 BGR pixels)
            // Can't load 24 bytes directly into __m256i, so load 32 and ignore last 8
            __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&srcData[src_idx]));      // 16 bytes
            __m128i high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&srcData[src_idx + 16])); // 8 bytes

            // Combine to get 24 bytes
            __m256i bytes24_extended = _mm256_set_m128i(high, low);

            // Extract first 8 bytes (B0 G0 R0 B1 G1 R1 B2 G2) -> convert to float
            __m128i bytes_low = _mm256_castsi256_si128(bytes24_extended);
            __m256i ints_low = _mm256_cvtepu8_epi32(bytes_low);
            __m256 floats_0 = _mm256_cvtepi32_ps(ints_low);
            floats_0 = _mm256_mul_ps(floats_0, scale_vec);

            // Extract next 8 bytes (R2 B3 G3 R3 B4 G4 R4 B5) -> convert to float
            __m128i bytes_mid = _mm_srli_si128(bytes_low, 8);
            __m256i ints_mid = _mm256_cvtepu8_epi32(bytes_mid);
            __m256 floats_1 = _mm256_cvtepi32_ps(ints_mid);
            floats_1 = _mm256_mul_ps(floats_1, scale_vec);

            // Extract last 8 bytes (G5 R5 B6 G6 R6 B7 G7 R7) -> convert to float
            __m128i bytes_high = _mm256_extracti128_si256(bytes24_extended, 1);
            __m256i ints_high = _mm256_cvtepu8_epi32(bytes_high);
            __m256 floats_2 = _mm256_cvtepi32_ps(ints_high);
            floats_2 = _mm256_mul_ps(floats_2, scale_vec);

            // Store results
            _mm256_storeu_ps(&dstData[dst_idx], floats_0);
            _mm256_storeu_ps(&dstData[dst_idx + 8], floats_1);
            _mm256_storeu_ps(&dstData[dst_idx + 16], floats_2);
        }

        // Handle remaining pixels with scalar code
        for (; i < totalPixels; ++i)
        {
            int idx = i * 3;
            dstData[idx]     = srcData[idx]     * scale;  // B
            dstData[idx + 1] = srcData[idx + 1] * scale;  // G
            dstData[idx + 2] = srcData[idx + 2] * scale;  // R
        }
    }
}

// Compute 1D Gaussian kernel
static std::vector<float> computeGaussianKernel(int kernelSize)
{
    std::vector<float> kernel(kernelSize);
    float sigma = 0.3f * ((kernelSize - 1) * 0.5f - 1.0f) + 0.8f;
    float sum = 0.0f;
    int center = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i)
    {
        float x = static_cast<float>(i - center);
        kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for (int i = 0; i < kernelSize; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

// Cache-friendly transpose using block tiling
static void transpose(const float* src, float* dst, int width, int height)
{
    const int BLOCK_SIZE = 32;  // Tuned for cache line size

    for (int y0 = 0; y0 < height; y0 += BLOCK_SIZE)
    {
        for (int x0 = 0; x0 < width; x0 += BLOCK_SIZE)
        {
            int yMax = std::min(y0 + BLOCK_SIZE, height);
            int xMax = std::min(x0 + BLOCK_SIZE, width);

            for (int y = y0; y < yMax; ++y)
            {
                for (int x = x0; x < xMax; ++x)
                {
                    dst[x * height + y] = src[y * width + x];
                }
            }
        }
    }
}

// AVX2-optimized 1D horizontal convolution for a single channel (planar float data)
static void convolve1D_AVX2(const float* src, float* dst, int length,
                            const float* kernel, int kernelSize)
{
    int radius = kernelSize / 2;

    for (int x = 0; x < length; ++x)
    {
        __m256 sum_vec = _mm256_setzero_ps();

        int k = 0;
        // Process 8 kernel weights at a time
        for (; k + 7 < kernelSize; k += 8)
        {
            // Load 8 kernel weights
            __m256 weights = _mm256_loadu_ps(&kernel[k]);

            // Load 8 pixel values using AVX2 gather (with border handling via SIMD clamp)
            // Create indices: [x+k-radius, x+k-radius+1, ..., x+k-radius+7]
            __m256i base_indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            __m256i offset = _mm256_set1_epi32(x + k - radius);
            __m256i indices = _mm256_add_epi32(offset, base_indices);

            // Clamp indices to [0, length-1] using SIMD
            __m256i min_idx = _mm256_setzero_si256();
            __m256i max_idx = _mm256_set1_epi32(length - 1);
            indices = _mm256_max_epi32(indices, min_idx);
            indices = _mm256_min_epi32(indices, max_idx);

            // Gather 8 floats at once using computed indices
            __m256 pixels_vec = _mm256_i32gather_ps(src, indices, 4); // scale = 4 bytes per float

            // FMA: sum += pixels * weights
            sum_vec = _mm256_fmadd_ps(pixels_vec, weights, sum_vec);
        }

        // Horizontal reduction: sum all 8 floats in sum_vec
        __m128 low = _mm256_castps256_ps128(sum_vec);
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Handle remaining kernel elements (scalar)
        for (; k < kernelSize; ++k)
        {
            int srcX = std::clamp(x + k - radius, 0, length - 1);
            sum += src[srcX] * kernel[k];
        }

        dst[x] = sum;
    }
}

// AVX2-optimized separable Gaussian blur for BGR uint8 images
static void gaussianBlurSeparableAVX2(cv::Mat& image, int kernelSize)
{
    if (kernelSize % 2 == 0 || kernelSize < 3)
    {
        throw std::invalid_argument("Kernel size must be odd and >= 3");
    }

    int width = image.cols;
    int height = image.rows;
    int radius = kernelSize / 2;

    // Compute Gaussian kernel
    std::vector<float> kernel = computeGaussianKernel(kernelSize);

    // Planar storage for intermediate results (B, G, R separate)
    std::vector<float> planarB(width * height);
    std::vector<float> planarG(width * height);
    std::vector<float> planarR(width * height);

    std::vector<float> tempB(width * height);
    std::vector<float> tempG(width * height);
    std::vector<float> tempR(width * height);

    // ===== Step 1: Deinterleave BGR -> planar B, G, R using SIMD =====
    const uint8_t* srcData = image.data;
    int totalPixels = width * height;
    int i = 0;

    // Process 8 pixels at a time with SIMD (24 bytes = 8 pixels * 3 channels)
    for (; i + 7 < totalPixels; i += 8)
    {
        // Load 24 uint8 values (8 BGR pixels)
        // Layout: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5 G5 R5 B6 G6 R6 B7 G7 R7
        __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&srcData[i * 3]));       // bytes 0-15
        __m128i high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&srcData[i * 3 + 16])); // bytes 16-23

        // Use shuffle to extract B channel (positions 0, 3, 6, 9, 12, 15, 18, 21)
        // Shuffle mask for extracting every 3rd byte starting at 0
        const __m128i shuffle_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shuffle_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i b_vals = _mm_shuffle_epi8(low, shuffle_b);
        __m128i b_vals2 = _mm_shuffle_epi8(high, shuffle_b2);
        b_vals = _mm_or_si128(b_vals, b_vals2);

        // Use shuffle to extract G channel (positions 1, 4, 7, 10, 13, 16, 19, 22)
        const __m128i shuffle_g = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shuffle_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i g_vals = _mm_shuffle_epi8(low, shuffle_g);
        __m128i g_vals2 = _mm_shuffle_epi8(high, shuffle_g2);
        g_vals = _mm_or_si128(g_vals, g_vals2);

        // Use shuffle to extract R channel (positions 2, 5, 8, 11, 14, 17, 20, 23)
        const __m128i shuffle_r = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shuffle_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i r_vals = _mm_shuffle_epi8(low, shuffle_r);
        __m128i r_vals2 = _mm_shuffle_epi8(high, shuffle_r2);
        r_vals = _mm_or_si128(r_vals, r_vals2);

        // Convert uint8 to int32 then to float (8 values each)
        __m256i b_i32 = _mm256_cvtepu8_epi32(b_vals);
        __m256i g_i32 = _mm256_cvtepu8_epi32(g_vals);
        __m256i r_i32 = _mm256_cvtepu8_epi32(r_vals);

        __m256 b_f32 = _mm256_cvtepi32_ps(b_i32);
        __m256 g_f32 = _mm256_cvtepi32_ps(g_i32);
        __m256 r_f32 = _mm256_cvtepi32_ps(r_i32);

        // Store to planar arrays
        _mm256_storeu_ps(&planarB[i], b_f32);
        _mm256_storeu_ps(&planarG[i], g_f32);
        _mm256_storeu_ps(&planarR[i], r_f32);
    }

    // Handle remaining pixels with scalar code
    for (; i < totalPixels; ++i)
    {
        planarB[i] = static_cast<float>(srcData[i * 3]);
        planarG[i] = static_cast<float>(srcData[i * 3 + 1]);
        planarR[i] = static_cast<float>(srcData[i * 3 + 2]);
    }

    // ===== Step 2: Horizontal pass (each row independently) =====
    for (int y = 0; y < height; ++y)
    {
        int offset = y * width;
        convolve1D_AVX2(&planarB[offset], &tempB[offset], width, kernel.data(), kernelSize);
        convolve1D_AVX2(&planarG[offset], &tempG[offset], width, kernel.data(), kernelSize);
        convolve1D_AVX2(&planarR[offset], &tempR[offset], width, kernel.data(), kernelSize);
    }

    // ===== Step 3: Vertical pass using transpose + horizontal convolution =====
    // This is more efficient than processing columns directly
    // Strategy: transpose -> convolve rows (which were columns) -> transpose back

    // Allocate temporary buffers for transposed data
    std::vector<float> transposeB(width * height);
    std::vector<float> transposeG(width * height);
    std::vector<float> transposeR(width * height);

    // Transpose: (width x height) -> (height x width)
    // After transpose, columns become rows
    transpose(tempB.data(), transposeB.data(), width, height);
    transpose(tempG.data(), transposeG.data(), width, height);
    transpose(tempR.data(), transposeR.data(), width, height);

    // Apply horizontal convolution on transposed data
    // Each "row" in transposed data is actually a column from original
    for (int x = 0; x < width; ++x)  // x is now the row index in transposed space
    {
        int offset = x * height;
        convolve1D_AVX2(&transposeB[offset], &transposeB[offset], height, kernel.data(), kernelSize);
        convolve1D_AVX2(&transposeG[offset], &transposeG[offset], height, kernel.data(), kernelSize);
        convolve1D_AVX2(&transposeR[offset], &transposeR[offset], height, kernel.data(), kernelSize);
    }

    // Transpose back: (height x width) -> (width x height)
    transpose(transposeB.data(), planarB.data(), height, width);
    transpose(transposeG.data(), planarG.data(), height, width);
    transpose(transposeR.data(), planarR.data(), height, width);

    // ===== Step 4: Interleave planar B, G, R -> BGR and convert to uint8 =====
    uint8_t* dstData = image.data;
    for (int i = 0; i < width * height; ++i)
    {
        dstData[i * 3] = static_cast<uint8_t>(std::clamp(planarB[i], 0.0f, 255.0f));
        dstData[i * 3 + 1] = static_cast<uint8_t>(std::clamp(planarG[i], 0.0f, 255.0f));
        dstData[i * 3 + 2] = static_cast<uint8_t>(std::clamp(planarR[i], 0.0f, 255.0f));
    }
}

// AVX2-optimized normalize for RGB interleaved data
static void normalizeAVX2(float* data, int totalPixels,
                          const std::array<float, 3>& mean,
                          const std::array<float, 3>& std)
{
    // Precompute inverse std for multiplication instead of division
    const float inv_std[3] = {1.0f / std[0], 1.0f / std[1], 1.0f / std[2]};

    // Load mean and inv_std into AVX2 registers for 3 different offsets
    // First 8 floats: R0 G0 B0 R1 G1 B1 R2 G2
    __m256 mean_vec0 = _mm256_setr_ps(
        mean[0], mean[1], mean[2],
        mean[0], mean[1], mean[2],
        mean[0], mean[1]
    );
    __m256 inv_std_vec0 = _mm256_setr_ps(
        inv_std[0], inv_std[1], inv_std[2],
        inv_std[0], inv_std[1], inv_std[2],
        inv_std[0], inv_std[1]
    );

    // Next 8 floats: B2 R3 G3 B3 R4 G4 B4 R5
    __m256 mean_vec1 = _mm256_setr_ps(
        mean[2], mean[0], mean[1], mean[2],
        mean[0], mean[1], mean[2], mean[0]
    );
    __m256 inv_std_vec1 = _mm256_setr_ps(
        inv_std[2], inv_std[0], inv_std[1], inv_std[2],
        inv_std[0], inv_std[1], inv_std[2], inv_std[0]
    );

    // Last 8 floats: G5 B5 R6 G6 B6 R7 G7 B7
    __m256 mean_vec2 = _mm256_setr_ps(
        mean[1], mean[2], mean[0], mean[1],
        mean[2], mean[0], mean[1], mean[2]
    );
    __m256 inv_std_vec2 = _mm256_setr_ps(
        inv_std[1], inv_std[2], inv_std[0], inv_std[1],
        inv_std[2], inv_std[0], inv_std[1], inv_std[2]
    );

    int i = 0;

    // Process 24 floats (8 complete RGB pixels) at a time with 3 AVX2 operations
    const int simd_end = (totalPixels / 8) * 24;

    for (; i < simd_end; i += 24)
    {
        // Process first 8 floats
        __m256 pixels0 = _mm256_loadu_ps(&data[i]);
        pixels0 = _mm256_sub_ps(pixels0, mean_vec0);
        pixels0 = _mm256_mul_ps(pixels0, inv_std_vec0);
        _mm256_storeu_ps(&data[i], pixels0);

        // Process next 8 floats
        __m256 pixels1 = _mm256_loadu_ps(&data[i + 8]);
        pixels1 = _mm256_sub_ps(pixels1, mean_vec1);
        pixels1 = _mm256_mul_ps(pixels1, inv_std_vec1);
        _mm256_storeu_ps(&data[i + 8], pixels1);

        // Process last 8 floats
        __m256 pixels2 = _mm256_loadu_ps(&data[i + 16]);
        pixels2 = _mm256_sub_ps(pixels2, mean_vec2);
        pixels2 = _mm256_mul_ps(pixels2, inv_std_vec2);
        _mm256_storeu_ps(&data[i + 16], pixels2);
    }

    // Handle remaining pixels with scalar code
    for (; i < totalPixels * 3; i += 3)
    {
        data[i]     = (data[i]     - mean[0]) * inv_std[0];
        data[i + 1] = (data[i + 1] - mean[1]) * inv_std[1];
        data[i + 2] = (data[i + 2] - mean[2]) * inv_std[2];
    }
}
#endif

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            // ========================================
            // Normalize Implementation
            // ========================================

            Normalize::Normalize(const std::array<float, 3>& mean, const std::array<float, 3>& std)
                : mean_(mean)
                , std_(std)
            {
                // Validate std values (must be non-zero)
                for (size_t i = 0; i < 3; ++i)
                {
                    if (std_[i] <= 0.0f)
                    {
                        throw std::invalid_argument("Standard deviation values must be positive");
                    }
                }
            }

            Normalize Normalize::imageNet()
            {
                // ImageNet mean and std (RGB order)
                return Normalize(
                    {0.485f, 0.456f, 0.406f},  // mean
                    {0.229f, 0.224f, 0.225f}   // std
                );
            }

            void Normalize::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Ensure image is float type
                if (image.type() != CV_32FC3)
                {
                    throw std::invalid_argument("Normalize requires CV_32FC3 image (use ToTensor first)");
                }

                // OpenCV uses BGR order, but mean/std are typically in RGB order
                // We need to apply: (img - mean) / std per channel
                // Assuming image is in RGB order after ToTensor

                // Direct pixel access for in-place normalization
                float* data = reinterpret_cast<float*>(image.data);
                int totalPixels = image.rows * image.cols;

#if defined(_MSC_VER) || defined(__AVX2__)
                // Use AVX2 if available
                if (hasAVX2())
                {
                    normalizeAVX2(data, totalPixels, mean_, std_);
                    return;
                }
#endif

                // Scalar fallback - precompute inverse std for multiplication
                const float inv_std[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};

                // Process all pixels
                for (int i = 0; i < totalPixels; ++i)
                {
                    int idx = i * 3;
                    data[idx]     = (data[idx]     - mean_[0]) * inv_std[0];  // Channel 0
                    data[idx + 1] = (data[idx + 1] - mean_[1]) * inv_std[1];  // Channel 1
                    data[idx + 2] = (data[idx + 2] - mean_[2]) * inv_std[2];  // Channel 2
                }
            }

            std::unique_ptr<Transform> Normalize::clone() const
            {
                return std::make_unique<Normalize>(mean_, std_);
            }

            // ========================================
            // ToTensor Implementation
            // ========================================

            ToTensor::ToTensor(bool bgrToRgb)
                : bgrToRgb_(bgrToRgb)
            {
            }

            void ToTensor::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Ensure input is uint8
                if (image.type() != CV_8UC3)
                {
                    throw std::invalid_argument("ToTensor requires CV_8UC3 image");
                }

                int totalPixels = image.rows * image.cols;

#if defined(_MSC_VER) || defined(__AVX2__)
                // Use AVX2 optimized path if available
                if (hasAVX2())
                {
                    // Create output Mat for float32
                    cv::Mat floatImage(image.rows, image.cols, CV_32FC3);

                    // Convert uint8 to float with optional BGR->RGB using AVX2
                    toTensorAVX2(image.data, reinterpret_cast<float*>(floatImage.data), totalPixels, bgrToRgb_);

                    // Replace input image with converted one (use move to avoid reference counting overhead)
                    image = std::move(floatImage);
                    return;
                }
#endif

                // Fallback: use OpenCV functions
                // Convert to float [0, 1] in-place
                image.convertTo(image, CV_32FC3, 1.0 / 255.0);

                // Convert BGR to RGB if requested (in-place)
                if (bgrToRgb_)
                {
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                }

                // Note: We keep the image in HWC format for OpenCV compatibility
                // The actual CHW conversion would be done by the framework (PyTorch/TensorFlow)
                // when copying to tensor memory
            }

            std::unique_ptr<Transform> ToTensor::clone() const
            {
                return std::make_unique<ToTensor>(bgrToRgb_);
            }

            // ========================================
            // ColorJitter Implementation
            // ========================================

            ColorJitter::ColorJitter(float brightness, float contrast, float saturation, float hue)
                : brightness_(brightness)
                , contrast_(contrast)
                , saturation_(saturation)
                , hue_(hue)
            {
                if (brightness < 0.0f || contrast < 0.0f || saturation < 0.0f)
                {
                    throw std::invalid_argument("Brightness, contrast, and saturation must be non-negative");
                }
            }

            void ColorJitter::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Ensure image is uint8
                if (image.type() != CV_8UC3)
                {
                    return;  // Skip if not standard format
                }

                // Apply brightness and contrast together (single convertTo call)
                if (brightness_ > 0.0f || contrast_ > 0.0f)
                {
                    float brightnessFactor = brightness_ > 0.0f ?
                        rng_.uniformFloat(1.0f - brightness_, 1.0f + brightness_) : 1.0f;
                    float contrastFactor = contrast_ > 0.0f ?
                        rng_.uniformFloat(1.0f - contrast_, 1.0f + contrast_) : 1.0f;

                    // Combined factor: brightness * contrast (in-place)
                    float combinedFactor = brightnessFactor * contrastFactor;
                    image.convertTo(image, -1, combinedFactor, 0);
                }

                // Apply saturation and hue adjustments using OpenCV cvtColor
                if (saturation_ > 0.0f || hue_ > 0.0f)
                {
                    float saturationFactor = saturation_ > 0.0f ?
                        rng_.uniformFloat(1.0f - saturation_, 1.0f + saturation_) : 1.0f;
                    float hueShift = hue_ > 0.0f ?
                        rng_.uniformFloat(-hue_, hue_) : 0.0f;

                    // Convert BGR to HSV using OpenCV (more accurate and avoids floating-point errors)
                    cv::Mat hsvImage;
                    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

                    // OpenCV HSV ranges: H=[0,180), S=[0,255], V=[0,255]
                    // Note: OpenCV uses H/2 to fit in uint8 (0-360° → 0-180)

                    // Adjust HSV values
                    uint8_t* data = hsvImage.data;
                    int totalPixels = hsvImage.rows * hsvImage.cols;

                    for (int i = 0; i < totalPixels; ++i)
                    {
                        int idx = i * 3;

                        // Apply hue shift (H channel is in [0, 180) range)
                        if (hue_ > 0.0f)
                        {
                            // Convert hueShift from degrees to OpenCV range [0, 180)
                            float hueShiftScaled = hueShift / 2.0f;
                            int h = static_cast<int>(data[idx] + hueShiftScaled);
                            // Wrap around [0, 180)
                            if (h < 0) h += 180;
                            if (h >= 180) h -= 180;
                            data[idx] = static_cast<uint8_t>(h);
                        }

                        // Apply saturation adjustment (S channel is in [0, 255] range)
                        if (saturation_ > 0.0f)
                        {
                            float s = data[idx + 1] * saturationFactor;
                            data[idx + 1] = static_cast<uint8_t>(std::clamp(s, 0.0f, 255.0f));
                        }
                    }

                    // Convert HSV back to BGR using OpenCV
                    cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
                }
            }

            std::unique_ptr<Transform> ColorJitter::clone() const
            {
                return std::make_unique<ColorJitter>(brightness_, contrast_, saturation_, hue_);
            }

            // ========================================
            // GaussianBlur Implementation
            // ========================================

            int GaussianBlur::makeOddKernelSize(int size)
            {
                if (size <= 0)
                {
                    return 1;
                }
                // Make odd by adding 1 if even
                return (size % 2 == 0) ? size + 1 : size;
            }

            GaussianBlur::GaussianBlur(int minKernelSize, int maxKernelSize, float probability)
                : minKernelSize_(makeOddKernelSize(minKernelSize))
                , maxKernelSize_(makeOddKernelSize(maxKernelSize))
                , probability_(probability)
            {
                if (minKernelSize_ <= 0 || maxKernelSize_ <= 0)
                {
                    throw std::invalid_argument("Kernel sizes must be positive");
                }

                if (minKernelSize_ > maxKernelSize_)
                {
                    throw std::invalid_argument("Min kernel size must be <= max kernel size");
                }

                if (probability < 0.0f || probability > 1.0f)
                {
                    throw std::invalid_argument("Probability must be between 0.0 and 1.0");
                }
            }

            void GaussianBlur::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Check probability - skip blur if probability check fails
                // For probability = 1.0, always apply; for probability = 0.0, never apply
                if (probability_ < 1.0f && !rng_.bernoulli(probability_))
                {
                    return;  // Don't apply blur
                }

                // Sample random kernel size (must be odd)
                int kernelSize;
                if (minKernelSize_ == maxKernelSize_)
                {
                    kernelSize = minKernelSize_;
                }
                else
                {
                    // Sample from range [min, max] with step 2 (to ensure odd)
                    int numSteps = (maxKernelSize_ - minKernelSize_) / 2 + 1;
                    int step = rng_.uniformInt(0, numSteps - 1);
                    kernelSize = minKernelSize_ + step * 2;
                }

#if defined(_MSC_VER) || defined(__AVX2__)
                // Use optimized separable Gaussian blur if available
                if (hasAVX2() && image.type() == CV_8UC3)
                {
                    try
                    {
                        gaussianBlurSeparableAVX2(image, kernelSize);
                        return;
                    }
                    catch (const std::bad_alloc&)
                    {
                        // Memory allocation failed for large image, fall back to OpenCV
                        // (Continues to OpenCV fallback below)
                    }
                    catch (...)
                    {
                        // Other errors, fall back to OpenCV
                    }
                }
#endif

                // Fallback: use OpenCV's GaussianBlur
                cv::GaussianBlur(image, image, cv::Size(kernelSize, kernelSize), 0);
            }

            std::unique_ptr<Transform> GaussianBlur::clone() const
            {
                return std::make_unique<GaussianBlur>(minKernelSize_, maxKernelSize_, probability_);
            }

            // ========================================
            // GaussianNoise
            // ========================================

            GaussianNoise::GaussianNoise(float mean, float stddev)
                : mean_(mean)
                , stddev_(stddev)
            {
                if (stddev < 0.0f)
                {
                    throw std::invalid_argument("Standard deviation must be non-negative");
                }
            }

            void GaussianNoise::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Check probability
                if (probability_ < rng_.uniformFloat(0, 1))
                {
                    return;
                }

                // Generate Gaussian noise
                cv::Mat noise(image.size(), image.type());
                cv::randn(noise, mean_, stddev_);

                // Add noise to image
                cv::Mat temp;
                image.convertTo(temp, CV_32F);
                noise.convertTo(noise, CV_32F);
                temp += noise;

                // Clamp to [0, 255] and convert back
                temp = cv::max(temp, 0.0f);
                temp = cv::min(temp, 255.0f);
                temp.convertTo(image, image.type());
            }

            std::unique_ptr<Transform> GaussianNoise::clone() const
            {
                return std::make_unique<GaussianNoise>(mean_, stddev_);
            }

            // ========================================
            // CLAHE
            // ========================================

            CLAHE::CLAHE(float clipLimit, int tileGridSize)
                : clipLimit_(clipLimit)
                , tileGridSize_(tileGridSize)
            {
                if (clipLimit <= 0.0f)
                {
                    throw std::invalid_argument("Clip limit must be positive");
                }
                if (tileGridSize <= 0)
                {
                    throw std::invalid_argument("Tile grid size must be positive");
                }
            }

            void CLAHE::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                if (probability_ < rng_.uniformFloat(0, 1))
                {
                    return;
				}

                if (image.channels() == 1)
                {
                    // Grayscale image
                    clahe_->apply(image, image);
                }
                else if (image.channels() == 3)
                {
                    // Color image - apply to luminance channel only
                    cv::Mat lab;
                    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

                    std::vector<cv::Mat> labChannels;
                    cv::split(lab, labChannels);

                    // Apply CLAHE to L channel
                    clahe_->apply(labChannels[0], labChannels[0]);

                    cv::merge(labChannels, lab);
                    cv::cvtColor(lab, image, cv::COLOR_Lab2BGR);
                }
            }

            std::unique_ptr<Transform> CLAHE::clone() const
            {
                return std::make_unique<CLAHE>(clipLimit_, tileGridSize_);
            }

            // ========================================
            // RandomGamma
            // ========================================

            RandomGamma::RandomGamma(float gammaRange)
                : gammaRange_(gammaRange)
            {
                if (gammaRange < 0.0f || gammaRange >= 1.0f)
                {
                    throw std::invalid_argument("Gamma range must be in [0.0, 1.0)");
                }
            }

            void RandomGamma::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Check probability
                if (probability_ < rng_.uniformFloat(0, 1))
                {
                    return;
				}
                // Sample gamma value from [1-range, 1+range]
                float gamma = rng_.uniformFloat(1.0f - gammaRange_, 1.0f + gammaRange_);

                // Build lookup table
                std::vector<uchar> lut(256);
                for (int i = 0; i < 256; ++i)
                {
                    float normalized = i / 255.0f;
                    float corrected = std::pow(normalized, gamma);
                    lut[i] = static_cast<uchar>(cv::saturate_cast<uchar>(corrected * 255.0f));
                }

                // Apply gamma correction
                cv::Mat lookupTable(1, 256, CV_8U, lut.data());
                cv::LUT(image, lookupTable, image);
            }

            std::unique_ptr<Transform> RandomGamma::clone() const
            {
                return std::make_unique<RandomGamma>(gammaRange_);
            }

            // ========================================
            // ToGray
            // ========================================

            ToGray::ToGray(bool keepChannels)
                : keepChannels_(keepChannels)
            {
            }

            void ToGray::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                if (probability_ < rng_.uniformFloat(0, 1))
                {
                    return;
                }

                if (image.channels() == 1)
                {
                    // Already grayscale
                    if (keepChannels_)
                    {
                        // Convert to 3-channel
                        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                    }
                    return;
                }

                if (image.channels() == 3)
                {
                    cv::Mat gray;
                    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

                    if (keepChannels_)
                    {
                        // Convert back to 3-channel (same value in all channels)
                        cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
                    }
                    else
                    {
                        image = gray;
                    }
                }
            }

            std::unique_ptr<Transform> ToGray::clone() const
            {
                return std::make_unique<ToGray>(keepChannels_);
            }

            // ========================================
            // SaltAndPepper
            // ========================================

            SaltAndPepper::SaltAndPepper(float saltProb, float pepperProb)
                : saltProb_(saltProb)
                , pepperProb_(pepperProb)
            {
                if (saltProb < 0.0f || saltProb > 1.0f)
                {
                    throw std::invalid_argument("Salt probability must be between 0.0 and 1.0");
                }
                if (pepperProb < 0.0f || pepperProb > 1.0f)
                {
                    throw std::invalid_argument("Pepper probability must be between 0.0 and 1.0");
                }
            }

            void SaltAndPepper::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Check probability
                if (probability_ < rng_.uniformFloat(0, 1))
                {
                    return;
				}

                // Apply salt and pepper noise
                for (int y = 0; y < image.rows; ++y)
                {
                    for (int x = 0; x < image.cols; ++x)
                    {
                        float rand = rng_.uniformFloat(0.0f, 1.0f);

                        if (rand < saltProb_)
                        {
                            // Salt (white)
                            if (image.channels() == 1)
                            {
                                image.at<uchar>(y, x) = 255;
                            }
                            else if (image.channels() == 3)
                            {
                                image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                            }
                        }
                        else if (rand < saltProb_ + pepperProb_)
                        {
                            // Pepper (black)
                            if (image.channels() == 1)
                            {
                                image.at<uchar>(y, x) = 0;
                            }
                            else if (image.channels() == 3)
                            {
                                image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                            }
                        }
                    }
                }
            }

            std::unique_ptr<Transform> SaltAndPepper::clone() const
            {
                return std::make_unique<SaltAndPepper>(saltProb_, pepperProb_);
            }

            // ========================================
            // MedianBlur
            // ========================================

            int MedianBlur::makeOddKernelSize(int size)
            {
                if (size <= 0)
                {
                    return 1;
                }
                return (size % 2 == 0) ? size + 1 : size;
            }

            MedianBlur::MedianBlur(int minKernelSize, int maxKernelSize, float probability)
                : minKernelSize_(makeOddKernelSize(minKernelSize))
                , maxKernelSize_(makeOddKernelSize(maxKernelSize))
                , probability_(probability)
            {
                if (minKernelSize_ <= 0 || maxKernelSize_ <= 0)
                {
                    throw std::invalid_argument("Kernel sizes must be positive");
                }
                if (minKernelSize_ > maxKernelSize_)
                {
                    throw std::invalid_argument("Min kernel size must be <= max kernel size");
                }
                if (probability < 0.0f || probability > 1.0f)
                {
                    throw std::invalid_argument("Probability must be between 0.0 and 1.0");
                }
            }

            void MedianBlur::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                {
                    return;
                }

                // Check probability
                if (probability_ < 1.0f && !rng_.bernoulli(probability_))
                {
                    return;
                }

                // Sample random kernel size (must be odd)
                int kernelSize;
                if (minKernelSize_ == maxKernelSize_)
                {
                    kernelSize = minKernelSize_;
                }
                else
                {
                    int numSteps = (maxKernelSize_ - minKernelSize_) / 2 + 1;
                    int step = rng_.uniformInt(0, numSteps - 1);
                    kernelSize = minKernelSize_ + step * 2;
                }

                // Apply median blur
                cv::medianBlur(image, image, kernelSize);
            }

            std::unique_ptr<Transform> MedianBlur::clone() const
            {
                return std::make_unique<MedianBlur>(minKernelSize_, maxKernelSize_, probability_);
            }

        } // namespace Transforms
    } // namespace Data
} // namespace WheelDL
