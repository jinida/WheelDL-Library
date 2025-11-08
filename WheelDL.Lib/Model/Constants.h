#pragma once

/**
 * @file Constants.h
 * @brief Unified constants for WheelDL Model components
 *
 * This file centralizes all numerical constants used throughout the model
 * components to ensure consistency and maintainability.
 */

namespace WheelDL {
    namespace Model {
        namespace Constants {

            // ============================================================================
            // Mathematical Constants
            // ============================================================================
            inline constexpr int INPUT_CHANNELS = 3;  ///< Default input channels for RGB images
            /// Pi constant for angle calculations
            inline constexpr float PI = 3.14159265358979323846f;

            // ============================================================================
            // Numerical Stability Epsilons
            // ============================================================================

            /// General purpose epsilon for numerical stability
            inline constexpr float EPSILON = 1e-7f;

            /// Larger epsilon for less sensitive operations
            inline constexpr float EPSILON_LARGE = 1e-5f;

            /// Smaller epsilon for very sensitive operations
            inline constexpr float EPSILON_SMALL = 1e-9f;

            // ============================================================================
            // SSIM Constants (Wang et al. IEEE TIP 2004)
            // ============================================================================

            /// SSIM K1 parameter (luminance stability constant)
            inline constexpr float SSIM_K1 = 0.01f;

            /// SSIM K2 parameter (contrast stability constant)
            inline constexpr float SSIM_K2 = 0.03f;

            /// SSIM C1 = (K1 * L)^2 where L is dynamic range (assuming L=1 for normalized images)
            inline constexpr float SSIM_C1 = SSIM_K1 * SSIM_K1;

            /// SSIM C2 = (K2 * L)^2 where L is dynamic range (assuming L=1 for normalized images)
            inline constexpr float SSIM_C2 = SSIM_K2 * SSIM_K2;

            // ============================================================================
            // Probability Clamping for Numerical Stability
            // ============================================================================

            /// Minimum probability value to avoid log(0)
            inline constexpr float PROB_MIN = EPSILON;

            /// Maximum probability value to avoid log(0)
            inline constexpr float PROB_MAX = 1.0f - EPSILON;

            /// Epsilon for probability clamping in focal loss
            inline constexpr float PROB_EPSILON = 1e-7f;

            // ============================================================================
            // IoU Computation Constants
            // ============================================================================

            /// Epsilon for IoU calculations to prevent division by zero
            inline constexpr float IOU_EPS = 1e-6f;

            /// Minimum height threshold for rotated bounding boxes
            inline constexpr float HEIGHT_MIN_THRESHOLD = 1e-5f;

            /// Minimum denominator value for probiou calculation
            inline constexpr float PROBIOU_DENOM_MIN = IOU_EPS * 10.0f;

            // ============================================================================
            // Focal Loss Constants
            // ============================================================================

            /// Maximum gamma value for focal loss to prevent numerical overflow
            inline constexpr float FOCAL_GAMMA_MAX = 5.0f;

            // ============================================================================
            // DFL (Distribution Focal Loss) Constants
            // ============================================================================

            /// Clamping epsilon for DFL target values
            inline constexpr float DFL_CLAMP_EPSILON = 0.01f;

            // ============================================================================
            // Task Assignment Constants
            // ============================================================================

            /// Epsilon for task-aligned assignment calculations
            inline constexpr float ASSIGNMENT_EPS = 1e-9f;

            // ============================================================================
            // Segmentation Loss Constants
            // ============================================================================

            /// Epsilon for dice coefficient calculation in segmentation loss
            inline constexpr float DICE_EPSILON = 1e-5f;

        }  // namespace Constants
    }  // namespace Model
}  // namespace WheelDL
