#pragma once

#include <random>
#include <mutex>

namespace WheelDL {
	namespace Utils {

		/**
		 * @class Random
		 * @brief Thread-safe random number generator utility
		 *
		 * Provides various random number generation methods using Mersenne Twister engine.
		 * All methods are thread-safe and can be called concurrently from multiple threads.
		 *
		 * Usage:
		 * @code
		 * Random rng(12345);
		 * int randomInt = rng.uniformInt(0, 100);
		 * float randomFloat = rng.uniformFloat(0.0f, 1.0f);
		 * float normalValue = rng.normal(0.0f, 1.0f);
		 * @endcode
		 */
		class Random {
		public:
			/**
			 * @brief Constructor with seed
			 * @param seed Random seed (default: random_device)
			 */
			explicit Random(unsigned int seed = std::random_device{}());

			/**
			 * @brief Set random seed
			 * @param seed Random seed
			 */
			void setSeed(unsigned int seed);

			/**
			 * @brief Generate random integer in [min, max] (inclusive)
			 * @param min Minimum value (inclusive)
			 * @param max Maximum value (inclusive)
			 * @return int Random integer
			 * @throws std::invalid_argument if min > max
			 */
			int uniformInt(int min, int max);

			/**
			 * @brief Generate random float in [min, max)
			 * @param min Minimum value (inclusive)
			 * @param max Maximum value (exclusive)
			 * @return float Random float
			 * @throws std::invalid_argument if min > max
			 */
			float uniformFloat(float min, float max);

			/**
			 * @brief Generate random double in [min, max)
			 * @param min Minimum value (inclusive)
			 * @param max Maximum value (exclusive)
			 * @return double Random double
			 * @throws std::invalid_argument if min > max
			 */
			double uniformDouble(double min, double max);

			/**
			 * @brief Generate random value from normal distribution
			 * @param mean Mean of the distribution
			 * @param stddev Standard deviation of the distribution
			 * @return float Random value from normal distribution
			 * @throws std::invalid_argument if stddev < 0
			 */
			float normal(float mean, float stddev);

			/**
			 * @brief Generate random boolean with given probability
			 * @param probability Probability of returning true (default: 0.5)
			 * @return bool Random boolean
			 * @throws std::invalid_argument if probability not in [0, 1]
			 */
			bool bernoulli(double probability = 0.5);

			/**
			 * @brief Shuffle a vector
			 * @tparam T Vector element type
			 * @param vec Vector to shuffle (modified in place)
			 */
			template<typename T>
			void shuffle(std::vector<T>& vec) {
				std::lock_guard<std::mutex> lock(_mutex);
				std::shuffle(vec.begin(), vec.end(), _generator);
			}

			/**
			 * @brief Get the internal generator (for advanced use)
			 * @return std::mt19937& Reference to the Mersenne Twister generator
			 * @warning Not thread-safe - caller must lock the mutex using getMutex()
			 *
			 * Example usage:
			 * @code
			 * Random rng;
			 * {
			 *     std::lock_guard<std::mutex> lock(rng.getMutex());
			 *     auto& gen = rng.getGenerator();
			 *     // Use gen safely within the locked scope
			 *     std::uniform_int_distribution<int> dist(0, 100);
			 *     int value = dist(gen);
			 * }
			 * @endcode
			 */
			std::mt19937& getGenerator() {
				return _generator;
			}

			/**
			 * @brief Get the mutex for manual synchronization
			 * @return std::mutex& Reference to the internal mutex
			 *
			 * Use this method to manually lock the mutex when using getGenerator()
			 * for thread-safe access to the random number generator.
			 */
			std::mutex& getMutex() {
				return _mutex;
			}

		private:
			std::mt19937 _generator;          ///< Mersenne Twister random number generator
			mutable std::mutex _mutex;        ///< Mutex for thread-safe access
		};

	} // namespace Utils
} // namespace WheelDL
