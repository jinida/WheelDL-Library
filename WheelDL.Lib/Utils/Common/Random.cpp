#include "pch.h"
#include "Random.h"
#include <algorithm>
#include <stdexcept>

namespace WheelDL {
	namespace Utils {

		Random::Random(unsigned int seed)
			: _generator(seed) {
		}

		void Random::setSeed(unsigned int seed) {
			std::lock_guard<std::mutex> lock(_mutex);
			_generator.seed(seed);
		}

		int Random::uniformInt(int min, int max) {
			if (min > max) {
				throw std::invalid_argument("Random::uniformInt: min must be <= max");
			}
			std::lock_guard<std::mutex> lock(_mutex);
			std::uniform_int_distribution<int> dist(min, max);
			return dist(_generator);
		}

		float Random::uniformFloat(float min, float max) {
			if (min > max) {
				throw std::invalid_argument("Random::uniformFloat: min must be <= max");
			}
			std::lock_guard<std::mutex> lock(_mutex);
			std::uniform_real_distribution<float> dist(min, max);
			return dist(_generator);
		}

		double Random::uniformDouble(double min, double max) {
			if (min > max) {
				throw std::invalid_argument("Random::uniformDouble: min must be <= max");
			}
			std::lock_guard<std::mutex> lock(_mutex);
			std::uniform_real_distribution<double> dist(min, max);
			return dist(_generator);
		}

		float Random::normal(float mean, float stddev) {
			if (stddev < 0.0f) {
				throw std::invalid_argument("Random::normal: stddev must be non-negative");
			}
			std::lock_guard<std::mutex> lock(_mutex);
			std::normal_distribution<float> dist(mean, stddev);
			return dist(_generator);
		}

		bool Random::bernoulli(double probability) {
			if (probability < 0.0 || probability > 1.0) {
				throw std::invalid_argument("Random::bernoulli: probability must be in [0, 1]");
			}
			std::lock_guard<std::mutex> lock(_mutex);
			std::bernoulli_distribution dist(probability);
			return dist(_generator);
		}

	} // namespace Utils
} // namespace WheelDL
