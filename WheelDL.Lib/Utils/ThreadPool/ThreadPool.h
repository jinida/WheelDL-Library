#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <string>

namespace WheelDL {
	namespace Utils {

		/**
		 * @class ThreadPool
		 * @brief Object Pool pattern based thread pool
		 *
		 * Manages a pool of worker threads that execute queued tasks.
		 * - Configurable number of worker threads
		 * - Task queue with FIFO ordering
		 * - Future-based result retrieval
		 * - Automatic cleanup on destruction
		 *
		 * @note Thread-safe implementation
		 */
		class ThreadPool {
		public:
			/**
			 * @brief Construct thread pool with specified number of threads
			 * @param threads Number of worker threads to create
			 */
			explicit ThreadPool(size_t threads);

			/**
			 * @brief Destructor - waits for all tasks to complete
			 */
			~ThreadPool();

			// Disable copy and move
			ThreadPool(const ThreadPool&) = delete;
			ThreadPool& operator=(const ThreadPool&) = delete;
			ThreadPool(ThreadPool&&) = delete;
			ThreadPool& operator=(ThreadPool&&) = delete;

			/**
			 * @brief Enqueue a task for execution
			 * @tparam F Callable type
			 * @tparam Args Argument types
			 * @param f Callable object (function, lambda, functor)
			 * @param args Arguments to pass to the callable
			 * @return std::future<return_type> Future for the result
			 * @throws std::runtime_error if enqueue on stopped pool
			 */
			template<class F, class... Args>
			auto enqueue(F&& f, Args&&... args)
				-> std::future<typename std::invoke_result<F, Args...>::type>;

			/**
			 * @brief Get number of worker threads
			 * @return size_t Number of threads in the pool
			 */
			size_t getNumThreads() const { return _workers.size(); }

			/**
			 * @brief Set maximum task queue size (prevents unbounded growth)
			 * @param maxSize Maximum number of queued tasks (0 = unlimited)
			 *
			 * When the queue is full, enqueue() will throw std::runtime_error.
			 * This helps prevent excessive memory usage in high-load scenarios.
			 */
			void setMaxQueueSize(size_t maxSize) {
				std::lock_guard<std::mutex> lock(_queueMutex);
				_maxQueueSize = maxSize;
			}

			/**
			 * @brief Get number of pending tasks
			 * @return size_t Number of tasks waiting in queue
			 */
			size_t getPendingTaskCount() const;

		private:
			// Worker thread function (extracted from lambda for better readability)
			void workerThreadFunction();

			// Worker threads
			std::vector<std::thread> _workers;

			// Task queue
			std::queue<std::function<void()>> _tasks;

			// Synchronization
			mutable std::mutex _queueMutex;
			std::condition_variable _condition;

			// Stop flag
			std::atomic<bool> _stop;

			// Maximum queue size (0 = unlimited)
			size_t _maxQueueSize;
		};

		// Template implementation

		template<class F, class... Args>
		auto ThreadPool::enqueue(F&& f, Args&&... args)
			-> std::future<typename std::invoke_result<F, Args...>::type>
		{
			using return_type = typename std::invoke_result<F, Args...>::type;

			// Create packaged task
			auto task = std::make_shared<std::packaged_task<return_type()>>(
				std::bind(std::forward<F>(f), std::forward<Args>(args)...)
			);

			std::future<return_type> res = task->get_future();

			{
				std::unique_lock<std::mutex> lock(_queueMutex);

				// Don't allow enqueueing after stopping the pool
				if (_stop.load(std::memory_order_acquire)) {
					throw std::runtime_error("enqueue on stopped ThreadPool");
				}

				// Check queue size limit
				if (_maxQueueSize > 0 && _tasks.size() >= _maxQueueSize) {
					throw std::runtime_error("ThreadPool queue is full (max size: " + std::to_string(_maxQueueSize) + ")");
				}

				_tasks.emplace([task]() { (*task)(); });
			}

			_condition.notify_one();
			return res;
		}

	} // namespace Utils
} // namespace WheelDL
