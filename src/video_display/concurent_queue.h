#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template<typename T>
class concurrent_queue {
        std::condition_variable queue_non_empty;
        mutable std::mutex mutex;
        std::queue<T> queue;
public:
        concurrent_queue() = default;

        std::queue<T>& get_underlying_unsynchronized_queue() {
                return queue;
        }

        void push(const T& value) {
                emplace(value);
        }

        void push(T&& value) {
                emplace(std::move(value));
        }

        bool empty() const {
                std::scoped_lock lock{ mutex };
                return queue.empty();
        }

        auto size() const {
                std::scoped_lock lock{ mutex };
                return queue.size();
        }

        template<typename... Args>
        void emplace(Args&&... args) {
                std::unique_lock lock{ mutex };
                queue.emplace(std::forward<Args>(args)...);
                lock.unlock();
                queue_non_empty.notify_one();
        }

        T pop() {
                std::unique_lock lock{ mutex };
                queue_non_empty.wait(lock, [&q = this->queue]() { return !q.empty(); });
                T result = std::move(queue.front());
                queue.pop();
                return result;
        }
        
        std::optional<T> try_pop() {
                std::scoped_lock lock{ mutex };
                if (queue.empty()) {
                        return {};
                }
                T result = std::move(queue.front());
                queue.pop();
                return result;
        }
};
