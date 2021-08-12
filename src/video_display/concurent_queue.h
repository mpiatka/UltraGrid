#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template<typename T>
class concurrent_queue {
        std::condition_variable queue_non_empty{};
        mutable std::mutex mutex{};
        std::deque<T> deque{};
public:
        concurrent_queue() = default;

        std::mutex& get_mutex() {
                return mutex;
        }

        std::deque<T>& get_underlying_unsynchronized_deque() {
                return deque;
        }

        bool empty() const {
                std::scoped_lock lock{ mutex };
                return deque.empty();
        }

        auto size() const {
                std::scoped_lock lock{ mutex };
                return deque.size();
        }

        void push(const T& value) {
                emplace(value);
        }

        void push(T&& value) {
                emplace(std::move(value));
        }

        template<typename... Args>
        void emplace(Args&&... args) {
                std::unique_lock lock{ mutex };
                deque.emplace_back(std::forward<Args>(args)...);
                lock.unlock();
                queue_non_empty.notify_one();
        }

        T pop() {
                std::unique_lock lock{ mutex };
                queue_non_empty.wait(lock, [&d = this->deque]() { return !d.empty(); });
                T result = std::move(deque.front());
                deque.pop_front();
                return result;
        }
        
        std::optional<T> try_pop() {
                std::scoped_lock lock{ mutex };
                if (deque.empty()) {
                        return {};
                }
                T result = std::move(deque.front());
                deque.pop_front();
                return result;
        }
};
