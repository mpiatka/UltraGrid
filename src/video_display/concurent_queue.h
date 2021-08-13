#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template<typename T>
class concurrent_queue {
        std::deque<T> deque{};
        mutable std::mutex mutex{};
public:
        std::condition_variable queue_non_empty{};

        concurrent_queue() = default;

        std::pair<std::unique_lock<std::mutex>, std::deque<T>&> get_underlying_deque() {
                return { std::unique_lock{mutex}, deque };
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
                emplace_back(value);
        }

        void push(T&& value) {
                emplace_back(std::move(value));
        }

        template<typename... Args>
        void emplace_back(Args&&... args) {
                {
                        std::scoped_lock lock{ mutex };
                        deque.emplace_back(std::forward<Args>(args)...);
                }
                queue_non_empty.notify_one();
        }

        template<typename... Args>
        void emplace_front(Args&&... args) {
                {
                        std::scoped_lock lock{ mutex };
                        deque.emplace_front(std::forward<Args>(args)...);
                }
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
