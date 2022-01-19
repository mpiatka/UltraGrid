#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template<typename T>
class concurrent_queue {
        std::deque<T> deque{};
        mutable std::mutex mutex{};
        std::atomic<bool> try_pop_skip = false; // try pop doesn't acquires scope_lock if queue is empty
        std::condition_variable queue_non_empty{};
public:
        concurrent_queue() = default;

        std::pair<std::unique_lock<std::mutex>, std::deque<T>&> get_underlying_deque() {
                return { std::unique_lock{mutex}, deque };
        }

        std::condition_variable& get_queue_non_empty_condition_var() {
                return queue_non_empty;
        }

        bool empty() const {
                std::scoped_lock lock{ mutex };
                return deque.empty();
        }

        auto size() const {
                std::scoped_lock lock{ mutex };
                auto size = deque.size();
                return size;
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
                try_pop_skip = false;
                queue_non_empty.notify_one();
        }

        template<typename... Args>
        void emplace_front(Args&&... args) {
                {
                        std::scoped_lock lock{ mutex };
                        deque.emplace_front(std::forward<Args>(args)...);
                }
                try_pop_skip = false;
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
                if (try_pop_skip){
                        return std::nullopt;
                }
                std::scoped_lock lock{ mutex };
                if (deque.empty()) {
                        try_pop_skip = true;
                        return std::nullopt;
                }
                T result = std::move(deque.front());
                deque.pop_front();
                return result;
        }
};
