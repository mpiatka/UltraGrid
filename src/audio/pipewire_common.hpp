#ifndef PIPEWIRE_COMMON_HPP_b6f55dc83d84
#define PIPEWIRE_COMMON_HPP_b6f55dc83d84

#include <memory>
#include <spa/param/audio/format-utils.h>
#include <pipewire/pipewire.h>

template<typename T, auto delete_func>
struct raii_deleter_helper { void operator()(T *p) { delete_func(p); } };
template<typename T, auto delete_func>
using raii_uniq_handle = std::unique_ptr<T, raii_deleter_helper<T, delete_func>>;

template<typename T>
struct raii_proxy_deleter_helper { void operator()(T *p) { pw_proxy_destroy(reinterpret_cast<pw_proxy *>(p)); } };
template<typename T>
using raii_uniq_proxy_handle = std::unique_ptr<T, raii_proxy_deleter_helper<T>>;

using pw_thread_loop_uniq = raii_uniq_handle<pw_thread_loop, pw_thread_loop_destroy>;
using pw_main_loop_uniq = raii_uniq_handle<pw_main_loop, pw_main_loop_destroy>;
using pw_stream_uniq = raii_uniq_handle<pw_stream, pw_stream_destroy>;
using pw_context_uniq = raii_uniq_handle<pw_context, pw_context_destroy>;
using pw_core_uniq = raii_uniq_handle<pw_core, pw_core_disconnect>;
using pw_registry_uniq = raii_uniq_proxy_handle<pw_registry>;

struct spa_hook_uniq{
    spa_hook_uniq(){
        spa_zero(hook);
    }
    ~spa_hook_uniq(){
        spa_hook_remove(&hook);
    }
    spa_hook_uniq(spa_hook_uniq&) = delete;
    spa_hook_uniq& operator=(spa_hook_uniq&) = delete;

    spa_hook& get() { return hook; }

    spa_hook hook;
};

class pipewire_thread_loop_lock_guard{
public:
    pipewire_thread_loop_lock_guard(pw_thread_loop *loop) : l(loop) {
        pw_thread_loop_lock(l);
    }
    ~pipewire_thread_loop_lock_guard(){
        pw_thread_loop_unlock(l);
    }
    pipewire_thread_loop_lock_guard(pipewire_thread_loop_lock_guard&) = delete;
    pipewire_thread_loop_lock_guard& operator=(pipewire_thread_loop_lock_guard&) = delete;

private:
    pw_thread_loop *l;
};

class pipewire_init_guard{
public:
    pipewire_init_guard() = default;
    pipewire_init_guard(pipewire_init_guard&) = delete;
    pipewire_init_guard& operator=(pipewire_init_guard&) = delete;
    pipewire_init_guard(pipewire_init_guard&& o) { std::swap(initialized, o.initialized);}
    pipewire_init_guard& operator=(pipewire_init_guard&& o) {
            std::swap(initialized, o.initialized);
            return *this;
    }

    ~pipewire_init_guard(){
            if(initialized)
                    pw_deinit();
    }

    void init(){
            pw_init(nullptr, nullptr);
            initialized = true;
    }

private:
    bool initialized = false;
};

struct pipewire_state_common{
    pipewire_init_guard init_guard;

    pw_thread_loop_uniq pipewire_loop;
    pw_context_uniq pipewire_context;
    pw_core_uniq pipewire_core;
    spa_hook_uniq core_listener;

    int pw_pending_seq = 0;
    int pw_last_seq = 0;
};

bool initialize_pw_common(pipewire_state_common& s);


#endif
