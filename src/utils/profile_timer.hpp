#ifndef PROFILE_TIMER_HPP
#define PROFILE_TIMER_HPP

#ifdef BUILD_PROFILED

#include <chrono>
#include <vector>
#include <mutex>
#include <thread>
#include <fstream>

#define THREAD_EVENT_BUF_SIZE 256

#define PROFILE_FUNC \
    Profile_timer PROFILER_PROFILE_TIMER_FUNC(Profiler_thread_inst::get_instance(), __PRETTY_FUNCTION__); \
    Profile_timer PROFILER_PROFILE_TIMER_DETAIL(Profiler_thread_inst::get_instance(), "");

#define PROFILE_DETAIL(name) \
    PROFILER_PROFILE_TIMER_DETAIL = Profile_timer(Profiler_thread_inst::get_instance(), (name));


struct Profile_event {
    Profile_event(std::string name, std::thread::id thread_id) : 
        name(name),
        start(std::chrono::steady_clock::now()),
        end(std::chrono::steady_clock::now()),
        thread_id(thread_id) {  }

    std::string name;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    std::thread::id thread_id;
};

class Profiler_thread_inst;

class Profiler {
friend Profiler_thread_inst;
public:
    static Profiler& get_instance();

    void write_events(const std::vector<Profile_event>& events);

private:
    Profiler(const std::string& filename);

    Profiler() : Profiler("ug_profile.json") {  }

    std::mutex io_mut;
    std::ofstream out_file;
    std::thread::id pid;

    static std::chrono::steady_clock::time_point start_point;
};

class Profile_timer;

class Profiler_thread_inst {
friend Profile_timer;
public:

    Profiler_thread_inst(Profiler& profiler);
    ~Profiler_thread_inst();

    static Profiler_thread_inst& get_instance();

    void add_event(Profile_event&& event);

private:
    Profiler& profiler;
    std::vector<Profile_event> thread_events;
};

class Profile_timer {
friend Profiler;
public:
    Profile_timer(Profiler_thread_inst& profiler, std::string name);
    ~Profile_timer();

    Profile_timer& operator=(Profile_timer&& rhs);

private:
    void commit();

    Profiler_thread_inst& profiler;
    Profile_event event;
};

#else //BUILD_PROFILED

#define PROFILE_FUNC 
#define PROFILE_DETAIL(name)

#endif //BUILD_PROFILED

#endif
