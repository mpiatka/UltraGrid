#ifdef BUILD_PROFILED

#include <cstdlib>
#include <cstring>
#include "profile_timer.hpp"

std::chrono::steady_clock::time_point Profiler::start_point;

Profiler& Profiler::get_instance(){
    static Profiler inst;
    if(inst.start_point.time_since_epoch() == std::chrono::steady_clock::duration::zero()){
        inst.start_point = std::chrono::steady_clock::now();
    }
    return inst;
}

void Profiler::write_events(const std::vector<Profile_event>& events){
    if(!active)
        return;

    std::lock_guard<std::mutex> lock(io_mut);
    for(const auto& event : events){
        out_file << "{ \"name\": \"" << event.name << "\""
            << ", \"cat\": \"function\""
            << ", \"ph\": \"X\""
            << ", \"pid\": \"" << pid << "\""
            << ", \"tid\": \"" << event.thread_id << "\""
            << ", \"ts\": \"" << std::chrono::duration_cast<std::chrono::microseconds>(event.start - start_point).count() << "\""
            << ", \"dur\": \"" << std::chrono::duration_cast<std::chrono::microseconds>(event.end - event.start).count() << "\""
            << "},\n ";
    }
}

Profiler::Profiler(){
    const char *env_p = std::getenv("UG_PROFILE");
    active = env_p && env_p[0] != '\0';
    if(!active){
        return;
    }

    std::string filename = "ug_";
    filename += env_p;
    filename += ".json";

    pid = std::this_thread::get_id();
    out_file.open(filename);

    out_file << "[ ";
}

Profiler_thread_inst::Profiler_thread_inst(Profiler& profiler) : profiler(profiler) {
    thread_events.reserve(THREAD_EVENT_BUF_SIZE);
}

Profiler_thread_inst& Profiler_thread_inst::get_instance(){
    static thread_local Profiler_thread_inst inst(Profiler::get_instance());
    return inst;
}

Profiler_thread_inst::~Profiler_thread_inst(){
    profiler.write_events(thread_events);
}

void Profiler_thread_inst::add_event(Profile_event&& event){
    if(thread_events.size() == THREAD_EVENT_BUF_SIZE){
        profiler.write_events(thread_events);
        thread_events.clear();
    }

    thread_events.emplace_back(std::move(event));
}

Profile_timer::Profile_timer(Profiler_thread_inst& profiler, std::string name) :
    profiler(profiler),
    event(name, std::this_thread::get_id())
{

}

Profile_timer::~Profile_timer(){
    commit();
}

Profile_timer& Profile_timer::operator=(Profile_timer&& rhs) {
    std::swap(event, rhs.event);
    return *this;
}

void Profile_timer::commit(){
    if(event.name.empty())
        return;

    event.end = std::chrono::steady_clock::now();
    profiler.add_event(std::move(event));
}

#endif //BUILD_PROFILED
