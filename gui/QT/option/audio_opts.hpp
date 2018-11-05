#ifndef AUDIO_OPTS_HPP
#define AUDIO_OPTS_HPP

#include <vector>
#include "available_settings.hpp"

struct SettingItem;

std::vector<SettingItem> getAudioCompress(AvailableSettings *availSettings);


#endif
