#ifndef VIDEO_OPTS_HPP
#define VIDEO_OPTS_HPP

#include <vector>
#include "available_settings.hpp"
#include "settings.hpp"

struct SettingItem;

class LineEditUi;

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoDisplay(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings);

void videoCompressBitrateCallback(LineEditUi *bitrateLine, Option &opt, bool suboption);


#endif
