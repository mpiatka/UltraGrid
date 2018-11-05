#include "audio_opts.hpp"
#include "combobox_ui.hpp"

std::vector<SettingItem> getAudioCompress(AvailableSettings *availSettings){
    const std::string optStr = "audio.compress";
    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getAvailableSettings(AUDIO_COMPRESS)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        res.push_back(std::move(item));
    }

    return res;
}
