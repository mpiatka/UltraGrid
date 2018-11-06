#include <cstring>

#include "ui_ultragrid_window.h"

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

const char * const sdiAudioCards[] = {
    "decklink",
    "aja",
    "dvs",
    "deltacast"
};

const char * const sdiAudio[] = {
    "analog",
    "AESEBU",
    "embedded"
};

std::vector<SettingItem> getAudioSrc(AvailableSettings *availSettings){
    const std::string optStr = "audio.source";
    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getAvailableSettings(AUDIO_SRC)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        res.push_back(std::move(item));
    }

    return res;
	box->addItem("none", QVariant(""));

	std::string vid = settings->getOption("video.source").getValue();

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_SRC)){
		if(!settings->isAdvancedMode() && vecContains(sdiAudio, i) && !vecContains(sdiAudioCards, vid))
			continue;
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
	setItem(box, prevData);
}

void audioCompressionCallback(Ui::UltragridWindow *win, Option &opt, bool suboption){
	if(suboption)
		return;

	const char *const losslessCodecs[] = {
        "",
		"FLAC",
		"u-law",
		"A-law",
		"PCM"
	};

	bool enableBitrate = true;

    for(const auto str : losslessCodecs){
        if(std::strcmp(str, opt.getValue().c_str()) == 0){
            enableBitrate = false;
            break;
        }
    }

	opt.getSettings()->getOption("audio.compress.bitrate").setEnabled(enableBitrate);
	win->audioBitrateEdit->setEnabled(enableBitrate);
	win->audioBitrateLabel->setEnabled(enableBitrate);
}

