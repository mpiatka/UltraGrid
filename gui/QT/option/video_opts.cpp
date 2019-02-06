#include <cstring>

#include "video_opts.hpp"
#include "combobox_ui.hpp"
#include "lineedit_ui.hpp"

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings){
	const char * const whiteList[] = {
//		"testcard",
//		"screen",
		"decklink",
		"aja",
		"dvs"
	};
    const std::string optStr = "video.source";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getAvailableSettings(VIDEO_SRC)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        bool whiteListed = false;
        for(const auto &white : whiteList){
            if(std::strcmp(white, i.c_str()) == 0){
                whiteListed = true;
            }
        }
        if(!whiteListed){
            item.conditions.push_back({{{"advanced", "t"}, false}});
        }
        res.push_back(std::move(item));
    }

    for(const auto &i : availSettings->getCapturers()){
        SettingItem item;
        item.name = i.name;
        item.opts.push_back({optStr, i.type});
        item.opts.push_back({optStr + "." + i.type + ".device", i.deviceOpt});
        res.push_back(std::move(item));
    }

    return res;
}

static std::string getResolutionStr(int w, int h, char separator){
	return std::to_string(w) + separator + std::to_string(h);
}

static void getTestcardModes(std::vector<SettingItem> &result){
	static const struct{
		int w;
		int h;
	} resolutions[] = {
		{1280, 720},
		{1920, 1080},
		{3840, 2160},
	};

	static int const rates[] = {24, 30, 60};

    const std::vector<std::vector<ConditionItem>> condition = {
        {{{"video.source", "testcard"}, false}}
    };

	SettingItem item;
	item.name = "Default";
	item.opts.push_back({"video.source.testcard.width", ""});
	item.opts.push_back({"video.source.testcard.height", ""});
	item.opts.push_back({"video.source.testcard.fps", ""});
	item.opts.push_back({"video.source.testcard.format", ""});
    item.conditions = condition;
    result.push_back(std::move(item));

	for(const auto &res : resolutions){
		for(const auto &rate : rates){
			item.opts.clear();
			item.name = getResolutionStr(res.w, res.h, 'x');
			item.name += ", " + std::to_string(rate) + " fps";
			item.opts.push_back({"video.source.testcard.width", std::to_string(res.w)});
			item.opts.push_back({"video.source.testcard.height", std::to_string(res.h)});
			item.opts.push_back({"video.source.testcard.fps", std::to_string(rate)});
			item.opts.push_back({"video.source.testcard.format", "UYVY"});
            item.conditions = condition;

            result.push_back(std::move(item));
		}
	}
}

static void getScreenModes(std::vector<SettingItem> &result){
	static int const rates[] = {24, 30, 60};

    const std::vector<std::vector<ConditionItem>> condition = {
        {{{"video.source", "screen"}, false}}
    };

	SettingItem item;
	item.name = "Default";
	item.opts.push_back({"video.source.screen.fps", ""});
    item.conditions = condition;

    result.push_back(std::move(item));

	for(const auto &rate : rates){
		item.opts.clear();
		item.name = std::to_string(rate) + " fps";
		item.opts.push_back({"video.source.screen.fps", std::to_string(rate)});
        item.conditions = condition;

        result.push_back(std::move(item));
	}
}

std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    //getTestcardModes(res);
    //getScreenModes(res);
    //

    for(const auto &cap : availSettings->getCapturers()){
        for(const auto &mode : cap.modes){
            SettingItem item;
            item.name = mode.name;
            item.conditions.push_back({{{"video.source", cap.type}, false}});
            for(const auto &opt : mode.opts){
                item.opts.push_back(
                        {"video.source." + cap.type + "." + opt.opt, opt.val});
            }
            res.push_back(std::move(item));
        } 
    }

    return res;
}

std::vector<SettingItem> getVideoDisplay(AvailableSettings *availSettings){
	const char * const whiteList[] = {
		"gl",
		"sdl",
		"decklink",
		"aja",
		"dvs"
	};

    const std::string optStr = "video.display";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getAvailableSettings(VIDEO_DISPLAY)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        bool whiteListed = false;
        for(const auto &white : whiteList){
            if(std::strcmp(white, i.c_str()) == 0){
                whiteListed = true;
            }
        }
        if(!whiteListed){
            item.conditions.push_back({{{"advanced", "t"}, false}});
        }
        res.push_back(std::move(item));
    }

    return res;
}

struct VideoCompressItem{
	const char * displayName;
	const char * value;
	bool isLibav;
};

static const VideoCompressItem videoCodecs[] = {
	{"None", "", false},
	{"H.264", "H.264", true},
	{"H.265", "H.265", true},
	{"MJPEG", "MJPEG", true},
	{"VP8", "VP8", true},
	{"Jpeg", "jpeg", false},
};

static bool isLibavCodec(const std::string &str){
	for(const auto &i : videoCodecs){
		if(i.value == str && i.isLibav)
			return true;
	}

	return false;
}

std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    const std::string optStr = "video.compress";

    for(const auto &i : videoCodecs){
        SettingItem item;
        item.name = i.displayName;
        std::string value;
        if(i.isLibav){
            value = "libavcodec";
            item.opts.push_back({optStr + ".libavcodec.codec", i.value});
        } else {
            value = i.value;
        }
        item.opts.push_back({optStr, value});

        res.push_back(std::move(item));
    }

    return res;
}

static std::string getBitrateOpt(Settings *settings){
	std::string codec = settings->getOption("video.compress").getValue();
	std::string opt = "video.compress." + codec;
	if(codec == "libavcodec"){
		opt += ".codec."
			+ settings->getOption("video.compress.libavcodec.codec").getValue()
			+ ".bitrate";
	} else if (codec == "jpeg") {
		opt += ".quality";
	} else {
		opt += codec + ".bitrate";
	}

	return opt;
}

void videoCompressBitrateCallback(LineEditUi *bitrateLine, Option &opt, bool suboption){
    if(suboption)
        return;

    bitrateLine->setOpt(getBitrateOpt(opt.getSettings()));
}
