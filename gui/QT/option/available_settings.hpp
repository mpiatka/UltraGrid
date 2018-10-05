#ifndef AVAILABLE_SETTINGS_HPP
#define AVAILABLE_SETTINGS_HPP

#include <string>
#include <vector>

enum SettingType{
	VIDEO_SRC = 0,
	VIDEO_DISPLAY,
	VIDEO_COMPRESS,
	VIDEO_CAPTURE_FILTER,
	AUDIO_SRC,
	AUDIO_PLAYBACK,
	AUDIO_COMPRESS,
	SETTING_TYPE_COUNT
};

struct Webcam{
	std::string name;
	std::string type;
	std::string config;
};

class AvailableSettings{
public:
	void query(const std::string &executable, SettingType type);
	void queryCap(const QStringList &lines, SettingType type, const char *capStr);
	void queryV4l2(const std::string &executable);

	void queryAll(const std::string &executable);

	bool isAvailable(const std::string &name, SettingType type) const;
	std::vector<std::string> getAvailableSettings(SettingType type) const;

private:
	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Webcam> webcams;

};

#endif
