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

struct VideoMode{
	int mode_num;
	std::string format;

	enum {
		Frame_size_dicrete,
		Frame_size_stepwise,
		Frame_size_cont,
	} frame_size_type;

	union {
		struct {
			int width;
			int height;
		} discrete;
		struct {
			int min_width;
			int max_width;
			int min_height;
			int max_height;
			int step_width;
			int step_height;
		} stepwise;
	} frame_size;

	enum {
		Fps_unknown,
		Fps_discrete,
		Fps_stepwise,
		Fps_cont,
	} fps_type;

	union {
		struct {
			long long numerator;
			int denominator;
		} fraction;

		struct {
			long long min_numerator;
			int min_denominator;
			long long max_numerator;
			int max_denominator;
			long long step_numerator;
			int step_denominator;
		} stepwise;
	} fps;
};

struct Webcam{
	std::string name;
	std::string id;
	std::string type;

	std::vector<VideoMode> modes;
};

class AvailableSettings{
public:
	void query(const std::string &executable, SettingType type);
	void queryCap(const QStringList &lines, SettingType type, const char *capStr);
	void queryV4l2(const QStringList &lines);

	void queryAll(const std::string &executable);

	bool isAvailable(const std::string &name, SettingType type) const;
	std::vector<std::string> getAvailableSettings(SettingType type) const;

	std::vector<Webcam> getWebcams() const;

private:
	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Webcam> webcams;

};

#endif
