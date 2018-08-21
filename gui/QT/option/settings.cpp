#include <iostream>
#include "settings.hpp"

std::string Option::getName() const{
	return name;
}

std::string Option::getValue() const{
	return value;
}

std::string Option::getParam() const{
	return param;
}

static void addSpace(std::string &str){
	if(!str.empty() && str.back() != ' '){
		str += ' ';
	}
}

std::string Option::getSubVals() const{
	std::string out;

	for(const auto &sub : suboptions){
		if(sub.first.empty() || sub.first == value){
			out += sub.second->getLaunchOption();
		}
	}

	return out;
}

std::string Option::getLaunchOption() const{
	if(!enabled || value.empty())
		return "";

	std::string out = param + value;
	out += getSubVals();

	return out;
}

void Option::changed(){
	for(const auto &callback : onChangeCallbacks){
		callback(*this);
	}
}

void Option::suboptionChanged(Option &opt){
	for(const auto &callback : onChangeCallbacks){
		callback(opt);
	}
}

void Option::setValue(const std::string &val){
	value = val;
	enabled = !val.empty();

	changed();
}

void Option::setEnabled(bool enable){
	enabled = enable;
	changed();
}

void Option::addSuboption(Option *sub, const std::string &limit){
	using namespace std::placeholders;
	sub->addOnChangeCallback(std::bind(&Option::suboptionChanged, this, _1));
	suboptions.push_back(std::make_pair(limit, sub));
}

void Option::addOnChangeCallback(Callback callback){
	onChangeCallbacks.push_back(callback);
}

static void test_callback(Option &opt){
	std::cout << "Callback: " << opt.getName()
		<< ": " << opt.getValue()
		<< " (" << opt.isEnabled() << ")" << std::endl;
}

const static struct{
	const char *name;
	const char *param;
	const char *defaultVal;
	bool enabled;
	const char *parent;
	const char *limit;
} optionList[] = {
	{"video.source", " -t ", "", false, "", ""},
	{"testcard.width", ":", "", false, "video.source", "testcard"},
	{"testcard.height", ":", "", false, "video.source", "testcard"},
	{"testcard.fps", ":", "", false, "video.source", "testcard"},
	{"testcard.format", ":", "", false, "video.source", "testcard"},
	{"screen.fps", ":fps=", "", false, "video.source", "screen"},
	{"video.display", " -d ", "", false, "", ""},
	{"gl.novsync", ":", "novsync", false, "video.display", "gl"},
	{"video.compress", " -c ", "", false, "", ""},
	{"libavcodec.codec", ":codec=", "", false, "video.compress", "libavcodec"},
	{"H.264.bitrate", ":bitrate=", "", false, "video.compress.libavcodec.codec", "H.264"},
	{"H.265.bitrate", ":bitrate=", "", false, "video.compress.libavcodec.codec", "H.265"},
	{"MJPEG.bitrate", ":bitrate=", "", false, "video.compress.libavcodec.codec", "MJPEG"},
	{"VP8.bitrate", ":bitrate=", "", false, "video.compress.libavcodec.codec", "VP8"},
	{"jpeg.quality", ":", "", false, "video.compress", "jpeg"},
	{"audio.source", " -s ", "", false, "", ""},
	{"audio.playback", " -r ", "", false, "", ""},
	{"advanced", "", "", false, "", ""},
	{"preview.display", "", "", true, "", ""},
};

Settings::Settings(){
	for(const auto &i : optionList){
		auto &opt = addOption(i.name,
				i.param,
				i.defaultVal,
				i.enabled,
				i.parent,
				i.limit);
		if(!i.parent[0])
			opt.addOnChangeCallback(test_callback);
	}

	std::cout << getLaunchParams() << std::endl;
}

std::string Settings::getLaunchParams() const{
	std::string out;
	out += getOption("video.source").getLaunchOption();
	out += getOption("video.compress").getLaunchOption();
	const auto &dispOpt = getOption("video.display");
	if(dispOpt.isEnabled() && getOption("preview.display").isEnabled()){
		out += dispOpt.getParam();
		out += "multiplier:" + dispOpt.getValue() + dispOpt.getSubVals();
		out += "#preview";
	} else {
		out += getOption("video.display").getLaunchOption();
	}
	out += getOption("audio.source").getLaunchOption();
	out += getOption("audio.playback").getLaunchOption();
	return out;
}

std::string Settings::getPreviewParams() const{
	return "";
}

Option& Settings::getOption(const std::string &opt){
	auto search = options.find(opt);

	if(search == options.end()){
		std::unique_ptr<Option> newOption(new Option(opt));
		auto p = newOption.get();
		options[opt] = std::move(newOption);
		return *p;
	}

	return *search->second;
}

const Option Settings::dummy;

const Option& Settings::getOption(const std::string &opt) const{
	auto search = options.find(opt);

	if(search == options.end())
		return dummy;

	return *search->second;
}

Option& Settings::addOption(std::string name,
		const std::string &param,
		const std::string &value,
		bool enabled,
		const std::string &parent,
		const std::string &limit)
{
	if(!parent.empty())
		name = parent + "." + name;

	auto search = options.find(name);

	Option *opt;

	if(search == options.end()){
		std::unique_ptr<Option> newOption(new Option(name));
		opt = newOption.get();
		options[name] = std::move(newOption);
	} else {
		opt = search->second.get();
	}

	opt->setParam(param);
	opt->setValue(value);
	opt->setEnabled(enabled);

	if(!parent.empty()){
		getOption(parent).addSuboption(opt, limit);
	}

	return *opt;
}
