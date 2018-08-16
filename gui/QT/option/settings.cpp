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

std::string Option::getLaunchOption() const{
	if(!enabled || value.empty())
		return "";

	std::string out = param + value;

	for(const auto &sub : suboptions){
		if(sub.first.empty() || sub.first == value){
			out += sub.second->getLaunchOption();
		}
	}

	addSpace(out);

	return out;
}

void Option::changed(){
	for(const auto &callback : onChangeCallbacks){
		callback(name, value, enabled);
	}
}

void Option::setValue(const std::string &val){
	value = val;
	changed();
}

void Option::setEnabled(bool enable){
	enabled = enable;
	changed();
}

void Option::addSuboption(Option *sub, const std::string &limit){
	sub->addOnChangeCallback(std::bind(&Option::changed, this));
	suboptions.push_back(std::make_pair(limit, sub));
}

void Option::addOnChangeCallback(Callback callback){
	onChangeCallbacks.push_back(callback);
}

static void test_callback(const std::string &name, const std::string &val, bool enabled){
	std::cout << "Callback: " << name << ": " << val << " (" << enabled << ")" << std::endl;
}

Settings::Settings(){
	addOption("video.source", "-t ").addOnChangeCallback(test_callback);
	addOption("testcard.width", ":", "1920", true, "video.source", "testcard");
	addOption("screen.width", ":", "1080", true, "video.source", "screen").addOnChangeCallback(test_callback);
;

	addOption("video.display", "-d ");
	addOption("gl.novsync", ":", "novsync", true, "video.display", "gl");

	addOption("advanced", "", "", false).addOnChangeCallback(test_callback);

	getOption("video.source").setValue("screen");
	getOption("video.source.screen.width").setValue("1920");
	getOption("video.display").setValue("gl");

	std::cout << getLaunchParams() << std::endl;
}

std::string Settings::getLaunchParams() const{
	std::string out;
	out += getOption("video.source").getLaunchOption();
	out += getOption("video.display").getLaunchOption();
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
