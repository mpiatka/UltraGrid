#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>

class Option{
public:
private:
	std::string name;

	std::string value;

	std::vector<std::function<void(const std::string &)>> onChangeCallbacks;
};

class Settings{
public:

	std::string getLaunchParams() const;
	std::string getPreviewParams() const;

private:

	std::map<std::string, Option> options;
};

#endif //SETTINGS_HPP
