#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

class Settings;

class Option{
public:
	using Callback = std::function<void(Option &, bool)>;

	Option(Settings *settings,
			const std::string &name,
			const std::string &param = "",
			const std::string &value = "",
			bool enabled = true) :
		name(name),
		param(param),
		value(value),
		enabled(enabled),
		settings(settings)	{  }

	Option(Settings *settings) : settings(settings) {  }

	virtual ~Option() {  }

	std::string getName() const;
	std::string getValue() const;
	std::string getSubVals() const;
	std::string getParam() const;
	virtual std::string getLaunchOption() const;

	virtual void setValue(const std::string &val);
	void setParam(const std::string &p) { param = p; }

	bool isEnabled() const { return enabled; }
	void setEnabled(bool enable);

	void addSuboption(Option *sub, const std::string &limit = "");
	void addOnChangeCallback(Callback callback);

	Settings *getSettings();

protected:
	std::string name;
	std::string param;
	std::string value;

	bool enabled;

	void changed();
	void suboptionChanged(Option &opt, bool suboption);

	std::vector<Callback> onChangeCallbacks;
	std::vector<std::pair<std::string, Option *>> suboptions;

	Settings *settings;
};

class Settings{
public:
	Settings();

	std::string getLaunchParams() const;
	std::string getPreviewParams() const;

	const Option& getOption(const std::string &opt) const;
	Option& getOption(const std::string &opt);

	Option& addOption(std::string name,
			const std::string &param,
			const std::string &value = "",
			bool enabled = true,
			const std::string &parent = "",
			const std::string &limit = "");

private:

	std::map<std::string, std::unique_ptr<Option>> options;

	const Option dummy;
};

#endif //SETTINGS_HPP
