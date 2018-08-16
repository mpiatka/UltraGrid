#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

class Option{
	using Callback = std::function<void(const std::string &, const std::string &, bool)>;
public:
	Option(const std::string &name,
			const std::string &param = "",
			const std::string &value = "",
			bool enabled = true) :
		name(name),
		param(param),
		value(value),
		enabled(enabled) {  }

	Option() {  }

	virtual ~Option() {  }

	std::string getName() const;
	std::string getValue() const;
	std::string getParam() const;
	virtual std::string getLaunchOption() const;

	virtual void setValue(const std::string &val);
	void setParam(const std::string &p) { param = p; }

	bool isEnabled() const { return enabled; }
	void setEnabled(bool enable);

	void addSuboption(Option *sub, const std::string &limit = "");
	void addOnChangeCallback(Callback callback);

protected:
	std::string name;
	std::string param;
	std::string value;

	bool enabled;

	void changed();

	std::vector<Callback> onChangeCallbacks;
	std::vector<std::pair<std::string, Option *>> suboptions;
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

	static const Option dummy;
};

#endif //SETTINGS_HPP
