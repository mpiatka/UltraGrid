#ifndef SETTINGS_UI_HPP
#define SETTINGS_UI_HPP

#include <QObject>
#include <QString>

#include "available_settings.hpp"
#include "ui_ultragrid_window.h"
#include "ui_settings.h"
#include "settings.hpp"

class SettingsUi : public QObject{
	Q_OBJECT

public:
	void init(Settings *settings, AvailableSettings *availableSettings);
	void initMainWin(Ui::UltragridWindow *ui);
	void initSettingsWin(Ui::Settings *ui);

private:
	Ui::UltragridWindow *mainWin = nullptr;
	Ui::Settings *settingsWin = nullptr;
	Settings *settings = nullptr;
	AvailableSettings *availableSettings = nullptr;


	void populateComboBox(QComboBox *box,
			SettingType type,
			const std::vector<std::string> &whitelist = {});

	void refreshVideoCompress();
	void refreshVideoSource();
	void refreshVideoDisplay();

	void refreshAudioSource();
	void refreshAudioPlayback();
	void refreshAudioCompression();

	void videoCompressionCallback(Option &opt, bool suboption); 
	void videoSourceCallback(Option &opt, bool suboption); 
	void videoDisplayCallback(Option &opt, bool suboption); 

	void audioSourceCallback(Option &opt, bool suboption);
	void audioPlaybackCallback(Option &opt, bool suboption);
	void audioCompressionCallback(Option &opt, bool suboption);

	bool isAdvancedMode();

	void setComboBox(QComboBox *box, const std::string &opt, int idx);
	void setString(const std::string &opt, const QString &str);
	void setBool(const std::string &opt, bool b);

	void connectSignals();
	void addCallbacks();

private slots:
	void setVideoCompression(int idx);
	void setVideoBitrate(const QString &);
	void setVideoSourceMode(int idx);

	void refreshAll();

	void test();

signals:
	void changed();
};


#endif //SETTINGS_UI
