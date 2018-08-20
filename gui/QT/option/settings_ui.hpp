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

	void initVideoCompress();
	void initVideoSource();

	void videoCompressionCallback(Option &opt); 
	void videoSourceCallback(Option &opt); 

	bool isAdvancedMode();

private slots:
	void setAdvanced(bool enable);
	void setVideoCompression(int idx);
	void setVideoBitrate(const QString &);
	void setVideoSource(int idx);
	void setVideoSourceMode(int idx);
	void test();
};


#endif //SETTINGS_UI
