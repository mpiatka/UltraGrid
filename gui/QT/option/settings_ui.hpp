#ifndef SETTINGS_UI_HPP
#define SETTINGS_UI_HPP

#include <QObject>
#include <QString>
#include <memory>

#include "available_settings.hpp"
#include "ui_ultragrid_window.h"
#include "ui_settings.h"
#include "settings.hpp"
#include "checkbox_ui.hpp"
#include "actionCheckable_ui.hpp"
#include "combobox_ui.hpp"
#include "lineedit_ui.hpp"
#include "spinbox_ui.hpp"
#include "audio_opts.hpp"
#include "video_opts.hpp"

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

	std::vector<std::unique_ptr<WidgetUi>> uiControls;


	void jpegLabelCallback(Option &opt, bool suboption); 

	void fecCallback(Option &opt, bool suboption);

	void connectSignals();
	void addCallbacks();

	void addControl(WidgetUi *widget);

private slots:
	void refreshAll();

	void test();

signals:
	void changed();
};


#endif //SETTINGS_UI
