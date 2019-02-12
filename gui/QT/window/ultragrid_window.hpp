#ifndef ULTRAGRID_WINDOW_HPP
#define ULTRAGRID_WINDOW_HPP

#include <QProcess>
#include <vector>
#include <memory>

#include "ui_ultragrid_window.h"
#include "ultragrid_option.hpp"
#include "log_window.hpp"
#include "settings_window.hpp"
#include "available_settings.hpp"
#include "settings.hpp"

class UltragridWindow : public QMainWindow{
	Q_OBJECT

public:
	UltragridWindow(QWidget *parent = 0);
	static QString findUltragridExecutable();

protected:
	void closeEvent(QCloseEvent *);

private:
	void checkPreview();

	Ui::UltragridWindow ui;

	QString ultragridExecutable;
	QProcess process;
	QProcess previewProcess;

	AvailableSettings availableSettings;

	QString launchArgs;
	QStringList getOptionsForParam(QString param);
	LogWindow log;
	SettingsWindow settingsWindow;

	std::vector<std::unique_ptr<UltragridOption>> opts;
	VideoSourceOption *sourceOption;
	VideoDisplayOption *displayOption;

	AudioSourceOption *audioSrcOption;

	QTimer previewTimer;
	Settings settings;


public slots:
	void about();
	void outputAvailable();
	void start();

	void editArgs(const QString &text);
	void setArgs();

	void showLog();
	void showSettings();

	void startPreview();
	void stopPreview();

private slots:
	void queryOpts();
	void setAdvanced(bool);
	void setStartBtnText(QProcess::ProcessState);
	void processStateChanged(QProcess::ProcessState);
	void enablePreview(bool);
	void schedulePreview();
};


#endif
