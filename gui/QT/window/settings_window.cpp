#include <QIntValidator> 
#include "settings_window.hpp"

SettingsWindow::SettingsWindow(QWidget *parent): QDialog(parent){
	ui.setupUi(this);
	ui.basePort->setValidator(new QIntValidator(0, 65535, this));
	ui.controlPort->setValidator(new QIntValidator(0, 65535, this));

}

void SettingsWindow::init(SettingsUi *settingsUi){
	settingsUi->initSettingsWin(&ui);
}

