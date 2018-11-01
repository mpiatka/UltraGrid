#include <functional>

#include "checkbox_ui.hpp"

CheckboxUi::CheckboxUi(QAbstractButton *box, Settings *settings, const std::string &opt) :
    WidgetUi(settings, opt),
    checkbox(box)
{
    updateUiState();
    registerCallback();
    connectSignals();
}

void CheckboxUi::connectSignals(){
	connect(checkbox, &QAbstractButton::clicked, this, &CheckboxUi::boxClicked);
}

void CheckboxUi::boxClicked(bool checked){
	settings->getOption(opt).setEnabled(checked);
	emit changed();
}

void CheckboxUi::optChangeCallback(Option &changedOpt, bool suboption){
    if(changedOpt.getName() == opt){
        updateUiState(changedOpt.isEnabled());
    }
}

bool CheckboxUi::getOptValue(){
    return settings->getOption(opt).isEnabled();
}

void CheckboxUi::updateUiState(bool checked){
    checkbox->setChecked(checked);
}

void CheckboxUi::updateUiState(){
    updateUiState(getOptValue());
}

