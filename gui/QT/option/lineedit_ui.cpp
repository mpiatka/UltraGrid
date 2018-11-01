#include "lineedit_ui.hpp"

LineEditUi::LineEditUi(QLineEdit *line, Settings *settings, const std::string &opt) : 
    WidgetUi(settings, opt),
    line(line)
{
    updateUiState();
    registerCallback();
    connectSignals();
}

void LineEditUi::connectSignals(){
	connect(line, &QLineEdit::textEdited, this, &LineEditUi::textEdited);
}

void LineEditUi::textEdited(const QString &text){
	settings->getOption(opt).setValue(text.toStdString());
	emit changed();
}

void LineEditUi::optChangeCallback(Option &changedOpt, bool suboption){
    if(changedOpt.getName() == opt){
        updateUiState(changedOpt.getValue());
    }
}

std::string LineEditUi::getOptValue(){
    return settings->getOption(opt).getValue();
}

void LineEditUi::updateUiState(const std::string &text){
    line->setText(QString::fromStdString(text));
}

void LineEditUi::updateUiState(){
    updateUiState(getOptValue());
}


