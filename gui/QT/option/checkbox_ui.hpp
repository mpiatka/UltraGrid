#ifndef CHECKBOX_UI_HPP
#define CHECKBOX_UI_HPP

#include <QAbstractButton>
#include <string>

#include "widget_ui.hpp"
#include "settings.hpp"

class CheckboxUi : public WidgetUi{
    Q_OBJECT

public:
    CheckboxUi(QAbstractButton *box, Settings *settings, const std::string &opt);

private:
    QAbstractButton *checkbox;

    void connectSignals() override;
    bool getOptValue();
    void updateUiState() override;
    void updateUiState(bool checked);

	void optChangeCallback(Option &opt, bool suboption) override;

private slots:
    void boxClicked(bool checked);
};

#endif
