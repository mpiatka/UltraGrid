#ifndef LINEEDIT_UI_HPP
#define LINEEDIT_UI_HPP

#include <QLineEdit>
#include "widget_ui.hpp"

class LineEditUi : public WidgetUi{
Q_OBJECT

public:
    LineEditUi(QLineEdit *line, Settings *settings, const std::string &opt);

private:
    QLineEdit *line;

    void connectSignals() override;
    std::string getOptValue();
    void updateUiState() override;
    void updateUiState(const std::string &text);

	void optChangeCallback(Option &opt, bool suboption) override;

private slots:
    void textEdited(const QString &text);
};

#endif
