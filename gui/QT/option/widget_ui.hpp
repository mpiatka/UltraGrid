#ifndef WIDGET_UI
#define WIDGET_UI

#include <QObject>
#include "settings.hpp"

class WidgetUi : public QObject{
Q_OBJECT

public:
    WidgetUi(Settings *settings, const std::string &opt);

    void setOpt(const std::string &opt);

protected:
    Settings *settings;
    std::string opt;

    void registerCallback();

    virtual void connectSignals() = 0;
    virtual void updateUiState() = 0;
	virtual void optChangeCallback(Option &opt, bool suboption) = 0;

signals:
    void changed();
};

#endif
