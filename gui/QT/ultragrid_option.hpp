#ifndef ULTRAGRID_OPTION_HPP
#define ULTRAGRID_OPTION_HPP

#include <QString>
#include <QObject>
#include "ui_ultragrid_window.h"

class QComboBox;
class QLineEdit;
class QLabel;

class UltragridWindow;

class UltragridOption : public QObject{
	Q_OBJECT
public:
	virtual QString getLaunchParam() = 0;
	virtual void setAdvanced(bool advanced);

public signals:
	void changed();

public slots:
	virtual void update() = 0;

protected:
	UltragridOption() :
		advanced(false) {  }

private:
	bool advanced;
};

class ComboBoxOption : public UltragridOption{
public:
	ComboBoxOption(QComboBox *box,
			const QString& ultragridExecutable,
			QString opt);

	virtual QString getLaunchParam() override;
	virtual void queryAvailOpts();

	QString getCurrentValue();

protected:
	QStringList getAvailOpts();

	void resetComboBox(QComboBox *box);

	// Used to filter options with whitelists, blacklists, etc.
	virtual bool filter(QString &item) { return true; }

	// Used to query options not returned from the help command, e.g. v4l2 devices
	virtual void queryExtraOpts() {  }

	// Returns extra launch params like video mode, bitrate, etc.
	virtual QString getExtraParams() { return ""; }

	void setItem(const QVariant &data);
	void setItem(const QString &data);

private:
	QComboBox *box;

	QString ultragridExecutable;
	QString opt;
};

class SourceOption : public ComboBoxOption{
	Q_OBJECT
public:

	SourceOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

protected:
	virtual bool filter(QString &item) override;
	virtual void queryExtraOpts(const QStringList &opts) override;
	virtual QString getExtraParams() override;
private:
	Ui::UltragridWindow *ui;

private slots:
	void srcChanged();
};

class DisplayOption : public UltragridOption{
	Q_OBJECT
public:
	DisplayOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	virtual bool filter(QString &item) override;
	virtual QString getLaunchParam() override;

private:
	Ui::UltragridWindow *ui;
	bool preview;

private slots:
	void enablePreview(bool);
};

class VideoCompressOption : public UltragridOption{
	Q_OBJECT
public:
	VideoCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class AudioSourceOption : public UltragridOption{
	Q_OBJECT
public:
	AudioSourceOption(Ui::UltragridWindow *ui,
			const SourceOption *videoSrc,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;
	const SourceOption *videoSource;

private slots:
	void update();
};

class AudioPlaybackOption : public UltragridOption{
	Q_OBJECT
public:
	AudioPlaybackOption(Ui::UltragridWindow *ui,
			const DisplayOption *videoDisplay,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;
	const DisplayOption *videoDisplay;

private slots:
	void update();
};

class AudioCompressOption : public UltragridOption{
	Q_OBJECT
public:
	AudioCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class GenericOption : public UltragridOption{
public:
	GenericOption(QComboBox *box,
			const QString& ultragridExecutable,
			QString opt);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	QComboBox *box;
};

class FecOption : public UltragridOption{
	Q_OBJECT
public:
	FecOption(Ui::UltragridWindow *ui);

	QString getLaunchParam() override;
	void queryAvailOpts() override {  }

	void setAdvanced(bool advanced) override {
		this->advanced = advanced;
		update();
	}

private:
	Ui::UltragridWindow *ui;

private slots:
	void update();
};

#endif
