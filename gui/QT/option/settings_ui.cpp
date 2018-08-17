#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include "settings_ui.hpp"

void SettingsUi::init(Settings *settings, AvailableSettings *availableSettings){
	this->settings = settings;
	this->availableSettings = availableSettings;
}

void SettingsUi::initMainWin(Ui::UltragridWindow *ui){
	mainWin = ui;

	connect(mainWin->actionAdvanced, SIGNAL(triggered(bool)), this, SLOT(setAdvanced(bool)));

	initVideoCompress();

	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));

}

struct VideoCompressItem{
	const char * displayName;
	const char * value;
	bool isLibav;
};

Q_DECLARE_METATYPE(VideoCompressItem);

static const VideoCompressItem videoCodecs[] = {
	{"none", "", false},
	{"H.264", "H.264", true},
	{"H.265", "H.265", true},
	{"MJPEG", "MJPEG", true},
	{"VP8", "VP8", true},
	{"Jpeg", "jpeg", false},
};

static bool isLibavCodec(const std::string &str){
	for(const auto &i : videoCodecs){
		if(i.value == str && i.isLibav)
			return true;
	}

	return false;
}

static std::string getBitrateOpt(Settings *settings){
	std::string codec = settings->getOption("video.compress").getValue();
	std::string opt = "video.compress." + codec;
	if(codec == "libavcodec"){
		opt += ".codec."
			+ settings->getOption("video.compress.libavcodec.codec").getValue()
			+ ".bitrate";
	} else if (codec == "jpeg") {
		opt += ".quality";
	} else {
		opt += codec + ".bitrate";
	}

	return opt;
}

void SettingsUi::test(){
	printf("%s\n", settings->getLaunchParams().c_str());
}

void SettingsUi::initVideoCompress(){
	QComboBox *box = mainWin->videoCompressionComboBox;
	connect(box, SIGNAL(activated(int)), this, SLOT(setVideoCompression(int)));
	connect(mainWin->videoBitrateEdit, SIGNAL(textChanged(const QString&)), this, SLOT(setVideoBitrate(const QString &)));
	/*
	for(const auto &i : availableSettings->getAvailableSettings(VIDEO_COMPRESS)){
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
	*/

	for(const auto& item : videoCodecs){
		box->addItem(QString(item.displayName), QVariant::fromValue(item));
	}

	settings->getOption("video.compress").addOnChangeCallback(
			[ui=mainWin, settings=settings](const std::string &name, const std::string &val, bool enabled)
			{
				ui->videoCompressionComboBox->setCurrentText(QString::fromStdString(val));
				if(val == "jpeg"){
					ui->videoBitrateLabel->setText(QString("Jpeg quality"));
				} else {
					ui->videoBitrateLabel->setText(QString("Bitrate"));
				}

				ui->videoBitrateEdit->setText(QString::fromStdString(
						settings->getOption(getBitrateOpt(settings)).getValue()
						));
			}
			);
}

void SettingsUi::setVideoCompression(int idx){
	QComboBox *box = mainWin->videoCompressionComboBox;

	VideoCompressItem i = box->itemData(idx).value<VideoCompressItem>();
	std::string codec;
	if(i.isLibav){
		codec = "libavcodec";
		settings->getOption("video.compress.libavcodec.codec").setValue(i.value);
	} else {
		codec = i.value;
	}

	settings->getOption("video.compress").setValue(codec);
}

void SettingsUi::setVideoBitrate(const QString &str){
	std::string opt = getBitrateOpt(settings);

	std::cout << opt << ": " << str.toStdString() << std::endl;
	settings->getOption(opt).setValue(str.toStdString());
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;


}

void SettingsUi::setAdvanced(bool enable){
	settings->getOption("advanced").setEnabled(enable);
}
