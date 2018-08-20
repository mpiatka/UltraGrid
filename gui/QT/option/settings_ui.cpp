#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include "settings_ui.hpp"

static bool vecContains(std::vector<std::string> vec, std::string str){
	for(const auto &i : vec){
		if(i == str)
			return true;
	}

	return false;
}

void SettingsUi::init(Settings *settings, AvailableSettings *availableSettings){
	this->settings = settings;
	this->availableSettings = availableSettings;
}

void SettingsUi::initMainWin(Ui::UltragridWindow *ui){
	mainWin = ui;

	connect(mainWin->actionAdvanced, SIGNAL(triggered(bool)), this, SLOT(setAdvanced(bool)));

	initVideoCompress();
	initVideoSource();

	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));

}

bool SettingsUi::isAdvancedMode(){
	return settings->getOption("advanced").isEnabled();
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

	using namespace std::placeholders;
	settings->getOption("video.compress").addOnChangeCallback(
			std::bind(&SettingsUi::videoCompressionCallback, this, _1)
			);
}

void SettingsUi::videoCompressionCallback(Option &opt){
	const std::string &val = opt.getValue();
	mainWin->videoCompressionComboBox->setCurrentText(QString::fromStdString(val));
	if(val == "jpeg"){
		mainWin->videoBitrateLabel->setText(QString("Jpeg quality"));
	} else {
		mainWin->videoBitrateLabel->setText(QString("Bitrate"));
	}

	mainWin->videoBitrateEdit->setText(QString::fromStdString(
				settings->getOption(getBitrateOpt(settings)).getValue()
				));
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

struct SettingValue{
	std::string opt;
	std::string val;
};

struct SettingItem{
	std::string name;
	std::vector<SettingValue> opts;
};

static void initTestcardModes(){
	static const struct{
		int w;
		int h;
	} resolutions[] = {
		{1280, 720},
		{1920, 1080},
		{3840, 2160},
	};

	static int const rates[] = {24, 30, 60};

	for(const auto &res : resolutions){
		for(const auto &rate : rates){
			SettingItem item;
		}
	}
}

void SettingsUi::initVideoSource(){
	QComboBox *box = mainWin->videoSourceComboBox;
	connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(setVideoSource(int)));
	connect(mainWin->videoModeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(setVideoSourceMode(int)));
	const std::vector<std::string> whiteList = {
		"testcard",
		"screen",
		"decklink",
		"aja",
		"dvs"
	};

	for(const auto &i : availableSettings->getAvailableSettings(VIDEO_SRC)){
		if(!isAdvancedMode() && !vecContains(whiteList, i))
			continue;

		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}

	using namespace std::placeholders;
	settings->getOption("video.source").addOnChangeCallback(
			std::bind(&SettingsUi::videoSourceCallback, this, _1)
			);

}

void SettingsUi::setVideoSource(int idx){
	
}

void SettingsUi::setVideoSourceMode(int idx){
	
}

void SettingsUi::videoSourceCallback(Option &opt){
	
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;


}

void SettingsUi::setAdvanced(bool enable){
	settings->getOption("advanced").setEnabled(enable);
}
