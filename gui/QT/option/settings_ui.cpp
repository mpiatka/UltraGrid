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
	initVideoDisplay();
	initAudioSource();
	initAudioPlayback();

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

Q_DECLARE_METATYPE(SettingItem);

static std::string getResolutionStr(int w, int h, char separator){
	return std::to_string(w) + separator + std::to_string(h);
}

void SettingsUi::populateComboBox(QComboBox *box,
		SettingType type,
		const std::vector<std::string> &whitelist)
{
	box->addItem("none", QVariant(""));

	for(const auto &i : availableSettings->getAvailableSettings(type)){
		if(!isAdvancedMode() && !vecContains(whitelist, i))
			continue;

		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
}

static void initTestcardModes(QComboBox *box){
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
			item.name = getResolutionStr(res.w, res.h, 'x');
			item.name += ", " + std::to_string(rate) + " fps";
			item.opts.push_back({"video.source.testcard.width", std::to_string(res.w)});
			item.opts.push_back({"video.source.testcard.height", std::to_string(res.h)});
			item.opts.push_back({"video.source.testcard.fps", std::to_string(rate)});
			item.opts.push_back({"video.source.testcard.format", "UYVY"});

			box->addItem(QString::fromStdString(item.name),
					QVariant::fromValue(item));
		}
	}
}

static void initScreenModes(QComboBox *box){
	SettingItem item;
	
	static int const rates[] = {24, 30, 60};

	for(const auto &rate : rates){
		item.name = std::to_string(rate) + " fps";
		item.opts.push_back({"video.source.screen.fps", std::to_string(rate)});

		box->addItem(QString::fromStdString(item.name),
				QVariant::fromValue(item));
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

	populateComboBox(box, VIDEO_SRC, whiteList);

	using namespace std::placeholders;
	settings->getOption("video.source").addOnChangeCallback(
			std::bind(&SettingsUi::videoSourceCallback, this, _1)
			);

}

void SettingsUi::setVideoSource(int idx){
	std::string val = mainWin->videoSourceComboBox->itemData(idx).toString().toStdString();
	settings->getOption("video.source").setValue(val);
}

void SettingsUi::setVideoSourceMode(int idx){
	const SettingItem &i = mainWin->videoModeComboBox->itemData(idx).value<SettingItem>();

	for(const auto &opt : i.opts){
		settings->getOption(opt.opt).setValue(opt.val);
	}
}

void SettingsUi::videoSourceCallback(Option &opt){
	if(opt.getName() != "video.source")
		return;

	mainWin->videoModeComboBox->clear();
	if(opt.getValue() == "testcard"){
		initTestcardModes(mainWin->videoModeComboBox);
	} else if (opt.getValue() == "screen"){
		initScreenModes(mainWin->videoModeComboBox);
	}
}

void SettingsUi::initVideoDisplay(){
	QComboBox *box = mainWin->videoDisplayComboBox;
	connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(setVideoDisplay(int)));
	const std::vector<std::string> whiteList = {
		"gl",
		"sdl",
		"decklink",
		"aja",
		"dvs"
	};

	populateComboBox(box, VIDEO_DISPLAY, whiteList);

	using namespace std::placeholders;
	settings->getOption("video.display").addOnChangeCallback(
			std::bind(&SettingsUi::videoDisplayCallback, this, _1)
			);

}

void SettingsUi::setVideoDisplay(int idx){
	std::string val = mainWin->videoDisplayComboBox->itemData(idx).toString().toStdString();
	settings->getOption("video.display").setValue(val);
}

void SettingsUi::videoDisplayCallback(Option &opt){
	
}

void SettingsUi::initAudioSource(){
	const QStringList sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const QStringList sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioSourceComboBox;
	connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(setAudioSource(int)));

	box->addItem("none", QVariant(""));

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_SRC)){
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}

	using namespace std::placeholders;
	settings->getOption("audio.source").addOnChangeCallback(
			std::bind(&SettingsUi::audioSourceCallback, this, _1)
			);
}

void SettingsUi::setAudioSource(int idx){
	std::string val = mainWin->audioSourceComboBox->itemData(idx).toString().toStdString();
	settings->getOption("audio.source").setValue(val);
}

void SettingsUi::audioSourceCallback(Option &opt){

}

void SettingsUi::initAudioPlayback(){
	const QStringList sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const QStringList sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioPlaybackComboBox;
	connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(setAudioPlayback(int)));

	box->addItem("none", QVariant(""));

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_PLAYBACK)){
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}

	using namespace std::placeholders;
	settings->getOption("audio.playback").addOnChangeCallback(
			std::bind(&SettingsUi::audioPlaybackCallback, this, _1)
			);
}

void SettingsUi::setAudioPlayback(int idx){
	std::string val = mainWin->audioPlaybackComboBox->itemData(idx).toString().toStdString();
	settings->getOption("audio.playback").setValue(val);
}

void SettingsUi::audioPlaybackCallback(Option &opt){

}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;


}

void SettingsUi::setAdvanced(bool enable){
	settings->getOption("advanced").setEnabled(enable);
}
