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
	using namespace std::placeholders;
	QObject::connect(mainWin->networkDestinationEdit, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.destination", _1));

	initVideoCompress();
	initVideoSource();
	initVideoDisplay();
	initAudioSource();
	initAudioPlayback();
	initAudioCompression();

	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));
	connect(mainWin->actionUse_hw_acceleration, SIGNAL(triggered(bool)), this, SLOT(setHwAccel(bool)));

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
		if(!whitelist.empty() && !isAdvancedMode() && !vecContains(whitelist, i))
			continue;

		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
}


void SettingsUi::initVideoCompress(){
	using namespace std::placeholders;
	QComboBox *box = mainWin->videoCompressionComboBox;
	//connect(box, SIGNAL(activated(int)), this, SLOT(setVideoCompression(int)));
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, &SettingsUi::setVideoCompression);
	connect(mainWin->videoBitrateEdit, SIGNAL(textEdited(const QString&)), this, SLOT(setVideoBitrate(const QString &)));
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
			std::bind(&SettingsUi::videoCompressionCallback, this, _1)
			);
}

void SettingsUi::initVideoSource(){
	using namespace std::placeholders;
	QComboBox *box = mainWin->videoSourceComboBox;
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, box, "video.source", _1));
	connect(mainWin->videoModeComboBox, SIGNAL(activated(int)), this, SLOT(setVideoSourceMode(int)));
	const std::vector<std::string> whiteList = {
		"testcard",
		"screen",
		"decklink",
		"aja",
		"dvs"
	};

	populateComboBox(box, VIDEO_SRC, whiteList);

	settings->getOption("video.source").addOnChangeCallback(
			std::bind(&SettingsUi::videoSourceCallback, this, _1, _2)
			);

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

	SettingItem item;
	item.name = "Default";
	item.opts.push_back({"video.source.testcard.width", ""});
	item.opts.push_back({"video.source.testcard.height", ""});
	item.opts.push_back({"video.source.testcard.fps", ""});
	item.opts.push_back({"video.source.testcard.format", ""});
	box->addItem(QString::fromStdString(item.name),
			QVariant::fromValue(item));

	for(const auto &res : resolutions){
		for(const auto &rate : rates){
			item.opts.clear();
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
	static int const rates[] = {24, 30, 60};

	SettingItem item;
	item.opts.push_back({"video.source.screen.fps", ""});

	box->addItem("Default",
			QVariant::fromValue(item));

	for(const auto &rate : rates){
		item.opts.clear();
		item.name = std::to_string(rate) + " fps";
		item.opts.push_back({"video.source.screen.fps", std::to_string(rate)});

		box->addItem(QString::fromStdString(item.name),
				QVariant::fromValue(item));
	}
}

void SettingsUi::initVideoDisplay(){
	using namespace std::placeholders;
	QComboBox *box = mainWin->videoDisplayComboBox;
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, box, "video.display", _1));
	const std::vector<std::string> whiteList = {
		"gl",
		"sdl",
		"decklink",
		"aja",
		"dvs"
	};

	populateComboBox(box, VIDEO_DISPLAY, whiteList);

	settings->getOption("video.display").addOnChangeCallback(
			std::bind(&SettingsUi::videoDisplayCallback, this, _1)
			);

}

void SettingsUi::initAudioSource(){
	const QStringList sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const QStringList sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioSourceComboBox;

	using namespace std::placeholders;
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, box, "audio.source", _1));
	QObject::connect(mainWin->audioChannelsSpinBox, QOverload<const QString &>::of(&QSpinBox::valueChanged),
			this, std::bind(&SettingsUi::setString, this, "audio.source.channels", _1));

	box->addItem("none", QVariant(""));

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_SRC)){
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}

	settings->getOption("audio.source").addOnChangeCallback(
			std::bind(&SettingsUi::audioSourceCallback, this, _1)
			);
}

void SettingsUi::initAudioPlayback(){
	const QStringList sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const QStringList sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioPlaybackComboBox;
	using namespace std::placeholders;
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, box, "audio.playback", _1));

	box->addItem("none", QVariant(""));

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_PLAYBACK)){
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}

	settings->getOption("audio.playback").addOnChangeCallback(
			std::bind(&SettingsUi::audioPlaybackCallback, this, _1)
			);
}

void SettingsUi::initAudioCompression(){
	QComboBox *box = mainWin->audioCompressionComboBox;
	using namespace std::placeholders;
	QObject::connect(box, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, box, "audio.compress", _1));
	QObject::connect(mainWin->audioBitrateEdit, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "audio.compress.bitrate", _1));

	populateComboBox(box, AUDIO_COMPRESS);

	settings->getOption("audio.compress").addOnChangeCallback(
			std::bind(&SettingsUi::audioCompressionCallback, this, _1, _2)
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

void SettingsUi::videoSourceCallback(Option &opt, bool suboption){
	if(suboption)
		return;

	mainWin->videoModeComboBox->clear();
	if(opt.getValue() == "testcard"){
		initTestcardModes(mainWin->videoModeComboBox);
	} else if (opt.getValue() == "screen"){
		initScreenModes(mainWin->videoModeComboBox);
	}
}

void SettingsUi::videoDisplayCallback(Option &opt){
	
}

void SettingsUi::audioSourceCallback(Option &opt){

}

void SettingsUi::audioPlaybackCallback(Option &opt){

}

void SettingsUi::audioCompressionCallback(Option &opt, bool suboption){
	if(suboption)
		return;

	static const std::vector<std::string> losslessCodecs = {
		"FLAC",
		"u-law",
		"A-law",
		"PCM"
	};

	bool enableBitrate = !vecContains(losslessCodecs, opt.getValue());

	settings->getOption("audio.compress.bitrate").setEnabled(enableBitrate);
	mainWin->audioBitrateEdit->setEnabled(enableBitrate);
	mainWin->audioBitrateLabel->setEnabled(enableBitrate);
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
	emit changed();
}

void SettingsUi::setVideoBitrate(const QString &str){
	std::string opt = getBitrateOpt(settings);

	settings->getOption(opt).setValue(str.toStdString());
	emit changed();
}

void SettingsUi::setVideoSourceMode(int idx){
	const SettingItem &i = mainWin->videoModeComboBox->itemData(idx).value<SettingItem>();

	for(const auto &opt : i.opts){
		settings->getOption(opt.opt).setValue(opt.val);
	}
	emit changed();
}

void SettingsUi::setComboBox(QComboBox *box, const std::string &opt, int idx){
	std::string val = box->itemData(idx).toString().toStdString();
	settings->getOption(opt).setValue(val);
	emit changed();
}

void SettingsUi::setString(const std::string &opt, const QString &str){
	settings->getOption(opt).setValue(str.toStdString());
	emit changed();
}

void SettingsUi::setAdvanced(bool enable){
	settings->getOption("advanced").setEnabled(enable);
	emit changed();
}

void SettingsUi::setHwAccel(bool b){
	settings->getOption("decode.hwaccel").setEnabled(b);
	emit changed();
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;


}
