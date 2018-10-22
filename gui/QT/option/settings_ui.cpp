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

static void setItem(QComboBox *box, const QVariant &data){
	if(!data.isValid())
		return;

	int idx = box->findData(data);

	if(idx != -1)
		box->setCurrentIndex(idx);
}

void SettingsUi::init(Settings *settings, AvailableSettings *availableSettings){
	this->settings = settings;
	this->availableSettings = availableSettings;
}

void SettingsUi::initMainWin(Ui::UltragridWindow *ui){
	mainWin = ui;

	connectSignals();

	refreshAll();
 
	addCallbacks();
	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));

}

void SettingsUi::refreshAll(){
	refreshVideoCompress();
	refreshVideoSource();
	refreshVideoDisplay();
	refreshAudioSource();
	refreshAudioPlayback();
	refreshAudioCompression();
}

void SettingsUi::connectSignals(){
	using namespace std::placeholders;
	//Window
	connect(mainWin->actionAdvanced, &QAction::triggered,
			this, std::bind(&SettingsUi::setBool, this, "advanced", _1));
	connect(mainWin->networkDestinationEdit, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.destination", _1));
	connect(mainWin->fECCheckBox, &QCheckBox::clicked,
			this, std::bind(&SettingsUi::setBool, this, "network.fec", _1));
	connect(mainWin->actionUse_hw_acceleration, &QAction::triggered,
			this, std::bind(&SettingsUi::setBool, this, "decode.hwaccel", _1));
	connect(mainWin->actionRefresh, &QAction::triggered,
			this, &SettingsUi::refreshAll);

	//Video
	connect(mainWin->videoCompressionComboBox,
			QOverload<int>::of(&QComboBox::activated),
			this, &SettingsUi::setVideoCompression);
	connect(mainWin->videoBitrateEdit, SIGNAL(textEdited(const QString&)),
			this, SLOT(setVideoBitrate(const QString &)));

	connect(mainWin->videoSourceComboBox, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, mainWin->videoSourceComboBox, "video.source", _1));
	connect(mainWin->videoModeComboBox, SIGNAL(activated(int)),
			this, SLOT(setVideoSourceMode(int)));

	connect(mainWin->videoDisplayComboBox, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, mainWin->videoDisplayComboBox, "video.display", _1));

	//Audio
	connect(mainWin->audioSourceComboBox, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioSourceComboBox, "audio.source", _1));
	connect(mainWin->audioChannelsSpinBox, QOverload<const QString &>::of(&QSpinBox::valueChanged),
			this, std::bind(&SettingsUi::setString, this, "audio.source.channels", _1));

	connect(mainWin->audioPlaybackComboBox, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioPlaybackComboBox, "audio.playback", _1));

	connect(mainWin->audioCompressionComboBox, QOverload<int>::of(&QComboBox::activated),
			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioCompressionComboBox, "audio.compress", _1));
	connect(mainWin->audioBitrateEdit, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "audio.compress.bitrate", _1));
}

void SettingsUi::addCallbacks(){
	using namespace std::placeholders;

#define CALLBACK(opt, fun) { opt, std::bind(fun, this, _1, _2) }
	const static struct{
		const char *opt;
		std::function<void(Option &, bool)> callback;
	} callbacks[] = {
		CALLBACK("video.compress", &SettingsUi::videoCompressionCallback),
		CALLBACK("video.source", &SettingsUi::videoSourceCallback),
		{"video.source", std::bind(&SettingsUi::refreshAudioSource, this)},
		CALLBACK("video.display", &SettingsUi::videoDisplayCallback),
		{"video.display", std::bind(&SettingsUi::refreshAudioPlayback, this)},
		CALLBACK("audio.source", &SettingsUi::audioSourceCallback),
		CALLBACK("audio.playback", &SettingsUi::audioPlaybackCallback),
		CALLBACK("audio.compress", &SettingsUi::audioCompressionCallback),
		CALLBACK("network.fec", &SettingsUi::fecCallback),
		{"advanced", std::bind(&SettingsUi::refreshAll, this)},
	};
#undef CALLBACK

	for(const auto & call : callbacks){
		settings->getOption(call.opt).addOnChangeCallback(call.callback);
	}
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
		if(!whitelist.empty() && !settings->isAdvancedMode() && !vecContains(whitelist, i))
			continue;

		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
}


void SettingsUi::refreshVideoCompress(){
	QComboBox *box = mainWin->videoCompressionComboBox;
	QVariant prevData = box->currentData();
	box->clear();
	for(const auto& item : videoCodecs){
		box->addItem(QString(item.displayName), QVariant::fromValue(item));
	}
	if(settings->isAdvancedMode()){
		populateComboBox(box, VIDEO_COMPRESS);
	}
	setItem(box, prevData);
}

static std::string getDeviceStr(const std::string &type, const std::string &id){
	return type + ":device=" + id;
}

static void addWebcams(QComboBox *box, AvailableSettings *avail){
	std::vector<Webcam> cams = avail->getWebcams();

	for(const auto &cam : cams){
		std::string dev = cam.type;

		box->addItem(QString::fromStdString(cam.name),
				QVariant(QString::fromStdString(getDeviceStr(cam.type, cam.id))));
	}
}

void SettingsUi::refreshVideoSource(){
	const std::vector<std::string> whiteList = {
		"testcard",
		"screen",
		"decklink",
		"aja",
		"dvs"
	};

	QComboBox *box = mainWin->videoSourceComboBox;
	QVariant prevData = box->currentData();
	box->clear();
	populateComboBox(box, VIDEO_SRC, whiteList);
	addWebcams(box, availableSettings);
	setItem(box, prevData);
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

void SettingsUi::refreshVideoDisplay(){
	const std::vector<std::string> whiteList = {
		"gl",
		"sdl",
		"decklink",
		"aja",
		"dvs"
	};

	QComboBox *box = mainWin->videoDisplayComboBox;
	QVariant prevData = box->currentData();
	box->clear();
	populateComboBox(box, VIDEO_DISPLAY, whiteList);
	setItem(box, prevData);
}

void SettingsUi::refreshAudioSource(){
	const std::vector<std::string> sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const std::vector<std::string> sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioSourceComboBox;
	QVariant prevData = box->currentData();
	box->clear();

	box->addItem("none", QVariant(""));

	std::string vid = settings->getOption("video.source").getValue();

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_SRC)){
		if(!settings->isAdvancedMode() && vecContains(sdiAudio, i) && !vecContains(sdiAudioCards, vid))
			continue;
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
	setItem(box, prevData);
}

void SettingsUi::refreshAudioPlayback(){
	const std::vector<std::string> sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
	const std::vector<std::string> sdiAudio = {"analog", "AESEBU", "embedded"};

	QComboBox *box = mainWin->audioPlaybackComboBox;
	QVariant prevData = box->currentData();
	box->clear();
	box->addItem("none", QVariant(""));

	std::string vid = settings->getOption("video.display").getValue();

	for(const auto &i : availableSettings->getAvailableSettings(AUDIO_PLAYBACK)){
		if(!settings->isAdvancedMode() && vecContains(sdiAudio, i) && !vecContains(sdiAudioCards, vid))
			continue;
		box->addItem(QString::fromStdString(i),
				QVariant(QString::fromStdString(i)));
	}
	setItem(box, prevData);
}

void SettingsUi::refreshAudioCompression(){
	QComboBox *box = mainWin->audioCompressionComboBox;
	QVariant prevData = box->currentData();
	box->clear();
	populateComboBox(box, AUDIO_COMPRESS);
	setItem(box, prevData);
}

void SettingsUi::videoCompressionCallback(Option &opt, bool){
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
	} else {
		mainWin->videoModeComboBox->addItem("Default",
				QVariant::fromValue(SettingItem()));
	}
}

void SettingsUi::videoDisplayCallback(Option &opt, bool){
	
}

void SettingsUi::audioSourceCallback(Option &opt, bool){

}

void SettingsUi::audioPlaybackCallback(Option &opt, bool){

}

void SettingsUi::fecCallback(Option &opt, bool){
	mainWin->fECCheckBox->setChecked(opt.isEnabled());
	settingsWin->fecGroupBox->setChecked(opt.isEnabled());
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

void SettingsUi::setBool(const std::string &opt, bool b){
	settings->getOption(opt).setEnabled(b);
	emit changed();
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;

	using namespace std::placeholders;
	connect(ui->basePort, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.port", _1));
	connect(ui->controlPort, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.control_port", _1));
	connect(ui->fecGroupBox, &QGroupBox::clicked,
			this, std::bind(&SettingsUi::setBool, this, "network.fec", _1));
	connect(ui->fecAutoCheck, &QCheckBox::clicked,
			this, std::bind(&SettingsUi::setBool, this, "network.fec.auto", _1));
}
