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

	uiControls.emplace_back(new CheckboxUi(ui->fECCheckBox, settings, "network.fec"));
	uiControls.emplace_back(new ActionCheckableUi(ui->actionAdvanced, settings, "advanced"));
	uiControls.emplace_back(new ActionCheckableUi(ui->actionUse_hw_acceleration,
				settings,
				"decode.hwaccel"));

	uiControls.emplace_back(
			new ComboBoxUi(ui->videoSourceComboBox,
				settings,
				"video.source",
				std::bind(getVideoSrc, availableSettings))
			);

	LineEditUi *videoBitrate = new LineEditUi(ui->videoBitrateEdit,
			settings,
			"");
	uiControls.emplace_back(videoBitrate);
	using namespace std::placeholders;
	settings->getOption("video.compress").addOnChangeCallback(
			std::bind(videoCompressBitrateCallback, videoBitrate, _1, _2)
			);

	uiControls.emplace_back(
			new ComboBoxUi(ui->videoCompressionComboBox,
				settings,
				"video.compress",
				std::bind(getVideoCompress, availableSettings))
			);

	uiControls.emplace_back(
			new ComboBoxUi(ui->videoDisplayComboBox,
				settings,
				"video.display",
				std::bind(getVideoDisplay, availableSettings))
			);

	ComboBoxUi *videoModeCombo = new ComboBoxUi(ui->videoModeComboBox,
				settings,
				"",
				std::bind(getVideoModes, availableSettings));
	videoModeCombo->registerCallback("video.source");
	uiControls.emplace_back(videoModeCombo);

	uiControls.emplace_back(
			new ComboBoxUi(ui->audioCompressionComboBox,
				settings,
				"audio.compress",
				std::bind(getAudioCompress, availableSettings))
			);

	ComboBoxUi *audioSrc = new ComboBoxUi(ui->audioSourceComboBox,
			settings,
			"audio.source",
			std::bind(getAudioSrc, availableSettings));
	audioSrc->registerCallback("video.source");
	uiControls.emplace_back(audioSrc);

	ComboBoxUi *audioPlayback = new ComboBoxUi(ui->audioPlaybackComboBox,
			settings,
			"audio.playback",
			std::bind(getAudioPlayback, availableSettings));
	audioPlayback->registerCallback("video.display");
	uiControls.emplace_back(audioPlayback);

	for(auto &i : uiControls){
		connect(i.get(), &WidgetUi::changed, this, &SettingsUi::changed);
	}
}

void SettingsUi::refreshAll(){
//	refreshVideoCompress();
//	refreshVideoSource();
//	refreshVideoDisplay();
//	refreshAudioSource();
//	refreshAudioPlayback();
	//refreshAudioCompression();
	
	for(auto &i : uiControls){
		i->refresh();
	}
}

void SettingsUi::connectSignals(){
	using namespace std::placeholders;
	//Window
//	connect(mainWin->actionAdvanced, &QAction::triggered,
//			this, std::bind(&SettingsUi::setBool, this, "advanced", _1));
	connect(mainWin->networkDestinationEdit, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.destination", _1));
//	connect(mainWin->fECCheckBox, &QCheckBox::clicked,
//			this, std::bind(&SettingsUi::setBool, this, "network.fec", _1));
//	connect(mainWin->actionUse_hw_acceleration, &QAction::triggered,
//			this, std::bind(&SettingsUi::setBool, this, "decode.hwaccel", _1));
	connect(mainWin->actionRefresh, &QAction::triggered,
			this, &SettingsUi::refreshAll);

	//Video
//	connect(mainWin->videoCompressionComboBox,
//			QOverload<int>::of(&QComboBox::activated),
//			this, &SettingsUi::setVideoCompression);
//	connect(mainWin->videoBitrateEdit, SIGNAL(textEdited(const QString&)),
//			this, SLOT(setVideoBitrate(const QString &)));

//	connect(mainWin->videoSourceComboBox, QOverload<int>::of(&QComboBox::activated),
//			this, std::bind(&SettingsUi::setComboBox, this, mainWin->videoSourceComboBox, "video.source", _1));
//	connect(mainWin->videoModeComboBox, SIGNAL(activated(int)),
//			this, SLOT(setVideoSourceMode(int)));

//	connect(mainWin->videoDisplayComboBox, QOverload<int>::of(&QComboBox::activated),
//			this, std::bind(&SettingsUi::setComboBox, this, mainWin->videoDisplayComboBox, "video.display", _1));

	//Audio
//	connect(mainWin->audioSourceComboBox, QOverload<int>::of(&QComboBox::activated),
//			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioSourceComboBox, "audio.source", _1));
	connect(mainWin->audioChannelsSpinBox, QOverload<const QString &>::of(&QSpinBox::valueChanged),
			this, std::bind(&SettingsUi::setString, this, "audio.source.channels", _1));

//	connect(mainWin->audioPlaybackComboBox, QOverload<int>::of(&QComboBox::activated),
//			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioPlaybackComboBox, "audio.playback", _1));

//	connect(mainWin->audioCompressionComboBox, QOverload<int>::of(&QComboBox::activated),
//			this, std::bind(&SettingsUi::setComboBox, this, mainWin->audioCompressionComboBox, "audio.compress", _1));
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
		CALLBACK("video.compress", &SettingsUi::jpegLabelCallback),
//		CALLBACK("video.source", &SettingsUi::videoSourceCallback),
		//{"video.source", std::bind(&SettingsUi::refreshAudioSource, this)},
//		CALLBACK("video.display", &SettingsUi::videoDisplayCallback),
//		{"video.display", std::bind(&SettingsUi::refreshAudioPlayback, this)},
		//CALLBACK("audio.source", &SettingsUi::audioSourceCallback),
//		CALLBACK("audio.playback", &SettingsUi::audioPlaybackCallback),
		//CALLBACK("audio.compress", &SettingsUi::audioCompressionCallback),
		{"audio.compress", std::bind(audioCompressionCallback, mainWin, _1, _2)},
		//CALLBACK("network.fec", &SettingsUi::fecCallback),
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

void SettingsUi::jpegLabelCallback(Option &opt, bool suboption){
	if(suboption)
		return;

	mainWin->videoBitrateLabel->setEnabled(true);
	mainWin->videoBitrateEdit->setEnabled(true);

	const std::string &val = opt.getValue();
	if(val == "jpeg"){
		mainWin->videoBitrateLabel->setText(QString("Jpeg quality"));
	} else {
		mainWin->videoBitrateLabel->setText(QString("Bitrate"));
		if(val == ""){
			mainWin->videoBitrateLabel->setEnabled(false);
			mainWin->videoBitrateEdit->setEnabled(false);
		}
	}
}

#if 0
static void initWebcamModes(QComboBox *box, const std::vector<Mode> &modes){
	SettingItem item;
	item.opts.push_back({"video.source.v4l2.conf", ""});

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
#endif

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
	settings->getOption(opt).setValue(b ? "t" : "f");
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
