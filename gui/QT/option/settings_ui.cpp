#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include "settings_ui.hpp"

void SettingsUi::init(Settings *settings, AvailableSettings *availableSettings){
	this->settings = settings;
	this->availableSettings = availableSettings;
}

void SettingsUi::addControl(WidgetUi *widget){
	uiControls.emplace_back(widget);

	connect(widget, &WidgetUi::changed, this, &SettingsUi::changed);
}

void SettingsUi::initMainWin(Ui::UltragridWindow *ui){
	mainWin = ui;

	connectSignals();

	refreshAll();
 
	addCallbacks();
	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));

	addControl(new CheckboxUi(ui->fECCheckBox, settings, "network.fec"));
	addControl(new CheckboxUi(ui->previewCheckBox, settings, "preview"));
	addControl(
			new LineEditUi(ui->networkDestinationEdit, settings, "network.destination")
			);
	addControl(new ActionCheckableUi(ui->actionAdvanced, settings, "advanced"));
	addControl(new ActionCheckableUi(ui->actionUse_hw_acceleration,
				settings,
				"decode.hwaccel"));

	addControl(
			new ComboBoxUi(ui->videoSourceComboBox,
				settings,
				"video.source",
				std::bind(getVideoSrc, availableSettings))
			);

	LineEditUi *videoBitrate = new LineEditUi(ui->videoBitrateEdit,
			settings,
			"");
	addControl(videoBitrate);
	using namespace std::placeholders;
	settings->getOption("video.compress").addOnChangeCallback(
			std::bind(videoCompressBitrateCallback, videoBitrate, _1, _2)
			);

	addControl(
			new ComboBoxUi(ui->videoCompressionComboBox,
				settings,
				"video.compress",
				std::bind(getVideoCompress, availableSettings))
			);

	addControl(
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
	addControl(videoModeCombo);

	addControl(
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
	addControl(audioSrc);

	ComboBoxUi *audioPlayback = new ComboBoxUi(ui->audioPlaybackComboBox,
			settings,
			"audio.playback",
			std::bind(getAudioPlayback, availableSettings));
	audioPlayback->registerCallback("video.display");
	addControl(audioPlayback);

	addControl(new LineEditUi(ui->audioBitrateEdit,
				settings,
				"audio.compress.bitrate"));

	addControl(new SpinBoxUi(ui->audioChannelsSpinBox,
				settings,
				"audio.source.channels"));

	addControl(new LineEditUi(ui->portLineEdit,
				settings,
				"network.port"));
}

void SettingsUi::refreshAll(){
	for(auto &i : uiControls){
		i->refresh();
	}
}

void SettingsUi::connectSignals(){
//	connect(mainWin->actionRefresh, &QAction::triggered,
//			this, &SettingsUi::refreshAll);
}

void SettingsUi::addCallbacks(){
	using namespace std::placeholders;

#define CALLBACK(opt, fun) { opt, std::bind(fun, this, _1, _2) }
	const static struct{
		const char *opt;
		std::function<void(Option &, bool)> callback;
	} callbacks[] = {
		CALLBACK("video.compress", &SettingsUi::jpegLabelCallback),
		{"audio.compress", std::bind(audioCompressionCallback, mainWin, _1, _2)},
		{"advanced", std::bind(&SettingsUi::refreshAll, this)},
	};
#undef CALLBACK

	for(const auto & call : callbacks){
		settings->getOption(call.opt).addOnChangeCallback(call.callback);
	}
}

void SettingsUi::test(){
	printf("%s\n", settings->getLaunchParams().c_str());
}

#if 0
static void addWebcams(QComboBox *box, AvailableSettings *avail){
	std::vector<Webcam> cams = avail->getWebcams();

	for(const auto &cam : cams){
		std::string dev = cam.type;

		box->addItem(QString::fromStdString(cam.name),
				QVariant(QString::fromStdString(getDeviceStr(cam.type, cam.id))));
	}
}
 #endif

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

void SettingsUi::fecCallback(Option &opt, bool){
	mainWin->fECCheckBox->setChecked(opt.isEnabled());
	settingsWin->fecGroupBox->setChecked(opt.isEnabled());
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;

#if 0
	using namespace std::placeholders;
	connect(ui->basePort, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.port", _1));
	connect(ui->controlPort, &QLineEdit::textEdited,
			this, std::bind(&SettingsUi::setString, this, "network.control_port", _1));
	connect(ui->fecGroupBox, &QGroupBox::clicked,
			this, std::bind(&SettingsUi::setBool, this, "network.fec", _1));
	connect(ui->fecAutoCheck, &QCheckBox::clicked,
			this, std::bind(&SettingsUi::setBool, this, "network.fec.auto", _1));
#endif

	addControl(new LineEditUi(ui->basePort,
			settings,
			"network.port"));

	addControl(new LineEditUi(ui->controlPort,
			settings,
			"network.control_port"));

	addControl(new CheckboxUi(ui->fecAutoCheck,
			settings,
			"network.fec.auto"));

	addControl(new SpinBoxUi(ui->multSpin,
			settings,
			"network.fec.mult.factor"));

	addControl(new SpinBoxUi(ui->ldgmK,
			settings,
			"network.fec.ldgm.k"));

	addControl(new SpinBoxUi(ui->ldgmM,
			settings,
			"network.fec.ldgm.m"));

	addControl(new SpinBoxUi(ui->ldgmC,
			settings,
			"network.fec.ldgm.c"));

	addControl(new SpinBoxUi(ui->rsK,
			settings,
			"network.fec.rs.k"));

	addControl(new SpinBoxUi(ui->rsN,
			settings,
			"network.fec.rs.n"));

}
