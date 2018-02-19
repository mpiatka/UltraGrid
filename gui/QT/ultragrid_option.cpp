#include <QStringList>
#include <QProcess>
#include <QComboBox>
#include <QLineEdit>
#include <QLabel>

#include "ultragrid_option.hpp"
#include "v4l2.hpp"

#define EXIT_FAIL_DISPLAY 4
#define EXIT_FAIL_CAPTURE 5

bool testCaptureDisplay(const QString &executable, const QString &cmd){
	QProcess process;

	QString command = executable + " " + cmd;
	process.start(command);
	process.waitForFinished(5000); // wait 5 seconds max in case it hangs

	if(process.exitStatus() != QProcess::NormalExit){
		//Ultragrid hanged or crashed
		return false;
	}

	int exitCode = process.exitCode();

	return exitCode != EXIT_FAIL_DISPLAY && exitCode != EXIT_FAIL_CAPTURE;
}

void UltragridOption::setAdvanced(bool advanced){
	this->advanced = advanced;
	update();
}

ComboBoxOption::ComboBoxOption(QComboBox *box,
		const QString& ultragridExecutable,
		QString opt):
	box(box)
{
	connect(box, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
}

QString ComboBoxOption::getLaunchParam(){
	QString param;

	if(box->currentText() != "none"){
		param += opt;
		param += " ";
		param += box->currentData().toString();
		param += getExtraParams();
		param += " ";
	}

	return param;
}

void ComboBoxOption::queryAvailOpts(){
	QVariant prevData = box->currentData();
	resetComboBox(box);
	QStringList opts = getAvailOpts();
	for(const auto& i : opts){
		if(advanced || filter(i)){
			box->addItem(i, QVariant(i));
		}
	}

	queryExtraOpts();

	setItem(box, prevData);
}

void ComboBoxOption::resetComboBox(QComboBox *box){
	box->clear();
	box->addItem(QString("none"));
}


QStringList ComboBoxOption::getAvailOpts() {
	QStringList out;

	QProcess process;

	QString command = ultragridExecutable;

	command += " ";
	command += opt;
	command += " help";

	process.start(command);

	process.waitForFinished();
	QByteArray output = process.readAllStandardOutput();
	QList<QByteArray> lines = output.split('\n');

	foreach ( const QByteArray &line, lines ) {
		if(line.size() > 0 && QChar(line[0]).isSpace()) {
			QString opt = QString(line).trimmed();
			if(opt != "none"
					&& !opt.startsWith("--")
					&& !opt.contains("unavailable"))
				out.append(QString(line).trimmed());
		}
	}

	return out;
}


void ComboBoxOption::setItem(const QVariant &data){
	if(!data.isValid())
		return;

	int idx = box->findData(data);

	if(idx != -1)
		box->setCurrentIndex(idx);
}

void UltragridOption::setItem(const QString &text){
	int idx = box->findText(text);

	if(idx != -1)
		box->setCurrentIndex(idx);
}


SourceOption::SourceOption(Ui::UltragridWindow *ui,
		const QString& ultragridExecutable) :
	ComboBoxOption(ui->videoSourceComboBox, ultragridExecutable, QString("-t")),
	ui(ui)
{
	connect(ui->videoSourceComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(srcChanged()));
	connect(ui->videoModeComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
}

QString SourceOption::getExtraParams(){
	return ui->videoModeComboBox->currentData().toString();
}

bool SourceOption::filter(QString &item){
	const QStringList whiteList = {"testcard", "screen", "decklink", "aja", "dvs"};
	return whiteList.contains(item)
					&& testCaptureDisplay(ultragridExecutable,
							QString("-t ") + item + QString(":help"));
}

void SourceOption::queryExtraOpts(const QStringList &opts){
	if(opts.contains(QString("v4l2"))){
		std::vector<Camera> cams = getCameras();
		for(const auto& cam : cams){
			QString name = QString::fromStdString(cam.name);
			QString opt = QString::fromStdString("v4l2:device=" + cam.path);
			src->addItem(name, QVariant(opt));
		}
	}
}

void SourceOption::srcChanged(){
	QComboBox *src = ui->videoSourceComboBox;
	QComboBox *mode = ui->videoModeComboBox;

	mode->clear();
	mode->addItem(QString("Default"), QVariant(QString("")));

	if(src->currentText() == "testcard"){
		mode->addItem(QString("1280x720, 30 fps"), QVariant(QString(":1280:720:30:UYVY")));
		mode->addItem(QString("1280x720, 24 fps"), QVariant(QString(":1280:720:24:UYVY")));
		mode->addItem(QString("1280x720, 60 fps"), QVariant(QString(":1280:720:60:UYVY")));
		mode->addItem(QString("1920x1080, 30 fps"), QVariant(QString(":1920:1080:30:UYVY")));
		mode->addItem(QString("1920x1080, 24 fps"), QVariant(QString(":1920:1080:24:UYVY")));
		mode->addItem(QString("1920x1080, 60 fps"), QVariant(QString(":1920:1080:60:UYVY")));
		mode->addItem(QString("3840x2160, 30 fps"), QVariant(QString(":3840:2160:30:UYVY")));
		mode->addItem(QString("3840x2160, 24 fps"), QVariant(QString(":3840:2160:24:UYVY")));
		mode->addItem(QString("3840x2160, 60 fps"), QVariant(QString(":3840:2160:60:UYVY")));
	} else if(src->currentText() == "screen"){
		mode->addItem(QString("60 fps"), QVariant(QString(":fps=60")));
		mode->addItem(QString("30 fps"), QVariant(QString(":fps=30")));
		mode->addItem(QString("24 fps"), QVariant(QString(":fps=24")));
	} else if(src->currentData().toString().contains("v4l2")){
		std::string path = src->currentData().toString().split("device=")[1].toStdString();
		std::vector<Mode> modes = getModes(path);

		for(const Mode& m : modes){
			QString name;
			QString param = ":size=";

			name += QString::number(m.width);
			name += "x";
			name += QString::number(m.height);
			name += ", ";

			name += QString::fromStdString(m.codec);
			name += ", ";

			float fps = (float) m.tpf_denominator / m.tpf_numerator;
			name += QString::number(fps);
			name += " fps";

			param += QString::number(m.width);
			param += "x";
			param += QString::number(m.height);

			param += ":codec=";
			param += QString::fromStdString(m.codec);

			param += ":tpf=";
			param += QString::number(m.tpf_numerator);
			param += "/";
			param += QString::number(m.tpf_denominator);

			mode->addItem(name, param);
		}
	}

	emit changed();
}

DisplayOption::DisplayOption(Ui::UltragridWindow *ui,
		const QString& ultragridExecutable):
	ComboBoxOption(ui->videoDisplayComboBox, ultragridExecutable, QString("-d")),
	ui(ui)
{
	connect(ui->previewCheckBox, SIGNAL(toggled(bool)), this, SLOT(enablePreview(bool)));
	preview = ui->previewCheckBox->isChecked();
}

QString DisplayOption::getLaunchParam(){
	QComboBox *disp = ui->videoDisplayComboBox;
	QString param;

	if(disp->currentText() != "none"){
		param += "-d ";
		if(preview){
			param += "multiplier:";
		}
		param += disp->currentData().toString();
		if(preview){
			param += "#preview";
		}
		param += " ";
	} else if(preview){
		param += "-d preview ";
	}

	return param;
}

bool DisplayOption::filter(QString &item){
	const QStringList whiteList = {"gl", "sdl", "decklink", "aja", "dvs"};

	return whiteList.contains(i)
					&& testCaptureDisplay(ultragridExecutable,
							QString("-d ") + i + QString(":help"));
}

void DisplayOption::enablePreview(bool enable){
	preview = enable;
	emit changed();
}

VideoCompressOption::VideoCompressOption(Ui::UltragridWindow *ui,
		const QString& ultragridExecutable) :
	UltragridOption(ultragridExecutable, QString("-c")),
	ui(ui)
{
	connect(ui->videoCompressionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(compChanged()));
	connect(ui->videoBitrateEdit, SIGNAL(textChanged(const QString&)), this, SIGNAL(changed()));
	connect(ui->videoCompressionComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
}

QString VideoCompressOption::getLaunchParam(){
	QComboBox *comp = ui->videoCompressionComboBox;
	QLineEdit *bitrate = ui->videoBitrateEdit;
	QString param;

	if(comp->currentText() != "none"){
		param += "-c ";
		param += comp->currentData().toString();
		if(!bitrate->text().isEmpty()){
			if(comp->currentText() == "JPEG"){
				param += ":" + bitrate->text();
			} else {
				param += ":bitrate=" + bitrate->text();
			}
		}
		param += " ";
	}

	return param;
}

void VideoCompressOption::queryAvailOpts(){
	QComboBox *comp = ui->videoCompressionComboBox;
	
	QVariant prevData = comp->currentData();

	resetComboBox(comp);

	comp->addItem(QString("H.264"), QVariant(QString("libavcodec:codec=H.264")));
	comp->addItem(QString("H.265"), QVariant(QString("libavcodec:codec=H.265")));
	comp->addItem(QString("MJPEG"), QVariant(QString("libavcodec:codec=MJPEG")));
	comp->addItem(QString("VP8"), QVariant(QString("libavcodec:codec=VP8")));
	comp->addItem(QString("JPEG"), QVariant(QString("jpeg")));

	setItem(comp, prevData);
}

void VideoCompressOption::compChanged(){
	QComboBox *comp = ui->audioCompressionComboBox;
	QLabel *label = ui->audioBitrateLabel;

	if(comp->currentText() == "JPEG"){
		label->setText(QString("Jpeg quality"));
	} else {
		label->setText(QString("Bitrate"));
	}
}

const QStringList sdiAudioCards = {"decklink", "aja", "dvs", "deltacast"};
const QStringList sdiAudio = {"analog", "AESEBU", "embedded"};

AudioSourceOption::AudioSourceOption(Ui::UltragridWindow *ui,
		const SourceOption *videoSource,
		const QString& ultragridExecutable) :
	UltragridOption(ultragridExecutable, QString("-s")),
	ui(ui),
	videoSource(videoSource)
{
	connect(ui->audioSourceComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
	connect(ui->audioChannelsSpinBox, SIGNAL(valueChanged(const QString&)), this, SIGNAL(changed()));
	connect(videoSource, SIGNAL(changed()), this, SLOT(update()));
}

void AudioSourceOption::update(){
	queryAvailOpts();
	emit changed();
}

QString AudioSourceOption::getLaunchParam(){
	QComboBox *src = ui->audioSourceComboBox;
	QSpinBox *channels = ui->audioChannelsSpinBox;
	QString param;

	if(src->currentText() != "none"){
		param += opt + " ";
		param += src->currentText();
		if(channels->value() != 1){
			param += " --audio-capture-format channels=" + QString::number(channels->value());
		}
		param += " ";
	}

	return param;
}

void AudioSourceOption::queryAvailOpts(){
	QComboBox *box = ui->audioSourceComboBox;

	QString prevText = box->currentText();
	resetComboBox(box);
	QString helpCommand = opt + " help";
	QStringList opts = getAvailOpts(ultragridExecutable, helpCommand);
	//box->addItems(opts);
	for(const auto& i : opts){
		if(!sdiAudio.contains(i)
				|| sdiAudioCards.contains(videoSource->getCard())
					|| advanced){
			box->addItem(i);
		}
	}

	setItem(box, prevText);
}

AudioPlaybackOption::AudioPlaybackOption(Ui::UltragridWindow *ui,
		const DisplayOption *videoDisplay,
		const QString& ultragridExecutable) :
	UltragridOption(ultragridExecutable, QString("-r")),
	ui(ui),
	videoDisplay(videoDisplay)
{
	connect(ui->audioPlaybackComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
	connect(videoDisplay, SIGNAL(changed()), this, SLOT(update()));
}

void AudioPlaybackOption::update(){
	queryAvailOpts();
	emit changed();
}

QString AudioPlaybackOption::getLaunchParam(){
	QComboBox *src = ui->audioPlaybackComboBox;
	QString param;

	if(src->currentText() != "none"){
		param += opt + " ";
		param += src->currentText();
		param += " ";
	}

	return param;
}

void AudioPlaybackOption::queryAvailOpts(){
	QComboBox *box = ui->audioPlaybackComboBox;

	QString prevText = box->currentText();
	resetComboBox(box);
	QString helpCommand = opt + " help";
	QStringList opts = getAvailOpts(ultragridExecutable, helpCommand);
	//box->addItems(opts);
	for(const auto& i : opts){
		if(!sdiAudio.contains(i)
				|| sdiAudioCards.contains(videoDisplay->getCard())
					|| advanced){
			box->addItem(i);
		}
	}

	setItem(box, prevText);
}

AudioCompressOption::AudioCompressOption(Ui::UltragridWindow *ui,
		const QString& ultragridExecutable) :
	UltragridOption(ultragridExecutable, QString("--audio-codec")),
	ui(ui)
{
	connect(ui->audioCompressionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(compChanged()));
	connect(ui->audioBitrateEdit, SIGNAL(textChanged(const QString&)), this, SIGNAL(changed()));
	connect(ui->audioCompressionComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
}

QString AudioCompressOption::getLaunchParam(){
	QComboBox *comp = ui->audioCompressionComboBox;
	QLineEdit *bitrate = ui->audioBitrateEdit;
	QString param;

	if(comp->currentText() != "none"){
		param += opt + " ";
		param += comp->currentText();
		if(!bitrate->text().isEmpty()){
			param += ":bitrate=" + bitrate->text();
		}
		param += " ";
	}

	return param;
}

void AudioCompressOption::queryAvailOpts(){
	QComboBox *box = ui->audioCompressionComboBox;

	QString prevText = box->currentText();
	resetComboBox(box);
	QString helpCommand = opt + " help";
	QStringList opts = getAvailOpts(ultragridExecutable, helpCommand);
	box->addItems(opts);

	setItem(box, prevText);
}

void AudioCompressOption::compChanged(){

}
FecOption::FecOption(Ui::UltragridWindow *ui) : 
	UltragridOption("", "-f"),
	ui(ui)
{
	connect(ui->fecNoneRadio, SIGNAL(toggled(bool)), this, SLOT(update()));
	connect(ui->fecMultRadio, SIGNAL(toggled(bool)), this, SLOT(update()));
	connect(ui->fecLdgmRadio, SIGNAL(toggled(bool)), this, SLOT(update()));
	connect(ui->fecRsRadio, SIGNAL(toggled(bool)), this, SLOT(update()));

	connect(ui->multSpin, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->ldgmMaxLoss, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->ldgmC, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->ldgmK, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->ldgmM, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->rsK, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));
	connect(ui->rsN, SIGNAL(valueChanged(int)), this, SIGNAL(changed()));

	connect(ui->ldgmSimpCpuRadio, SIGNAL(toggled(bool)), this, SIGNAL(changed()));
	connect(ui->ldgmSimpGpuRadio, SIGNAL(toggled(bool)), this, SIGNAL(changed()));
	connect(ui->ldgmAdvCpuRadio, SIGNAL(toggled(bool)), this, SIGNAL(changed()));
	connect(ui->ldgmAdvGpuRadio, SIGNAL(toggled(bool)), this, SIGNAL(changed()));
}

QString FecOption::getLaunchParam(){
	QString param = "";

	if(!ui->fecNoneRadio->isChecked()){
		param += opt;

		if(ui->fecMultRadio->isChecked()){
			param += " mult:" + ui->multSpin->cleanText();
		} else if(ui->fecLdgmRadio->isChecked()){
			if(advanced){
				param += " ldgm:" + ui->ldgmK->cleanText() + ":";
				param += ui->ldgmM->cleanText() + ":";
				param += ui->ldgmC->cleanText();
				if(ui->ldgmAdvCpuRadio->isChecked()){
					param += " --param ldgm-device=CPU";
				} else if(ui->ldgmAdvGpuRadio->isChecked()){
					param += " --param ldgm-device=GPU";
				}
			} else {
				param += " ldgm:" + ui->ldgmMaxLoss->cleanText() + "%";
				if(ui->ldgmSimpCpuRadio->isChecked()){
					param += " --param ldgm-device=CPU";
				} else if(ui->ldgmSimpGpuRadio->isChecked()){
					param += " --param ldgm-device=GPU";
				}
			}
		} else if(ui->fecRsRadio->isChecked()){
			param += " rs:" + ui->rsK->cleanText();
			param += ":" + ui->rsN->cleanText();
		}
		param += " ";
	}

	return param;
}

void FecOption::update(){
	if(ui->fecMultRadio->isChecked()){
		ui->stackedWidget->setCurrentIndex(0);
	} else if(ui->fecLdgmRadio->isChecked()){
		if(advanced)
			ui->stackedWidget->setCurrentIndex(2);
		else
			ui->stackedWidget->setCurrentIndex(1);
	} else if(ui->fecRsRadio->isChecked()){
		ui->stackedWidget->setCurrentIndex(3);
	}

	emit changed();
}
