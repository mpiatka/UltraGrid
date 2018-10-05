#include <QProcess>
#include <QString>
#include <cstring>
#include "available_settings.hpp"

#include <iostream>

static bool vectorContains(const std::vector<std::string> &v, const std::string & s){
	for(unsigned i = 0; i < v.size(); i++){
		if(v[i] == s)
			return true;
	}
	return false;
}

static QStringList getProcessOutput(const std::string& executable, const std::string& command){
	QProcess process;
	process.start((executable + command).c_str());
	process.waitForFinished();

	QString output = QString(process.readAllStandardOutput());
#ifdef WIN32
	QString lineSeparator = "\r\n";
#else
	QString lineSeparator = "\n";
#endif

	QStringList lines = output.split(lineSeparator);

	return lines;
}

static std::vector<std::string> getAvailOpts(const std::string &opt, const std::string &executable){
	std::vector<std::string> out;

	QProcess process;

	std::string command = executable;

	command += " ";
	command += opt;
	command += " help";

	process.start(command.c_str());

	process.waitForFinished();
	QByteArray output = process.readAllStandardOutput();
	QList<QByteArray> lines = output.split('\n');

	foreach ( const QByteArray &line, lines ) {
		if(line.size() > 0 && QChar(line[0]).isSpace()) {
			QString opt = QString(line).trimmed();
			if(opt != "none"
					&& !opt.startsWith("--")
					&& !opt.contains("unavailable"))
				out.push_back(QString(line).trimmed().toStdString());
		}
	}

	return out;
}

void AvailableSettings::queryAll(const std::string &executable){
	QStringList lines = getProcessOutput(executable, " --capabilities");

	queryCap(lines, VIDEO_SRC, "[cap][capture] ");
	queryCap(lines, VIDEO_DISPLAY, "[cap][display] ");
	queryCap(lines, VIDEO_COMPRESS, "[cap][compress] ");
	queryCap(lines, VIDEO_CAPTURE_FILTER, "[cap][capture_filter] ");
	queryCap(lines, AUDIO_SRC, "[cap][audio_cap] ");
	queryCap(lines, AUDIO_PLAYBACK, "[cap][audio_play] ");
	
	queryV4l2(executable);

	query(executable, AUDIO_COMPRESS);
}

void AvailableSettings::queryV4l2(const std::string &executable){
	std::string cmd = " -t v4l2:help";

	QStringList lines = getProcessOutput(executable, cmd);

	const QString dev_str = "Device /dev/video";
	const int path_offset = strlen("Device ");

	for(int i = 0; i < lines.count(); i++){
		QString line = lines[i].trimmed();
		int idx;
		if((idx = line.indexOf(dev_str)) >= 0){
			Webcam cam;
			cam.config = "device=" + line.mid(idx + path_offset).split(' ')[0].toStdString();
			QString name = line.mid(line.indexOf('(', idx));
			name.chop(2);
			cam.name = name.toStdString();
			cam.type = "v4l2";
			webcams.push_back(std::move(cam));
		}
	}

	for(const auto cam : webcams){
		std::cout << cam.name << ": " << cam.config << std::endl;
	}
#if 0
	const char *v4l2str = "[cap] (v4l2:";
	const size_t capStrLen = strlen(v4l2str);

	foreach ( QString line, lines ) {
		if(line.startsWith(v4l2str)){
			line.remove(0, capStrLen);
			int pos = line.indexOf(';');
			std::string path = line.mid(0, pos).toStdString();
			std::string name = line.mid(pos).toStdString();
			v4l2Devices.push_back({name, path});
		}
	}
#endif

}

void AvailableSettings::queryCap(const QStringList &lines,
		SettingType type,
		const char *capStr)
{
	size_t capStrLen = strlen(capStr);
	foreach ( const QString &line, lines ) {
		if(line.startsWith(capStr)){
			available[type].push_back(line.mid(capStrLen).toStdString());
		}
	}
}

void AvailableSettings::query(const std::string &executable, SettingType type){
	std::string opt;

	switch(type){
		case VIDEO_SRC: opt = "-t"; break;
		case VIDEO_DISPLAY: opt = "-d"; break;
		case VIDEO_COMPRESS: opt = "-c"; break;
		case VIDEO_CAPTURE_FILTER: opt = "--capture-filter"; break;
		case AUDIO_SRC: opt = "-s"; break;
		case AUDIO_PLAYBACK: opt = "-r"; break;
		case AUDIO_COMPRESS: opt = "--audio-codec"; break;
		default: return;
	}

	available[type] = getAvailOpts(opt, executable);
}

bool AvailableSettings::isAvailable(const std::string &name, SettingType type) const{
	return vectorContains(available[type], name);
}

std::vector<std::string> AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

