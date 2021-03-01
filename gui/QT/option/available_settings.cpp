#include <QProcess>
#include <QString>
#include <QStringList>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <cstring>
#include "available_settings.hpp"

#include <iostream>
#include <map>
#include <algorithm>

static bool vectorContains(const std::vector<std::string> &v, const std::string & s){
	for(unsigned i = 0; i < v.size(); i++){
		if(v[i] == s)
			return true;
	}
	return false;
}

static QStringList getProcessOutput(const std::string& executable, const QStringList& args){
	QProcess process;
	process.start(executable.c_str(), args);
	process.waitForFinished();

	QString output = QString(process.readAllStandardOutput());
	QStringList lines = output.split(QRegularExpression("\n|\r\n|\r"));

	return lines;
}

void AvailableSettings::queryAll(const std::string &executable){
	QStringList lines = getProcessOutput(executable, QStringList() << "--capabilities");

	queryCap(lines, VIDEO_SRC, "[cap][capture] ");
	queryCap(lines, VIDEO_DISPLAY, "[cap][display] ");
	queryCap(lines, VIDEO_COMPRESS, "[cap][compress] ");
	queryCap(lines, VIDEO_CAPTURE_FILTER, "[cap][capture_filter] ");
	queryCap(lines, AUDIO_SRC, "[cap][audio_cap] ");
	queryCap(lines, AUDIO_PLAYBACK, "[cap][audio_play] ");
	queryCap(lines, AUDIO_COMPRESS, "[cap][audio_compress] ");
	
	queryDevices(lines);
	queryVideoCompress(lines);
}

static void maybeWriteString (const QJsonObject& obj,
		const char *key,
		std::string& result)
{
		if(obj.contains(key) && obj[key].isString()){
			result = obj[key].toString().toStdString();
		}
}

void AvailableSettings::queryVideoCompress(const QStringList &lines){
	const char * const devStr = "[capability][video_compress][v3]";
	size_t devStrLen = strlen(devStr);

	videoCompressModules.clear();

	videoCompressModules.emplace_back(CompressModule{"", {}, });
	videoCompressCodecs.emplace_back(Codec{"None", "", {Encoder{"default", ""}}, 0});

	foreach ( const QString &line, lines ) {
		if(!line.startsWith(devStr))
			continue;

		QJsonDocument doc = QJsonDocument::fromJson(line.mid(devStrLen).toUtf8());
		if(!doc.isObject())
			continue;

		CompressModule compMod;
		QJsonObject obj = doc.object();
		maybeWriteString(obj, "name", compMod.name);

		if(obj.contains("options") && obj["options"].isArray()){
			for(const QJsonValue &val : obj["options"].toArray()){
				QJsonObject optJson = val.toObject();

				CapabOpt capabOpt;
				maybeWriteString(optJson, "display_name", capabOpt.displayName);
				maybeWriteString(optJson, "display_desc", capabOpt.displayDesc);
				maybeWriteString(optJson, "key", capabOpt.key);
				maybeWriteString(optJson, "opt_str", capabOpt.optStr);
				if(optJson.contains("is_boolean") && optJson["is_boolean"].isString()){
					capabOpt.booleanOpt = optJson["is_boolean"].toString() == "t";
				}

				compMod.opts.emplace_back(std::move(capabOpt));
			}
		}

		if(obj.contains("codecs") && obj["codecs"].isArray()){
			for(const QJsonValue &val : obj["codecs"].toArray()){
				QJsonObject codecJson = val.toObject();

				Codec codec;
				maybeWriteString(codecJson, "name", codec.name);
				codec.module_name = compMod.name;
				if(codecJson.contains("priority") && codecJson["priority"].isDouble()){
					codec.priority = codecJson["priority"].toInt();
				}

				if(codecJson.contains("encoders") && codecJson["encoders"].isArray()){
					for(const QJsonValue &val : codecJson["encoders"].toArray()){
						QJsonObject encoderJson = val.toObject();

						Encoder encoder;
						maybeWriteString(encoderJson, "name", encoder.name);
						maybeWriteString(encoderJson, "opt_str", encoder.optStr);

						codec.encoders.emplace_back(std::move(encoder));
					}
				}

				if(codec.encoders.empty()){
					codec.encoders.emplace_back(Encoder{"default", ""});
				}

				videoCompressCodecs.emplace_back(std::move(codec));
			}
		}

		videoCompressModules.emplace_back(std::move(compMod));
	}

	std::sort(videoCompressCodecs.begin(), videoCompressCodecs.end(),
			[](const Codec& a, const Codec& b){
				return a.priority < b.priority;
			});

}

void AvailableSettings::queryDevices(const QStringList &lines){
	const char * const devStr = "[capability][device][v2]";
	size_t devStrLen = strlen(devStr);

	for(auto& i : devices){
		i.clear();
	}

	static std::map<std::string, SettingType> settingTypeMap = {
		{"video_cap", VIDEO_SRC},
		{"video_disp", VIDEO_DISPLAY},
		{"audio_play", AUDIO_PLAYBACK},
		{"audio_cap", AUDIO_SRC},
	};

	foreach ( const QString &line, lines ) {
		if(line.startsWith(devStr)){
			QJsonDocument doc = QJsonDocument::fromJson(line.mid(devStrLen).toUtf8());
			if(!doc.isObject())
				return;

			Device dev;
			QJsonObject obj = doc.object();
			if(obj.contains("name") && obj["name"].isString()){
				dev.name = obj["name"].toString().toStdString();
			}
			if(obj.contains("type") && obj["type"].isString()){
				dev.type = obj["type"].toString().toStdString();
			}
			if(obj.contains("device") && obj["device"].isString()){
				dev.deviceOpt = obj["device"].toString().toStdString();
			}

			if(obj.contains("modes") && obj["modes"].isArray()){
				for(const QJsonValue &val : obj["modes"].toArray()){
					if(val.isObject()){
						QJsonObject modeJson = val.toObject();

						DeviceMode mode;

						if(modeJson.contains("name") && modeJson["name"].isString()){
							mode.name = modeJson["name"].toString().toStdString();
						}

						if(modeJson.contains("opts") && modeJson["opts"].isObject()){
							QJsonObject modeOpts = modeJson["opts"].toObject();
							for(const QString &key : modeOpts.keys()){
								if(modeOpts[key].isString()){
									mode.opts.push_back(SettingVal{key.toStdString(),
											modeOpts[key].toString().toStdString()});
								}
							}
							dev.modes.push_back(std::move(mode));
						}
					}
				}
			}

			SettingType settingType = SETTING_TYPE_UNKNOWN;
			if(obj.contains("purpose") && obj["purpose"].isString()){
				settingType = settingTypeMap[obj["purpose"].toString().toStdString()];
			}
			devices[settingType].push_back(std::move(dev));
		}
	}
}

void AvailableSettings::queryCap(const QStringList &lines,
		SettingType type,
		const char *capStr)
{
	available[type].clear();

	size_t capStrLen = strlen(capStr);
	foreach ( const QString &line, lines ) {
		if(line.startsWith(capStr)){
			available[type].push_back(line.mid(capStrLen).toStdString());
		}
	}
}

bool AvailableSettings::isAvailable(const std::string &name, SettingType type) const{
	return vectorContains(available[type], name);
}

std::vector<std::string> AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

const std::vector<Device>& AvailableSettings::getDevices(SettingType type) const{
	return devices[type];
}

