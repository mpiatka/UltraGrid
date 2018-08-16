######################################################################
# Automatically generated by qmake (3.0) Wed Jun 21 13:05:24 2017
######################################################################

TEMPLATE = app
TARGET = uv-qt
INCLUDEPATH += .
INCLUDEPATH += $$PWD/../../tools/
INCLUDEPATH += $$PWD/../../src
INCLUDEPATH += window/
INCLUDEPATH += util/
INCLUDEPATH += widget/
INCLUDEPATH += option/
RC_FILE = uv-qt.rc

DEFINES += GUI_BUILD

QT += widgets

CONFIG += c++11

LIBS += $$PWD/../../tools/astat.a
macx {
	LIBS += -framework CoreFoundation
} win32 {
	LIBS += -lWs2_32
}

astat.target = astat_lib
astat.commands = cd $$PWD/../../tools && make -f Makefile.astat lib

QMAKE_EXTRA_TARGETS += astat
PRE_TARGETDEPS += astat_lib


# Input
HEADERS += window/ultragrid_window.hpp \
	option/ultragrid_option.hpp \
	option/available_settings.hpp \
	option/settings.hpp \
	option/settings_ui.hpp \
	util/v4l2.hpp \
	widget/previewWidget.hpp \
	window/log_window.hpp \
	../../tools/astat.h \
	../../src/shared_mem_frame.hpp \
	widget/vuMeterWidget.hpp \
	window/settings_window.hpp \

FORMS += ui/ultragrid_window.ui \
	ui/log_window.ui \
	ui/settings.ui

SOURCES += window/ultragrid_window.cpp \
	option/ultragrid_option.cpp \
	option/available_settings.cpp \
	option/settings.cpp \
	option/settings_ui.cpp \
	util/v4l2.cpp \
	widget/previewWidget.cpp \
	window/log_window.cpp \
	widget/vuMeterWidget.cpp \
	window/settings_window.cpp \
	../../src/shared_mem_frame.cpp \
	main.cpp
