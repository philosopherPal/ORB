#-------------------------------------------------
#
# Project created by QtCreator 2014-07-07T23:25:47
# Author: Pallavi Taneja
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = untitled
TEMPLATE = app


SOURCES += main.cpp \





INCLUDEPATH += /opencv2.4.9/include

LIBS += -L/usr/lib/ \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_nonfree \
        -lopencv_ml \
        -lopencv_features2d
