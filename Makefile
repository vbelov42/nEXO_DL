BOOST_DIR=
BOOST_LIBS=-lboost_python3 -lboost_numpy3 -lpython3.7m
SNIPER_DIR=${HOME}/bin/sniper
SNIPER_LIBS=$(SNIPER_DIR)/lib64/libSniperKernel.so
ROOT_DIR=/usr/include/root
ROOT_LIBS=
NEXO_DIR=${HOME}/work/nexo-offline/build
NEXO_LIBS=-L$(NEXO_DIR)/lib -lEvtNavigator -lEDMUtil -lBaseEvent -lSimEvent -lPidTmvaEvent

CXX_FLAGS = -fPIC   -std=c++14 -g -ggdb

CXX_DEFINES = 

CXX_INCLUDES = -I$(NEXO_DIR)/include -I$(SNIPER_DIR)/include -I$(ROOT_DIR) -I/usr/include/python3.7m 

all: DnnEventTagger/libDnnEventTagger.so

DnnEventTagger.o: DnnEventTagger.cc DnnEventTagger.hh
	g++ $(CXX_FLAGS) $(CXX_DEFINES) $(CXX_INCLUDES) -c -o $@ $<

DnnEventTagger/libDnnEventTagger.so: DnnEventTagger.o
	g++ -fPIC -g -ggdb  -shared -Wl,-soname,libDnnEventTagger.so -o $@ $^ -Wl,-rpath,$(NEXO_DIR)/lib:$(SNIPER_DIR)/lib64:/usr/lib64/root $(NEXO_LIBS) $(SNIPER_LIBS) $(BOOST_LIBS)

tar:
	tar -czf DnnEventTagger.tgz DnnEventTagger.{hh,cc} DnnEventTagger/*.py tagger-run.py Makefile
