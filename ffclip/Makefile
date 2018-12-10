CC = gcc
CXX = g++
CFLAGS = -g -Wno-deprecated -Wl,-Bdynamic
LDFLAGS = 
#LIBS = -lavformat -lavcodec -lavutil -lyasm -lx265 -lwebp -lwavpack -lvorbisfile -lvorbisenc -lvorbis -lvo-amrwbenc -lvo-aacenc -ltwolame -ltheoraenc -ltheoradec -lpng -ltheora -lswscale -lswresample -lpostproc -lrtmp -lopus -lilbc -lfdk-aac -lpthread -lmp3lame -lopenjpeg -lspeex -logg -ltheoradec -ltheoraenc -lvo-aacenc -lvo-amrwbenc -lvorbisenc -lvorbis -lvpx -lx264 -lfreetype -lm -lz -lass -lfontconfig -lexpat -lfribidi -lfreetype -logg -lgnutls -lssl -lcrypto -lm -lz -lm -lz -pthread -ldl -lrt
LIBS = -lavformat -lavdevice -lavcodec -lavutil -lpthread -lswscale -lswresample -lm -ldl -lrt

INC_DIR = -I/usr/local/ssl/include -I/usr/local/lib/include
LIB_DIR = -Llib/

objects = FFClips.o FFKeyFrameClips.o

EXECNAME = ffclipinfo

all:$(objects)
	g++ -o $(EXECNAME)_g *.o $(LIB_DIR) $(LIBS)
	cp $(EXECNAME)_g $(EXECNAME)
	strip $(EXECNAME)

$(objects):FFClips.h
	$(CXX) $(CFLAGS) -c FFClips.cpp $(INC_DIR)
	$(CXX) $(CFLAGS) -c FFKeyFrameClips.cpp $(INC_DIR)
    
clean:
	rm -f $(EXECNAME).* $(objects)
