CC  = gcc
CPP = g++

########################################
CFLAGS = -fPIC -lpthread
CPPFLAGS = -fPIC

LINK_FLAGS = -w -g -D__STDC_CONSTANT_MACROS
LIBS =  -lopencv_highgui -lopencv_core -lopencv_imgproc \
		#-lopencv_imgcodecs -lavformat -lavcodec -lavutil -lswscale \
#########################################################

TARGET0 = cvtest
TARGETS = $(TARGET0)

CPPOBJS = main.o \
			testcv.o \
			cv_image.o

COBJS = 

$(TARGET0):$(CPPOBJS) $(COBJS)
	$(CPP) $(LINK_FLAGS) -o $@ $(CPPOBJS)  $(COBJS) $(LIBS) 

clean:
	clear
	rm -rf $(COBJS) $(CPPOBJS) *.o $(TARGETS) *.ts

#$(OBJS):%.o :%.c  先用$(OBJS)中的一项，比如foo.o: %.o : %.c  含义为:试着用%.o匹配foo.o。如果成功%就等于foo。如果不成功，  
# Make就会警告，然后。给foo.o添加依赖文件foo.c(用foo替换了%.c里的%)  
# 也可以不要下面的这个生成规则，因为下面的 include $(DEF)  就隐含了。此处为了明了，易懂。故留着  
$(COBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
$(CPPOBJS): %.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@
