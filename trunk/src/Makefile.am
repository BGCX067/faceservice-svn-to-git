LIBS = $(OPENCV_LIBS) $(CURL_LIBS) -L$(cgicc_libdir) -lcgicc 
INCLUDES = $(OPENCV_CFLAGS) $(cgicc_CFLAGS) $(CURL_CFLAGS) -I$(cgicc_includedir)

bin_PROGRAMS = faceservice.cgi
faceservice_cgi_SOURCES = faceservice.cpp
