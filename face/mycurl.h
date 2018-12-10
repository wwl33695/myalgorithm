#include <stdio.h>
#include <curl/curl.h>
#include <stdlib.h>
#include <string>
#include <string.h>
using namespace std;
//g++ curl.cpp -o mycurl  -lcurl

class SendFile
{
public:
	SendFile(string &url);
	~SendFile();
	int InitSendFile();
	int MySend(char *fileUrl, char *imgName);
	//size_t read_data(void *buffer, size_t size, size_t nmemb, void *user_p);
private:
	CURL * m_curl;
	string m_sendUrl;
};




