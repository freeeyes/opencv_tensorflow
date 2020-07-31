#pragma once

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

//opencv相关
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//科学计算相关
#include "Numcpp/NumCpp.hpp"

//http客户端相关
#include "brynet/net/http/HttpService.hpp"
#include "brynet/net/http/HttpFormat.hpp"
#include "brynet/net/AsyncConnector.hpp"
#include "brynet/net/wrapper/HttpConnectionBuilder.hpp"
#include "brynet/base/AppStatus.hpp"

//ffmpeg编码传输相关
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}

#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "swscale.lib")
#ifdef _DEBUG
#pragma comment(lib, "opencv_world420d.lib")
#else
#pragma comment(lib, "opencv_world420.lib")
#endif

using namespace cv;
using namespace std;

using namespace brynet;
using namespace brynet::net;
using namespace brynet::net::http;

enum enum_start_state
{
	VEDIO_RUN_OK = 0,
	FFMPEG_INIT_ERROR,
	FFMPEG_ENCODE_ERROR,
	OPENCV_READ_CAPTURE_ERROR,
};

//数据控制接口
class Cvedio_option_control
{
public:
	bool curr_run_state = false;   //运行状态标记，false为停止，true为运行
	int start_error = enum_start_state::VEDIO_RUN_OK;
};

bool get_person_mask_back_image(const brynet::net::AsyncConnector::Ptr& connector, 
	brynet::net::TcpService::Ptr service, 
	std::string& mask_array_data,
	std::mutex& mtx,
	std::condition_variable& cv,
	Mat& img, 
	int video_width, 
	int video_height, 
	string mask_url, 
	const char* url_ip,
	short url_port,
	string back_img_file);

void get_virtual_img_mat(string mask_array_data, Mat& curr_frame, Mat back_img_frame, int video_width, int video_height);

bool init_ffmpeg_rtmp_out_stream(int infps,
	const char* rtmpurl,
	SwsContext*& vsc,
	int video_width,
	int video_height,
	AVFrame*& yuv,
	AVCodecContext*& encodec_ctx,
	AVFormatContext*& ofc,
	AVStream*& outstream);

bool send_ffmpeg_rtmp_frame(Mat frame, 
	SwsContext* vsc, 
	AVFrame* yuv, 
	AVCodecContext* encodec_ctx, 
	AVFormatContext* ofc, 
	unsigned int& frams_index,
	AVStream* outstream,
	AVPacket& outpacket);

void opencv_virtual_backimg(std::mutex& mtx_virtual,
	std::condition_variable& cv_virtual,
	int video_width,
	int video_height, 
	const char* mask_url, 
	const char* url_ip, 
	short url_port, 
	const char* back_img_file, 
	const char* rtmp_rul, 
	Cvedio_option_control*& vedio_option_control);

void opencv_beauty(std::mutex& mtx_beauty,
	std::condition_variable& cv_beauty, 
	int video_width,
	int video_height, 
	int beauty_value, 
	const char* rtmp_rul, 
	Cvedio_option_control*& vedio_option_control);

void opencv_camera_to_rtmp(int videowidth, int videogeight, const char* rtmpurl);

void init_face_flter(CascadeClassifier& face_cascade);

void detect_and_display(Mat& frame, CascadeClassifier& face_cascade, Mat& faceMask);

void display_contour(string file_img);

void sports_background();

int get_vedio_device_count();

//美颜接口
Cvedio_option_control* beauty_start(int video_width, int video_height, int beauty_value, const char* rtmp_rul);

//虚拟背景接口
Cvedio_option_control* virtual_background_start(int video_width, int video_height, const char* mask_url, const char* url_ip, short url_port, const char* back_img_file, const char* rtmp_rul);

//关闭美颜线程
void beauty_stop(Cvedio_option_control* vedio_option_control);

//关闭虚拟背景线程
void virtual_background_stop(Cvedio_option_control* vedio_option_control);