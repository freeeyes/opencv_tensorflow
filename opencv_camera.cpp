#include "opencv_camera.h"

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
	string back_img_file)
{
	std::vector<uchar> url_img_buf;
	bool ret = false;
	ret = imencode(".jpg", img, url_img_buf);

	std::string send_buff(url_img_buf.begin(), url_img_buf.end());

	if (true == ret)
	{
		//组装发送字符串
		brynet::net::http::HttpRequest request;
		request.setMethod(brynet::net::http::HttpRequest::HTTP_METHOD::HTTP_METHOD_POST);
		request.setContentType("application/octet-stream");
		request.setBody(send_buff);
		request.setUrl(mask_url);
		request.setHost(mask_url);

		std::string requestStr = request.getResult();

		wrapper::HttpConnectionBuilder()
			.configureConnector(connector)
			.configureService(service)
			.configureConnectOptions({
				ConnectOption::WithAddr(url_ip, url_port),
				ConnectOption::WithTimeout(std::chrono::seconds(10)),
				ConnectOption::WithFailedCallback([&mtx, &cv]() {
						std::unique_lock <std::mutex> lck(mtx);
						std::cout << "connect failed" << std::endl;
						cv.notify_one();
					}),
				})
			.configureConnectionOptions({
				AddSocketOption::WithMaxRecvBufferSize(1024),
				AddSocketOption::AddEnterCallback([](const TcpConnection::Ptr& session) {
					// do something for session
					std::cout << "connect OK" << std::endl;
				})
				})
			.configureEnterCallback([requestStr, &mask_array_data, &mtx, &cv](const HttpSession::Ptr& session, HttpSessionHandlers& handlers) {
				(void)session;
				//std::cout << "connect success" << std::endl;
				session->send(requestStr.c_str(), requestStr.size());
				handlers.setHttpCallback([requestStr, &mask_array_data, &mtx, &cv](const HTTPParser& httpParser,
					const HttpSession::Ptr& session) {
						std::unique_lock <std::mutex> lck(mtx);

						//获得返回数据包
						mask_array_data = httpParser.getBody();
						
						cv.notify_one();
					});
						})
			.asyncConnect();
	}

	return true;
}

void get_virtual_img_mat(string mask_array_data, Mat& curr_frame, Mat back_img_frame, int video_width, int video_height)
{
	//处理接收到的数据 mask_array_data
	std::cout << "[get_virtual_img_mat]recv data:" << mask_array_data.length() << std::endl;

	//如果返回蒙版大小
	if (mask_array_data.length() > 0)
	{
		nc::NdArray<nc::uint8> mask = nc::frombuffer<nc::uint8>(mask_array_data.c_str(), mask_array_data.length());
		mask = mask.reshape(video_width, video_height);

		//开始按照行生成新图像(算法，img[:,:,c] = img[:,:,c]*mask + replacement_bg[:,:,c]*inv_mask)
		for (int i = 0; i < curr_frame.rows; i++)
		{
			for (int j = 0; j < curr_frame.cols; j++)
			{
				cv::Vec3b& srcColor = curr_frame.at<cv::Vec3b>(i, j);
				const cv::Vec3b& backimgColor = back_img_frame.at<cv::Vec3b>(i, j);
				srcColor[0] = srcColor[0] * mask.at(i * curr_frame.cols + j) + backimgColor[0] * (1 - mask.at(i * curr_frame.cols + j));
				srcColor[1] = srcColor[1] * mask.at(i * curr_frame.cols + j) + backimgColor[1] * (1 - mask.at(i * curr_frame.cols + j));
				srcColor[2] = srcColor[2] * mask.at(i * curr_frame.cols + j) + backimgColor[2] * (1 - mask.at(i * curr_frame.cols + j));
			}
		}
	}
	else
	{
		//直接显示虚拟背景
		curr_frame = back_img_frame;
	}
}

bool init_ffmpeg_rtmp_out_stream(int infps, 
	const char* rtmpurl, 
	SwsContext*& vsc, 
	int video_width, 
	int video_height, 
	AVFrame*& yuv, 
	AVCodecContext*& encodec_ctx,
	AVFormatContext*& ofc,
	AVStream*& outstream)
{
	int ret = 0;
	vsc = sws_getCachedContext(vsc,
		video_width,
		video_height,
		AV_PIX_FMT_BGR24,
		video_width,
		video_height,
		AV_PIX_FMT_YUV420P,
		SWS_BICUBIC,
		nullptr,
		nullptr,
		nullptr);

	if (!vsc)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]create sws error." << endl;
		return false;
	}

	yuv = av_frame_alloc();
	yuv->format = AV_PIX_FMT_YUV420P;
	yuv->width = video_width;
	yuv->height = video_height;
	yuv->pts = 0;
	ret = av_frame_get_buffer(yuv, 32);
	if (ret != 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]av_frame_get_buffer error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	AVCodec* encodec = nullptr;
	encodec = avcodec_find_encoder(AV_CODEC_ID_H264);
	if (!encodec)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]get libx264 error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	encodec_ctx = avcodec_alloc_context3(encodec);
	if (!encodec_ctx)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]get encodec_ctx error." << endl;
		sws_freeContext(vsc);
		return  false;
	}
	encodec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
	encodec_ctx->codec_id = encodec->id;
	encodec_ctx->thread_count = 4;
	encodec_ctx->bit_rate = 50 * 1024 * 8;
	encodec_ctx->width = video_width;
	encodec_ctx->height = video_height;
	encodec_ctx->time_base = { 1, infps };
	encodec_ctx->framerate = { infps, 1 };
	encodec_ctx->gop_size = 1;
	encodec_ctx->max_b_frames = 0;
	encodec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

	AVDictionary* param = nullptr;
	av_dict_set(&param, "preset", "superfast", 0);
	av_dict_set(&param, "tune", "zerolatency", 0);
	av_dict_set(&param, "profile", "main", 0);

	ret = avcodec_open2(encodec_ctx, nullptr, &param);
	if (ret != 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]avcodec_open2 error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	av_dict_free(&param);

	//设置实处流
	ret = avformat_alloc_output_context2(&ofc, nullptr, "flv", rtmpurl);
	if (ret != 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]avformat_alloc_output_context2 error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	outstream = avformat_new_stream(ofc, nullptr);
	if (ret != 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]avformat_new_stream error." << endl;
		sws_freeContext(vsc);
		return false;
	}
	outstream->codecpar->codec_tag = 0;
	avcodec_parameters_from_context(outstream->codecpar, encodec_ctx);

	av_dump_format(ofc, 0, rtmpurl, 1);

	ret = avio_open(&ofc->pb, rtmpurl, AVIO_FLAG_WRITE);
	if (ret != 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]avio_open error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	ret = avformat_write_header(ofc, nullptr);
	if (ret < 0)
	{
		cout << "[init_ffmpeg_rtmp_out_stream]avformat_write_header error." << endl;
		sws_freeContext(vsc);
		return false;
	}

	return true;
}

bool send_ffmpeg_rtmp_frame(Mat frame, 
	SwsContext* vsc, 
	AVFrame* yuv, 
	AVCodecContext* encodec_ctx, 
	AVFormatContext* ofc, 
	unsigned int& frams_index,
	AVStream* outstream,
	AVPacket& outpacket)
{
	uint8_t* indata[AV_NUM_DATA_POINTERS] = { 0 };
	int inlinesize[AV_NUM_DATA_POINTERS] = { 0 };
	indata[0] = frame.data;
	inlinesize[0] = frame.cols * frame.elemSize();

	int outhight = sws_scale(vsc, indata, inlinesize, 0, frame.rows,
		yuv->data, yuv->linesize);
	if (outhight <= 0)
	{
		return true;
	}

	yuv->pts = frams_index;
	std::cout << "yuv pts(1)=" << yuv->pts << endl;
	frams_index++;
	

	int ret = avcodec_send_frame(encodec_ctx, yuv);
	if (ret != 0)
	{
		return true;
	}

	ret = avcodec_receive_packet(encodec_ctx, &outpacket);
	if (ret == 0 && outpacket.size > 0)
	{
		//encode OK
		std::cout << "pts=" << av_rescale_q(outpacket.pts, encodec_ctx->time_base, outstream->time_base) << std::endl;
		std::cout << "dts=" << av_rescale_q(outpacket.dts, encodec_ctx->time_base, outstream->time_base) << std::endl;
		std::cout << "duration=" << av_rescale_q(outpacket.duration, encodec_ctx->time_base, outstream->time_base)  << std::endl;

		outpacket.pts = av_rescale_q(outpacket.pts, encodec_ctx->time_base, outstream->time_base);
		outpacket.dts = av_rescale_q(outpacket.dts, encodec_ctx->time_base, outstream->time_base);
		outpacket.duration = av_rescale_q(outpacket.duration, encodec_ctx->time_base, outstream->time_base);

		ret = av_interleaved_write_frame(ofc, &outpacket);
		if (ret == 0)
		{
			cout << "#" << flush;
		}
	}
	else
	{
		char av_error[1024] = { '\0' };
		av_strerror(ret, av_error, 1024);
		cerr << "[av_error_string_output]error=" << av_error << endl;
	}

	return true;
}

void opencv_virtual_backimg(std::mutex& mtx_virtual,
	std::condition_variable& cv_virtual, 
	int video_width, 
	int video_height, 
	const char* mask_url, 
	const char* url_ip, 
	short url_port, 
	const char* back_img_file, 
	const char* rtmp_rul, 
	Cvedio_option_control*& vedio_option_control)
{
	std::mutex mtx;
	std::condition_variable cv;

	VideoCapture cam;

	//初始化ffmpeg编码参数和流格式
	SwsContext* vsc = nullptr;
	AVFrame* yuv = nullptr;
	AVFormatContext* ofc = nullptr;
	AVStream* outstream = nullptr;
	AVCodecContext* encodec_ctx = nullptr;

	unsigned int frams_index = 0;
	AVPacket outpacket;
	memset(&outpacket, 0, sizeof(outpacket));
	Mat frame;

	//初始化http服务
	brynet::net::AsyncConnector::Ptr connector = brynet::net::AsyncConnector::Create();
	connector->startWorkerThread();
	brynet::net::TcpService::Ptr service = TcpService::Create();
	service->startWorkerThread(1);

	//获取背景数据
	Mat background_img = imread(back_img_file);
	Mat background_img_resize;

	//这个cv是用来告诉上层摄像头已经准备好了，锁作用域
	{
		std::unique_lock <std::mutex> lck(mtx_virtual);
		//创建一个新的控制对象
		if (nullptr != vedio_option_control)
		{
			delete vedio_option_control;
			vedio_option_control = nullptr;
		}
		vedio_option_control = new Cvedio_option_control();

		avformat_network_init();


		cv::resize(background_img, background_img_resize, cv::Size(video_width, video_height));

		if (false == cam.open(0 + cv::CAP_DSHOW))
		{
			cout << "[opencv_camera_to_rtmp]capture error." << endl;
			vedio_option_control->start_error = enum_start_state::OPENCV_READ_CAPTURE_ERROR;
			cv_virtual.notify_one();
			return;
		}

		if (!cam.isOpened())
		{
			cout << "[opencv_camera_to_rtmp]capture open error." << endl;
			vedio_option_control->start_error = enum_start_state::OPENCV_READ_CAPTURE_ERROR;
			cv_virtual.notify_one();
			return;
		}

		//获得分辨率
		cam.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, video_width);
		cam.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, video_height);
		cam.set(VideoCaptureProperties::CAP_PROP_FPS, 30);

		//获得帧率
		int infps = (int)cam.get(VideoCaptureProperties::CAP_PROP_FPS);

		//如果需要传输rtmp
		if (strlen(rtmp_rul) > 0)
		{
			if (false == init_ffmpeg_rtmp_out_stream(infps, rtmp_rul, vsc, video_width, video_height, yuv, encodec_ctx, ofc, outstream))
			{
				cam.release();
				cv::destroyAllWindows();
				vedio_option_control->start_error = enum_start_state::FFMPEG_INIT_ERROR;
				cv_virtual.notify_one();
				return;
			}
		}

		vedio_option_control->curr_run_state = true;
		cv_virtual.notify_one();
	}

	//获取视频帧
	while(vedio_option_control->curr_run_state)
	{
		if (cam.read(frame))
		{
			//测试虚拟背景
			if (!frame.empty())
			{
				//std::cout << "width=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH) << std::endl;
				//std::cout << "heigth=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT) << std::endl;

				imshow("src", frame);
				waitKey(1);

				std::unique_lock <std::mutex> lck(mtx);
				string mask_array_data;

				//调用Http发送数据蒙版
				get_person_mask_back_image(connector, 
					service,
					mask_array_data,
					mtx, 
					cv,
					frame, 
					video_width, 
					video_height, 
					mask_url, 
					url_ip,
					url_port,
					back_img_file);

				cv.wait(lck);

				get_virtual_img_mat(mask_array_data, frame, background_img_resize, video_width, video_height);

				imshow("dec", frame);

				//如果需要传输rtmp
				if (strlen(rtmp_rul) > 0)
				{
					send_ffmpeg_rtmp_frame(frame, 
						vsc, 
						yuv,
						encodec_ctx,
						ofc,
						frams_index,
						outstream,
						outpacket);
				}

			}
		}
	}

	//回收资源
	cam.release();
	cv::destroyAllWindows();

	//回收ffmpeg
	if (nullptr != ofc && nullptr != vsc)
	{
		avio_closep(&ofc->pb);
		sws_freeContext(vsc);
	}

	//回收控制器
	delete vedio_option_control;
	return;

}

void opencv_beauty(std::mutex& mtx_beauty,
	std::condition_variable& cv_beauty, 
	int video_width, 
	int video_height, 
	int beauty_value, 
	const char* rtmp_rul, 
	Cvedio_option_control*& vedio_option_control)
{
	VideoCapture cam;

	//初始化ffmpeg编码参数和流格式
	SwsContext* vsc = nullptr;
	AVFrame* yuv = nullptr;
	AVFormatContext* ofc = nullptr;
	AVStream* outstream = nullptr;
	AVCodecContext* encodec_ctx = nullptr;

	unsigned int frams_index = 0;
	AVPacket outpacket;
	memset(&outpacket, 0, sizeof(outpacket));

	//这个cv是用来告诉上层摄像头已经准备好了，锁作用域
	{
		std::unique_lock <std::mutex> lck(mtx_beauty);

		//创建一个新的控制对象
		if (nullptr != vedio_option_control)
		{
			delete vedio_option_control;
			vedio_option_control = nullptr;
		}

		vedio_option_control = new Cvedio_option_control();

		avformat_network_init();

		if (false == cam.open(0 + cv::CAP_DSHOW))
		{
			cout << "[opencv_camera_to_rtmp]capture error." << endl;
			vedio_option_control->curr_run_state = enum_start_state::OPENCV_READ_CAPTURE_ERROR;
			cv_beauty.notify_one();
			return;
		}

		if (!cam.isOpened())
		{
			cout << "[opencv_camera_to_rtmp]capture open error." << endl;
			vedio_option_control->curr_run_state = enum_start_state::OPENCV_READ_CAPTURE_ERROR;
			cv_beauty.notify_one();
			return;
		}

		cam.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, video_width);
		cam.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, video_height);
		cam.set(VideoCaptureProperties::CAP_PROP_FPS, 30);

		cout << "capture success." << endl;

		int infps = (int)cam.get(VideoCaptureProperties::CAP_PROP_FPS);

		cout << "CAP_PROP_FRAME_WIDTH=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH) << endl;
		cout << "CAP_PROP_FRAME_WIDTH=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "infps=" << infps << endl;

		//如果需要传输rtmp
		if (strlen(rtmp_rul) > 0)
		{
			if (false == init_ffmpeg_rtmp_out_stream(infps, rtmp_rul, vsc, video_width, video_height, yuv, encodec_ctx, ofc, outstream))
			{
				cam.release();
				cv::destroyAllWindows();
				vedio_option_control->curr_run_state = enum_start_state::FFMPEG_INIT_ERROR;
				cv_beauty.notify_one();
				return;
			}
		}

		//通知上层启动摄像头成功
		vedio_option_control->curr_run_state = true;
		cv_beauty.notify_one();
	}

	Mat frame;
	Mat image_frame;
	//获取视频帧
	while(vedio_option_control->curr_run_state)
	{
		if (cam.read(frame))
		{
			//测试虚拟背景
			if (!frame.empty())
			{
				imshow("src", frame);
				waitKey(1);

				cv::bilateralFilter(frame, image_frame, beauty_value, (double)(beauty_value * 2), (double)(beauty_value / 2));

				imshow("dec", image_frame);

				//如果需要传输rtmp
				if (strlen(rtmp_rul) > 0)
				{
					send_ffmpeg_rtmp_frame(image_frame,
						vsc,
						yuv,
						encodec_ctx,
						ofc,
						frams_index,
						outstream,
						outpacket);
				}
			}
		}
	}

	//回收资源
	cam.release();
	cv::destroyAllWindows();

	//回收ffmpeg
	if (nullptr != ofc && nullptr != vsc)
	{
		avio_closep(&ofc->pb);
		sws_freeContext(vsc);
	}
}

void opencv_camera_to_rtmp(int videowidth, int videoheight, const char* rtmpurl)
{
	CascadeClassifier face_cascade;      //人脸检测的类

	VideoCapture cam;

	//namedWindow("freeeyes", WINDOW_AUTOSIZE);
	avformat_network_init();

	init_face_flter(face_cascade);

	//定义面具
	Mat faceMask = imread("C:\\freeeyeswork\\Learn-OpenCV-4-By-Building-Projects-Second-Edition\\Chapter_07\\resources\\mask.jpg");

	if (false == cam.open(0))
	{
		cout << "[opencv_camera_to_rtmp]capture error." << endl;
		return;
	}

	if (!cam.isOpened())
	{
		cout << "[opencv_camera_to_rtmp]capture open error." << endl;
		return;
	}

	cam.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, videowidth);
	cam.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, videoheight);

	cout << "capture success." << endl;

	int infps = (int)cam.get(VideoCaptureProperties::CAP_PROP_FPS);

	cout << "CAP_PROP_FRAME_WIDTH=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH) << endl;
	cout << "CAP_PROP_FRAME_WIDTH=" << cam.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "CAP_PROP_FRAME_WIDTH=" << infps << endl;

	infps = 2;

	Mat frame;

	//初始化ffmpeg编码参数和流格式
	SwsContext* vsc = nullptr;
	AVFrame* yuv = nullptr;
	AVFormatContext* ofc = nullptr;
	AVStream* outstream = nullptr;
	AVCodecContext* encodec_ctx = nullptr;

	unsigned int frams_index = 0;
	AVPacket outpacket;
	memset(&outpacket, 0, sizeof(outpacket));

	//如果需要传输rtmp
	if (strlen(rtmpurl) > 0)
	{
		if (false == init_ffmpeg_rtmp_out_stream(infps, rtmpurl, vsc, videowidth, videoheight, yuv, encodec_ctx, ofc, outstream))
		{
			cam.release();
			cv::destroyAllWindows();
			return;
		}
	}

	for (;;)
	{
		if (cam.read(frame))
		{
			//测试人脸识别
			if (!frame.empty())
			{
				//imwrite("c.jpg", frame);
				//detect_and_display(frame, face_cascade, faceMask);
				imshow("src", frame);
				waitKey(500);

				//如果需要传输rtmp
				if (strlen(rtmpurl) > 0)
				{
					send_ffmpeg_rtmp_frame(frame,
						vsc,
						yuv,
						encodec_ctx,
						ofc,
						frams_index,
						outstream,
						outpacket);
				}
			}
			

		}
		else
		{
			continue;
		}
	}

	avio_closep(&ofc->pb);
	//avcodec_free_context(&encodec_ctx);
	sws_freeContext(vsc);
	vsc = nullptr;
}

void init_face_flter(CascadeClassifier& face_cascade)
{
	//加载人脸识别
	string face_cascade_name = "C:\\Tools\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";   //加载分类器

	//加载级联分类器文件
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "[detectAndDisplay]Error loading face_cascade" << endl;
		return;
	};
}

void detect_and_display(Mat& frame, CascadeClassifier& face_cascade, Mat& faceMask)
{
	string window_name = "Capture - Face detection";

	Mat faceROI;
	Mat faceMaskSmall;
	Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
	Mat maskedFace, maskedFrame;
	                                                                                                   
	vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);    //直方图均衡化

	//-- 多尺寸检测人脸
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 2 | 0, Size(30, 30));
	for (auto face : faces)
	{
		//绘制出圆形区域
		//Point center(face.x + face.width * 0.5, face.y + face.height * 0.5);
		//ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		//绘制出矩形区域 
		rectangle(frame, face, Scalar(127, 255, 212), 2);
		//faceROI = frame_gray(face);

		/*
		// Custom parameters to make the mask fit your face. You may have to play around with them to make sure it works.
		int x = face.x - int(0.1 * face.width);
		int y = face.y - int(0.0 * face.height);
		int w = int(1.1 * face.width);
		int h = int(1.3 * face.height);

		//切割出图片位置
		Mat frameROI = frame(Rect(x, y, w, h));

		resize(faceMask, faceMaskSmall, Size(w, h));

		// Convert the above image to grayscale
		cvtColor(faceMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);

		// Threshold the above image to isolate the pixels associated only with the face mask
		threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, THRESH_BINARY_INV);

		// Create mask by inverting the above image (because we don't want the background to affect the overlay)
		bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);

		// Use bitwise "AND" operator to extract precise boundary of face mask
		bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh);

		// Use bitwise "AND" operator to overlay face mask
		bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);

		// Add the above masked images and place it in the original frame ROI to create the final image
		if (x > 0 && y > 0 && x + w < frame.cols && y + h < frame.rows)
		{
			add(maskedFace, maskedFrame, frame(Rect(x, y, w, h)));
		}
		*/
		//imshow(window_name, frameROI);
	}

	imshow(window_name, frame);
	waitKey(1);
}

void display_contour(string file_img)
{
	//切割图像背景
	Mat src = imread(file_img);
	Mat src_noise;

	if (src.empty()) {
		cout << "[display_contour]could not load image...\n" << endl;
		return;
	}

	medianBlur(src, src_noise, 3);

	imshow("原图", src_noise);

	// 1.将二维图像数据线性化
	Mat data;
	for (int i = 0; i < src.rows; i++)     //像素点线性排列
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b point = src.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	// 2.使用K-means聚类；分离出背景色
	int numCluster = 4;
	Mat labels;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, numCluster, labels, criteria, 3, KMEANS_PP_CENTERS);

	// 3.背景与人物二值化
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = src.rows * 2 + 2;  //获取点（2，2）作为背景色
	int cindex = labels.at<int>(index);
	/*  提取背景特征 */
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			index = row * src.cols + col;
			int label = labels.at<int>(index);
			if (label == cindex) { // 背景
				mask.at<uchar>(row, col) = 0;
			}
			else {
				mask.at<uchar>(row, col) = 255;
			}
		}
	}
	//imshow("mask", mask);

	// 4.腐蚀 + 高斯模糊：图像与背景交汇处高斯模糊化
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	erode(mask, mask, k);
	//imshow("erode-mask", mask);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);
	//imshow("Blur Mask", mask);

	// 5.更换背景色以及交汇处融合处理
	RNG rng(12345);
	Vec3b color;  //设置的背景色
	color[0] = 217;//rng.uniform(0, 255);
	color[1] = 60;// rng.uniform(0, 255);
	color[2] = 160;// rng.uniform(0, 255);
	Mat result(src.size(), src.type());

	double w = 0.0;   //融合权重
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			int m = mask.at<uchar>(row, col);
			if (m == 255) {
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col); // 前景
			}
			else if (m == 0) {
				result.at<Vec3b>(row, col) = color; // 背景
			}
			else {/* 融合处理部分 */
				w = m / 255.0;
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				b = b1 * w + b2 * (1.0 - w);
				g = g1 * w + g2 * (1.0 - w);
				r = r1 * w + r2 * (1.0 - w);

				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;
			}
		}
	}
	imshow("背景替换", result);

	waitKey(0);
}

void sports_background()
{
	Ptr<BackgroundSubtractor> pBackSub;

	pBackSub = createBackgroundSubtractorMOG2();

	Mat se = getStructuringElement(MORPH_RECT, Point(3, 3));

	VideoCapture capture;

	if (false == capture.open(0))
	{
		cout << "[sports_background]capture error." << endl;
		return;
	}

	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "[sports_background]Unable to open: " << endl;
		return;
	}

	capture.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1024);
	capture.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 768);
	capture.set(VideoCaptureProperties::CAP_PROP_FPS, 30);

	Mat frame, fgMask;
	Mat frame_shold, frame_morpho, frame_back_img;
	while (true) {
		capture >> frame;
		if (frame.empty())
			break;
		//update the background model
		pBackSub->apply(frame, fgMask);
		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		
		//threshold(fgMask, frame_shold, 220, 255, THRESH_BINARY);

		//morphologyEx(frame_shold, frame_morpho, MORPH_OPEN, se);

		//pBackSub->getBackgroundImage(frame_back_img);

		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		//get the input from the keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}

}

int get_vedio_device_count()
{
	cv::VideoCapture temp_camera;
	int maxTested = 10;
	for (int i = 0; i < maxTested; i++) {
		cv::VideoCapture temp_camera(i);
		bool res = (!temp_camera.isOpened());
		temp_camera.release();
		if (res)
		{
			return i;
		}
	}
	return maxTested;
}

//以下是dll接口实现部分

Cvedio_option_control* beauty_start(int video_width, int video_height, int beauty_value, const char* rtmp_rul)
{
	std::mutex mtx_beauty;
	std::condition_variable cv_beauty;

	std::unique_lock <std::mutex> lck(mtx_beauty);

	Cvedio_option_control* vedio_option_control = nullptr;
	std::thread vedio_thread([&vedio_option_control, &mtx_beauty, &cv_beauty, video_width, video_height, beauty_value, rtmp_rul]()
		{
			opencv_beauty(mtx_beauty, cv_beauty, video_width, video_height, beauty_value, rtmp_rul, vedio_option_control);
		});
	vedio_thread.detach();

	cv_beauty.wait(lck);

	return vedio_option_control;
}

Cvedio_option_control* virtual_background_start(int video_width, int video_height, const char* mask_url, const char* url_ip, short url_port, const char* back_img_file, const char* rtmp_rul)
{
	Cvedio_option_control* vedio_option_control = nullptr;
	std::mutex mtx_virtual;
	std::condition_variable cv_virtual;

	std::unique_lock <std::mutex> lck(mtx_virtual);

	std::thread vedio_thread([&vedio_option_control, &mtx_virtual, &cv_virtual, video_width, video_height, mask_url, url_ip, url_port, back_img_file, rtmp_rul]()
		{
			opencv_virtual_backimg(mtx_virtual, cv_virtual, video_width, video_height, mask_url, url_ip, url_port, back_img_file, rtmp_rul, vedio_option_control);
		});
	vedio_thread.detach();

	cv_virtual.wait(lck);

	return vedio_option_control;
}

void beauty_stop(Cvedio_option_control* vedio_option_control)
{
	if (nullptr != vedio_option_control)
	{
		vedio_option_control->curr_run_state = false;
	}
}

void virtual_background_stop(Cvedio_option_control* vedio_option_control)
{
	if (nullptr != vedio_option_control)
	{
		vedio_option_control->curr_run_state = false;
	}
}
