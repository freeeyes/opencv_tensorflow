// viewcamera.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "opencv_camera.h"

int main()
{
    Cvedio_option_control* vedio_option_control = nullptr;
	//dnn::Net net = cv::dnn::readNetFromTensorflow("");

    //opencv_virtual_backimg(320, 240, "http://3.15.16.242:9000", "3.15.16.242", 9000, "e.jpg", "");
    //opencv_virtual_backimg(320, 240, "http://localhost:9000", "127.0.0.1", 9000, "e.jpg", "");

    //opencv_camera_to_rtmp(640, 480, "rtmp://172.18.231.245:1935/live/test");
	//opencv_camera_to_rtmp(320, 240, "");
   
    //opencv_beauty(320, 240, 20, "rtmp://172.18.231.245:1935/live/test", vedio_option_control);

    vedio_option_control = beauty_start(320, 240, 15, "rtmp://172.18.231.245:1935/live/test");

	/*
	std::mutex mtx_beauty;
	std::condition_variable cv_beauty;

	std::unique_lock <std::mutex> lck(mtx_beauty);

	std::thread test([&mtx_beauty, &cv_beauty]() {
		{
			std::unique_lock <std::mutex> lck(mtx_beauty);

			std::cout << "thread" << std::endl;
			cv_beauty.notify_one();
		}

		std::this_thread::sleep_for(std::chrono::seconds(10));
		std::cout << "thread end" << std::endl;
		});

	cv_beauty.wait(lck);
	std::cout << "main" << std::endl;
	*/

    getchar();
    return 0;
}
