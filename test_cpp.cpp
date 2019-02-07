#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

int main(int argc, char **argv)
{
  cv::Mat ori = cv::imread("bd_projet_scia/damier/G0010391.JPG", cv::IMREAD_COLOR);
  cv::Mat gray;
  cv::Mat img;
  auto height = ori.size().height;
  auto width = ori.size().width;
  auto size = 30;
  cv::resize(ori, img, cv::Size(height * size / 100, width * size / 100));

  cv::imshow("Image", img);


  cv::cvtColor(img, gray, CV_BGR2GRAY);
  cv::Size patternsize(8, 9);
  cv::Size patternsize2(9, 8);
  std::vector<cv::Point2f> corners;

  // SEARCH FOR A CHESSBOARD
  bool patternfound = findChessboardCorners(gray, patternsize, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
      + cv::CALIB_CB_FAST_CHECK);

  std::cout << "CHESSBOARD found : " << patternfound << std::endl;
  std::cout << "  coordonates found : " << corners << std::endl;

  if (patternfound)
    cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

  std::cout << "============================================" << std::endl;
  std::cout << "  coordonates found : " << corners << std::endl;

  drawChessboardCorners(img, patternsize, cv::Mat(corners), patternfound);

  cv::imwrite("IMAGE.jpg", img);

  cv::imshow("Image", img);
  cv::waitKey(0);
  cv::destroyWindow("Image");
  return 0;
}
