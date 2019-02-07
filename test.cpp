#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

int main(int argc, char **argv)
{
  cv::Mat ori = cv::imread("bd_projet_scia/damier/G0010566.JPG", cv::IMREAD_COLOR);
  auto gray = ori.clone(); 
  auto img = ori.clone();
  cv::cvtColor(ori, gray, CV_BGR2GRAY);
  cv::Size patternsize(8, 9);
  std::vector<cv::Point2f> corners;

  bool patternfound = findChessboardCorners(ori, patternsize, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
      + cv::CALIB_CB_FAST_CHECK);

  std::cout << "CHESSBOARD found : " << patternfound << std::endl;

  //if (patternfound)
  //  cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
  //      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

  drawChessboardCorners(img, patternsize, cv::Mat(corners), patternfound);

  cv::imwrite("IMAGE.jpg", img);

  cv::imshow("Image", img);
  cv::waitKey(0);
  cv::destroyWindow("Image");
  return 0;
}
