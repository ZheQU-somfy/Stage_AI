#include <iostream>
#include <vector>
#include <fstream>
#include </home/zhe/install-opencv/include/opencv4/opencv2/core/core.hpp>
#include </home/zhe/install-opencv/include/opencv4/opencv2/dnn/dnn.hpp>
#include </home/zhe/install-opencv/include/opencv4/opencv2/imgproc/imgproc.hpp>
#include </home/zhe/install-opencv/include/opencv4/opencv2/highgui/highgui.hpp>
#include </home/zhe/install-opencv/include/opencv4/opencv2/core/types.hpp>

using namespace std;

int readClassNamesFromFile(string fileName, vector<string>& classNames){
    ifstream fp;
    fp.open(fileName);
    if(!fp.is_open()){
        cout<<"can't open file"<<fileName<<endl;
        return -1;
    }

    string name;
    while(!fp.eof()){
        getline(fp,name);
        if(name.length()){
            classNames.push_back(name);
        }
    }
    fp.close();
    return 0;
}
int drawPrediction(const vector<string>& labelNames, int classId, float conf, int left, int top, int right, int bottom, cv::Mat& img)
{
	rectangle(img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 2);
 
 
	if (labelNames.empty())
	{
		cout << "labelNames is empty!" << endl;
		return -1;
	}
	if (classId >= (int)labelNames.size())
	{
		cout << "classId is out of boundary!" << endl;
		return -1;
	}
 
	string label = labelNames[classId];
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1);
 
	return 0;
}

void postprocess(const std::vector<string>& labelNames, cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, float thresh, float nms)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > thresh)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, thresh, nms, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            drawPrediction(labelNames,classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
        }
    }


}

int main()
{
    std::string modelPath = "/home/zhe/opencv_study/yolov3-tiny_xnor_last.weights";
    std::string configPath = "/home/zhe/opencv_study/yolov3-tiny_xnor.cfg";
    std::string labelPath = "coco.names";
    std::string imagePath = "lena.jpg";
    int networkW = 416;
    int networkH = 416;
    float thresh = 0.5;
    float nms = 0.45;
 
    cv::dnn::Net net= cv::dnn::readNet(modelPath , configPath );
    vector<string> labels;
    int err = readClassNamesFromFile(labelPath , labels);
 
    std::vector<string> outNames = net.getUnconnectedOutLayersNames();
 
    cv::Mat srcImg = cv::imread(imagePath);
    cv::Mat inputBlob = cv::dnn::blobFromImage(srcImg, 1.0/255, cv::Size(networkW, networkH), cv::Scalar(), false, false);
 
    net.setInput(inputBlob);
    vector<cv::Mat> outs;
    net.forward(outs, outNames);
 
    postprocess(labels,srcImg, outs, net, thresh, nms);
 
    return 0;
}
