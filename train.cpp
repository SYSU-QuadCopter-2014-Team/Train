#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

vector<float> getHog(Mat test) {

	resize(test, test, Size(64, 64));

	//winsize, blocksize, blockstride, cellsize, nbins
	HOGDescriptor hog(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
	vector<float>descriptors;//结果数组       
	hog.compute(test, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算   
	
	// for (int i = 0 ; i< descriptors.size(); i++)
	// 	descriptors[i] = descriptors[i] * 1e5;
	
	cout << "Descriptor Dim " << descriptors.size() << endl;
	return descriptors;  
	
}

int main()
{   

    int sample_count = 699, dim = 1764;
    // cout << "Input samples, dim: ";
    // cin >> sample_count >> dim;
    int* labels = new int[sample_count];
    float** trainingData = new float*[sample_count];
    for (int i = 0; i < sample_count; i++){

        trainingData[i] = new float[dim];
		// printf("TraningData: %d\n", i);
			
    }
    FILE *file;
    file = fopen("./train.txt", "r");
    for (int i = 0; i < sample_count; i++) {
        // trainingData[i] = new float[dim];
        fscanf(file, "%d", &labels[i]);
        for (int j = 0; j < dim; j++){

            fscanf(file, "%f", &trainingData[i][j]);
			
        }
		// if (i == 0)
		// 	for (int k = 0; k < dim; k++)
		// 		printf("%f ", trainingData[0][k]);
		
		// printf("TraningData: %d\n", i);
			
    }

    Mat labelsMat(sample_count, 1, CV_32SC1, labels);
    Mat trainingDataMat(sample_count, dim, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;

    // 训练分类器
		CvSVM model;
    model.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    model.save("SVMModel.xml");

		model.load("SVMModel.xml");

	Mat t = imread("./test.jpg");
	
	vector<float> hog = getHog(t);
	
	float* data = hog.data();
	
	Mat sampleMat(1, dim, CV_32FC1, data);
	float response = model.predict(sampleMat);

	cout << response << endl;
		

    delete []labels;
    for (int i = 0; i < sample_count; i++)
        delete []trainingData[i];
    delete []trainingData;
    return 0;
}








