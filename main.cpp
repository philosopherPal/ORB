#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/ml/ml.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>


using namespace cv;
using namespace std;
#define DICTIONARY_BUILD 0

int main()
{
#if DICTIONARY_BUILD == 1
    DIR *directory;
    struct dirent *dir_info;
    struct stat file_st;
    //FileStorage fs;
    string file_path;
    int debug_error=1;
    vector<KeyPoint> keypoints;
    int64 t0=cv::getTickCount();
    Mat descriptor;
    Mat Desc_Unclustered(0,0,CV_32F);
    OrbDescriptorExtractor detector;
    string image_path = "/home/pallavi/101_ObjectCategories/train1";
    Mat input;
    directory = opendir(image_path.c_str());
    int m=1;
    while((dir_info = readdir(directory))) {
        file_path = image_path+"/"+dir_info->d_name;
        if (stat( file_path.c_str(), &file_st )) continue;
        if (S_ISDIR( file_st.st_mode )) continue;
        input = imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);
        if(input.empty()){
            cout<<"empty\n";
           }
        if(debug_error){
            cout<<file_path<<endl;
            cout<<m++<<endl;
           }
           //detect feature points
           detector.detect(input, keypoints);
           //compute the descriptors for each keypoint
           detector.compute(input, keypoints,descriptor);
           Desc_Unclustered.push_back(descriptor);
        }
    closedir(directory);
    image_path = "/home/pallavi/101_ObjectCategories/train2";
    directory = opendir(image_path.c_str());
    while((dir_info = readdir(directory))) {
       file_path = image_path+"/"+dir_info->d_name;
       if (stat( file_path.c_str(), &file_st )) continue;
       if (S_ISDIR( file_st.st_mode )) continue;
       input = imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);
       if(input.empty()){
          cout<<"empty\n";
       }
       if(debug_error){
           cout<<file_path<<endl;
           cout<<m++<<endl;
       }
       //detect feature points
       detector.detect(input, keypoints);
       //compute the descriptors for each keypoint
       detector.compute(input, keypoints,descriptor);
       Desc_Unclustered.push_back(descriptor);

        }
    closedir(directory);
    int64 t1=cv::getTickCount();
    double secs=(t1-t0)/cv::getTickFrequency();
    cout<<"\n"<<secs<<"\n";
    //the number of bags
    int dictionarySize=160;
    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
       //retries number
    int retries=1;
       //necessary flags
    int flags=KMEANS_RANDOM_CENTERS;
       //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
       //convert Desc_Unclustered to type CV_32F
    Mat Desc_UnclusteredF(Desc_Unclustered.rows,Desc_Unclustered.cols,CV_32F);
    Desc_Unclustered.convertTo(Desc_UnclusteredF,CV_32F);
    bowTrainer.add(Desc_UnclusteredF);
    cout << "Calculating vocabulary for "<<bowTrainer.descripotorsCount()<<" descriptors "<<endl;
    //cluster the feature vectors
    Mat dictionary=bowTrainer.cluster(Desc_UnclusteredF);
    //store the vocabulary
    FileStorage fs("voc.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();
    cout<<"Building vocabulary\n";
    cout<<"saving vocab..........";
#else

    int dictionarySize=160;
    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    //retries number
    int retries=1;
    //necessary flags
    int flags=KMEANS_RANDOM_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    DIR *directory1;
    struct dirent *dir_info1;
    struct stat file_st1;
    string file_path1;
    string image_path1 = "/home/pallavi/101_ObjectCategories/train1";
    int debug_error1=1;
    Mat dictionaryF;
    FileStorage fs("voc.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionaryF;
    fs.release();
    //convert to 8bit unsigned format
    Mat dictionary(dictionaryF.rows,dictionaryF.cols,CV_8U);
    dictionaryF.convertTo(dictionary,CV_8U);
    //create a matcher with BruteForce-Hamming distance
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<FeatureDetector> detector(new OrbFeatureDetector());
    Ptr<DescriptorExtractor> extractor(new OrbDescriptorExtractor());
    // BoF  descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary in the first step
    bowDE.setVocabulary(dictionary);
    FileStorage fs1("descriptor1.yml", FileStorage::WRITE);
        //cout<<"here\n";
    Mat input1;
    directory1 = opendir(image_path1.c_str());
    vector<KeyPoint> keypoints;
    Mat bowDescriptor;
    int m1=1;
    //cout<<"here\n";
    while((dir_info1 = readdir(directory1))) {
        file_path1 = image_path1+"/"+dir_info1->d_name;
        if (stat( file_path1.c_str(), &file_st1 )) continue;
        if (S_ISDIR( file_st1.st_mode )) continue;
        input1 = imread(file_path1, CV_LOAD_IMAGE_GRAYSCALE);
        if(input1.empty()){
            cout<<"empty\n";
          }
        if(debug_error1){
            cout<<file_path1<<endl;
            cout<<m1++<<endl;
          }
        detector->detect(input1,keypoints);
        bowDE.compute(input1,keypoints,bowDescriptor);
      //cout<<"here";
        fs1<<"descriptors"<<bowDescriptor;
        }
    image_path1="/home/pallavi/101_ObjectCategories/train2";
    directory1 = opendir(image_path1.c_str());
    while((dir_info1 = readdir(directory1))) {
        file_path1 = image_path1+"/"+dir_info1->d_name;
        if (stat( file_path1.c_str(), &file_st1 )) continue;
        if (S_ISDIR( file_st1.st_mode )) continue;
        input1 = imread(file_path1, CV_LOAD_IMAGE_GRAYSCALE);
        if(input1.empty()){
            cout<<"empty\n";
          }
        if(debug_error1){
            cout<<file_path1<<endl;
            cout<<m1++<<endl;
          }
        detector->detect(input1,keypoints);
        bowDE.compute(input1,keypoints,bowDescriptor);
        //cout<<"here";
        fs1<<"descriptors"<<bowDescriptor;
        }
    fs1.release();
    Mat groundTruth(0, 1, CV_32FC1);
    Mat labels(0, 1, CV_32FC1);
    Mat trainingData(0, dictionarySize, CV_32FC1);
    Mat img;
    int m2=1;
    int j=1;
    DIR *directory2;
    struct dirent *dir_info2;
    struct stat file_st2;
    string file_path2;
    Mat desc1;
    int debug_error2=1;
    vector<KeyPoint> kp1;
    string image_path2="/home/pallavi/101_ObjectCategories/train1";
    directory2 = opendir(image_path2.c_str());
    while((dir_info2 = readdir(directory2))) {
        file_path2 = image_path2+"/"+dir_info2->d_name;
        if (stat( file_path2.c_str(), &file_st2 )) continue;
        if (S_ISDIR( file_st2.st_mode )) continue;
        img= imread(file_path2, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty()){
            cout<<"empty\n";
        }
        if(debug_error1){
            cout<<file_path2<<endl;
            cout<<m2++<<endl;
        }
    detector->detect(img,kp1);
    bowDE.compute(img,kp1,desc1);
    //cout<<"here";
    trainingData.push_back(desc1);
    labels.push_back((float) j);
    }
    int l=2;
    image_path2="/home/pallavi/101_ObjectCategories/train2";
    directory2 = opendir(image_path2.c_str());
    while((dir_info2 = readdir(directory2))) {
    file_path2= image_path2+"/"+dir_info2->d_name;
    if (stat( file_path2.c_str(), &file_st2 )) continue;
    if (S_ISDIR( file_st2.st_mode )) continue;
    img= imread(file_path2, CV_LOAD_IMAGE_GRAYSCALE);
    if(img.empty()){
       cout<<"empty\n";
    }
    if(debug_error1){
        cout<<file_path2<<endl;
        cout<<m2++<<endl;
    }
    detector->detect(img,kp1);
    bowDE.compute(img,kp1,desc1);
    //cout<<"here";
    trainingData.push_back(desc1);
    labels.push_back((float) l);
    }


    for(int i=0; i<labels.rows; i++)
        for(int j=0; j<labels.cols; j++)
            printf("labels(%d, %d) = %f n", i, j, labels.at<float>(i,j));
    CvSVMParams params;
    params.kernel_type = CvSVM::RBF;
    params.svm_type = CvSVM::C_SVC;
    params.gamma = 0.50625000000000009;
// 0.50625000000000009;
    params.C = 312.50000000000000;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
    CvSVM svm;
    printf("%s\n", "Training SVM classifier");
    bool res = svm.train(trainingData, labels, cv::Mat(),cv::Mat(), params);
    cout<<"Processing evaluation data..."<<endl;
    Mat evalData(0, dictionarySize, CV_32FC1);
    vector<KeyPoint> kp2;
    int m3=1;
    Mat desc2;
    Mat results(0, 1, CV_32FC1);
    DIR *directory4;
    struct dirent *dir_info4;
    struct stat file_st4;
    string file_path4;
    string image_path4="/home/pallavi/101_ObjectCategories/test1";
    directory4 = opendir(image_path4.c_str());
    while((dir_info4 = readdir(directory4))){
        file_path4= image_path4+"/"+dir_info4->d_name;
        if (stat( file_path4.c_str(), &file_st4 )) continue;
        if (S_ISDIR( file_st4.st_mode )) continue;
        img= imread(file_path4, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty()){
           cout<<"empty\n";
           break;
        }
          if(debug_error2){
              //cout<<file_path4<<endl;
            cout<<m3++<<endl;
        }
        detector->detect(img,kp2);
        bowDE.compute(img,kp2,desc2);
        //cout<<"here";
        evalData.push_back(desc2);
        float response = svm.predict(desc2);
        cout<<file_path4<<" ";
        cout<<response<<"\n";
        results.push_back(response);
        groundTruth.push_back((float) j);
    }

    image_path4="/home/pallavi/101_ObjectCategories/test2";
    directory4 = opendir(image_path4.c_str());
    while((dir_info4 = readdir(directory4))){
        file_path4= image_path4+"/"+dir_info4->d_name;
        if (stat( file_path4.c_str(), &file_st4 )) continue;
        if (S_ISDIR( file_st4.st_mode )) continue;
        img= imread(file_path4, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty()){
        cout<<"empty\n";
        break;
        }
        if(debug_error2){
        //cout<<file_path4<<endl;
        cout<<m3++<<endl;
        }
        detector->detect(img,kp2);
        bowDE.compute(img,kp2,desc2);
        //cout<<"here";
        evalData.push_back(desc2);
        float response = svm.predict(desc2);
        cout<<file_path4<<" ";
        cout<<response<<"\n";
        results.push_back(response);
        groundTruth.push_back((float) l);
    }
    float A=0,AnB=0;
    for(int i=0; i<results.rows; i++){
        for(int j=0; j<results.cols; j++)
            {
            printf("results(%d, %d) = %f n", i, j, results.at<float>(i,j));
            if(results.at<float>(i,j)==1)
                AnB++;
            if(results.at<float>(i,j)==1)
                if(results.at<float>(i,j) == groundTruth.at<float>(i,j))
                    A++;
            }
            printf("\n");}
    double errorRate = (double) countNonZero(groundTruth- results) / evalData.rows;
    cout<<"\n"<<errorRate<<"\n";
    cout<<"A="<<A<<"\n";
    cout<<"A+B="<<AnB<<"\n";
    float P = (A/(AnB));
    cout<<"P="<<P<<"\n";
    float R = (A/448);
    cout<<"R="<<R<<"\n";
#endif
     printf("\ndone\n");
return 0;
}


