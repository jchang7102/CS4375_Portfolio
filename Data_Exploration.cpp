#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

//Find the sum of a vector
double findSum(vector<double> v) {
    double sum = 0.0;

    //Loop through the vector and add each value to a total sum
    for(int i = 0; i < v.size(); i++) {
        sum = sum + v[i];
    }

    return sum;
}

//Find the mean of a vector
double findMean(vector<double> v) {
    double sum = findSum(v);
    int length = v.size();
    //Calculate mean by dividing the sum by the length
    double mean = sum/length;

    return mean;
}

//Find the median of a vector
double findMedian(vector<double> v) {
    int length = v.size();
    double median;

    //Sort the vector
    sort(v.begin(), v.end());

    //If the vector has an even number of values, the median is the average of the middle two values
    if(length % 2 == 0) {
        median = (v[length/2 - 1] + v[length/2]) / 2;
    }
    //Otherwise, the median is the middle value
    else {
        median = v[length/2];
    }

    return median;
}

//Find the range of a vector
double findRange(vector<double> v) {
    double min = 100.0;
    double max = 0.0;
    double range;

    //Loop through all the values of the vector
    for(int i = 0; i < v.size(); i++) {
        //If the value is greater than the current max, make it the new max
        if(v[i] > max) {
            max = v[i];
        }

        //If the value is smaller than the current min, make it the new min
        if(v[i] < min) {
            min = v[i];
        }
    }
    range = max - min;

    return range;
}

//Compute the covariance between two vectors
double computeCovariance(vector<double> u, vector<double> v) {
    double uMean = findMean(u);
    double vMean = findMean(v);
    double summation = 0.0;

    //Calculate the summation part of the covariance formula
    for(int i = 0; i < u.size(); i++) {
        summation = summation + ((u[i] - uMean) * (v[i] - vMean));
    }

    double denom = u.size() - 1;
    double cov = summation / denom;

    return cov;
}

//Compute the correlation between two vectors
double computeCorrelation(vector<double> u, vector<double> v) {
    //Calculate the sigma of a vector as square root of variance(u,u)
    double uSigma = sqrt(computeCovariance(u,u));
    double vSigma = sqrt(computeCovariance(v,v));

    double cov = computeCovariance(u,v);
    //Calculate correlation by dividing covariance by uSigma * vSigma
    double cor = cov / (uSigma * vSigma);

    return cor;
}


int main(int argc, char** argv) {
    ifstream inFS;      //Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    //Try to open file
    cout << "Opening file Boston.csv" << endl;

    inFS.open("Boston.csv");
    if(!inFS.is_open()) {
        cout << "Could not open file Boston.csv" << endl;
        return 1; //1 indicates error
    }

    //Can now use inFS stream like cin stream
    //Boston.csv should contain two doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    //echo heading
    cout <<"heading: " << line << endl;

    int numObservations = 0;
    while(inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    cout << "Closing file Boston.csv" << endl;
    inFS.close(); //Done with file, so close it

    cout << "Number of records: " << numObservations << endl;

    cout << "\n*****Stats for rm:*****" << endl;
    cout << "Sum of rm: " << findSum(rm) << endl;
    cout << "Mean of rm: " << findMean(rm) << endl;
    cout << "Median of rm: " << findMedian(rm) << endl;
    cout << "Range of rm: " << findRange(rm) << endl;

    cout << "\n*****Stats for medv:*****" << endl;
    cout << "Sum of medv: " << findSum(medv) << endl;
    cout << "Mean of medv: " << findMean(medv) << endl;
    cout << "Median of medv: " << findMedian(medv) << endl;
    cout << "Range of medv: " << findRange(medv) << endl;

    cout << "\nCovariance = " << computeCovariance(rm, medv) << endl;

    cout << "\nCorrelation = " << computeCorrelation(rm, medv) << endl;

    cout << "\nProgram terminated.";

    return 0;
}