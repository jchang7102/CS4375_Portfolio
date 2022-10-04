//Jeffrey Li & Valari Graham
//CS 4375
//Portfolio: ML from Scratch

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;


//Calculates the dot product of two vectors (only used in sigmoid function)
double dotProduct(vector<double> u, vector<double> v) {
    double sum = 0;
    for(int i = 0; i < u.size(); i++) {
        sum += u[i] * v[i];
    }

    return sum;
}

//Calculates the product of a vector with a scalar
vector<double> matrixMultiplication(vector<double> u, double scalar) {
    vector<double> output(u.size());
    for(int i = 0; i < u.size(); i++) {
        output[i] = scalar * u[i];
    }

    return output;
}

//Calculates the product of a matrix with a vector
vector<double> matrixMultiplication(vector<vector<double>> &matrix, vector<double> u) {
    vector<double> output(matrix.size());
    for(int i = 0; i < matrix.size(); i++) {
        double sum = 0;
        for(int j = 0; j < u.size(); j++) {
            sum += matrix[i][j] * u[j];
        }

        output[i] = sum;
    }

    return output;
}

//Calculates vector a + vector b
vector<double> vectorAddition(vector<double> u, vector<double> v) {
    vector<double> output(u.size());
    for(int i = 0; i < u.size(); i++) {
        output[i] = u[i] + v[i];
    }

    return output;
}

//Calculates vector a - vector b
vector<double> vectorSubtraction(vector<double> u, vector<double> v) {
    vector<double> output(u.size());
    for(int i = 0; i < u.size(); i++) {
        output[i] = u[i] - v[i];
    }

    return output;
}

//Calculates a probability vector based on input matrix and weights
vector<double> sigmoid(vector<vector<double>> input, const vector<double> &weights) {
    vector<double> output(input.size());
    for(int i = 0; i < input.size(); i++) {
        output[i] = 1 / (1 + exp(-1 * dotProduct(input[i], weights)));
    }

    return output;
}

//Transposes a matrix
vector<vector<double>> transpose(vector<vector<double>> input) {
    vector<vector<double>> output(input[0].size(), vector<double>(input.size()));
    for(int i = 0; i < input.size(); i++) {
        for(int j = 0; j < input[i].size(); j++) {
            output[j][i] = input[i][j];
        }
    }

    return output;
}

//Calculates optimal weights
vector<double> gradientDescent(const vector<vector<double>> &matrix, const vector<double> &labels) {
    double learning_rate = 0.001;
    vector<double> weights{1,1};
    vector<double> prob_vector;
    vector<double> error;
    vector<double> transError_vector;
    vector<double> scaledtransError_vector;
    vector<vector<double>> transposedMatrix = transpose(matrix);

    for(int i = 0; i < 500; i++) {
        //prob_vector = sigmoid(data_matrix %*% weights)
        prob_vector = sigmoid(matrix, weights);
        //error = labels - prob_vector
        error = vectorSubtraction(labels, prob_vector);
        //transError_vector = t(data_matrix) %*% error
        transError_vector = matrixMultiplication(transposedMatrix, error);
        //scaledtransError_vector = learning_rate * transError_vector
        scaledtransError_vector = matrixMultiplication(transError_vector, learning_rate);
        //weights = weights + scaledtransError_vector
        weights = vectorAddition(weights, scaledtransError_vector);
    }

    return weights;
}

//Logistic regression that returns optimal weights
vector<double> logisticRegression(vector<vector<double>> matrix, vector<int> index, int target) {
    vector<double> labels(matrix.size());
    vector<vector<double>> dataMatrix(matrix.size(), vector<double>(index.size() + 1, 1));

    for(int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < index.size(); j++) {
            dataMatrix[i][j + 1] = matrix[i][index[j]];
        }
        labels[i] = matrix[i][target];
    }

    return gradientDescent(dataMatrix, labels);
}

//Calculates prediction vector
vector<double> predict(vector<vector<double>> matrix, vector<int> index, vector<double> weights) {
    vector<vector<double>> dataMatrix(matrix.size(), vector<double>(index.size() + 1, 1));
    for(int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < index.size(); j++) {
            dataMatrix[i][j + 1] = matrix[i][index[j]];
        }
    }

    vector<double> logOdds = matrixMultiplication(dataMatrix, weights);
    vector<double> output(logOdds.size());
    //Turn logOdds into probabilities with formula exp(logOdds) / (1 + exp(logOdds))
    for(int i = 0; i < logOdds.size(); i++) {
        output[i] = exp(logOdds[i]) / (1 + exp(logOdds[i]));
    }

    return output;
}

//Calculates confusion matrix based on actual and predicted values
//Confusion matrix looks like:
//             Actual
// Predicted  TP    FP
//            FN    TN
vector<int> confusionMatrix(vector<double> actual, vector<double> predicted) {
    int truePositive = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    int trueNegative = 0;

    for(int i = 0; i < actual.size(); i++) {
        if(actual[i] == 1 && predicted[i] >= 0.5) {
            truePositive++;
        }
        else if(actual[i] == 0 && predicted[i] >= 0.5) {
            falsePositive++;
        }
        else if(actual[i] == 1 && predicted[i] < 0.5) {
            falseNegative++;
        }
        else {
            trueNegative++;
        }
    }

    return {truePositive, falsePositive, falseNegative, trueNegative};
}


int main(int argc, char **argv) {
    ifstream inFS;
    string data;
    string column;
    stringstream data_stream(data);
    vector<vector<double>> trainingMatrix;
    vector<vector<double>> testingMatrix;

    inFS.open("titanic_project.csv");
    cout << "Reading header" << endl;
    getline(inFS, data);
    cout <<"Header: " << data << endl << endl;

    int numObservations = 0;
    while(inFS.good()) {
        vector<double> row;
        getline(inFS, data);
        data_stream = stringstream(data);

        //Skip the first column
        getline(data_stream, column, ',');

        //Read data into rows
        while(data_stream.good()) {
            getline(data_stream, column, ',');
            row.push_back(stoi(column));
        }

        //First 800 rows used for training
        if(numObservations < 800) {
            trainingMatrix.push_back(row);
        }
        //Rest used for testing
        else {
            testingMatrix.push_back(row);
        }

        numObservations++;
    }

    inFS.close(); //Done with file, so close it


    //Perform the logistic regression and time it
    vector<int> index = {2};
    auto start = system_clock::now();
    vector<double> logReg = logisticRegression(trainingMatrix, index, 1);
    auto stop = system_clock::now();
    ::duration<double> duration = stop - start;


    //Output the coefficients
    cout << "Intercept coefficient: " << logReg[0] << endl;
    cout << "Sex coefficient:      " << logReg[1] << endl << endl;


    //Predict with the model
    vector<double> predictions = predict(testingMatrix, index, logReg);

    //Compute the confusion matrix
    vector<double> confColumn(testingMatrix.size());
    for(int i = 0; i < testingMatrix.size(); i++) {
        confColumn[i] = testingMatrix[i][1];
    }
    vector<int> confusionM = confusionMatrix(confColumn, predictions);

    //Output the metrics
    //Accuracy is calculated by (TP + TN) / (TP + TN + FP + FN)
    cout << "Accuracy:    " << (confusionM[0] + confusionM[3]) / (double) (confusionM[0] + confusionM[1] + confusionM[2] + confusionM[3]) << endl;

    //Sensitivity is calculated by TP / (TP + FN)
    cout << "Sensitivity: " << confusionM[0] / (double) (confusionM[0] + confusionM[2]) << endl;

    //Specificity is calculated by TN / (TN + FP)
    cout << "Specificity: " << confusionM[3] / (double) (confusionM[3] + confusionM[1]) << endl << endl;


    //Output the algorithm run time
    cout << "Algorithm run time: " << duration.count() << "s" << endl;


    return 0;
}