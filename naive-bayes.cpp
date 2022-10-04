// Valari Graham & Jeffrey Li
// CS 4375 - Dr. Mazidi 

// Portfolio: ML from Scratch 

#include <chrono>
#include <iostream> // input, output stream 
#include <fstream> // file read in 
#include <string>
#include <cmath> // for pi 
#include <vector> // arrays that can change in size 

using namespace std;  

// Displays the results of accuracy, sensitivity, and specificity
void results(double test[], double data[]){
    // Predicition step 
    int probabilities[246];

    // Sort probabilities based on prediction results 
    for(int i = 0; i < 246; i++){
        if(data[i] > .5)
            probabilities[i] = 1;
        else
            probabilities[i] = 0;
    }

    // Accuracy count 
    double sum = 0;
    for(int i = 0; i < 246; i++)
        if(probabilities[i] == test[i])
            sum++;

     // Specificity & Sensitivity
    double TN = 0, FP = 0, TP = 0, FN = 0;;
    for(int i = 0; i<246; i++){
        // True Negative  
        if(probabilities[i] == 0 && test[i] == 0)
            TN++;
        // True Positive 
        if(probabilities[i] == 1 && test[i] == 1)
            TP++;
        // False Negative 
        if(probabilities[i] == 0 && test[i] == 1)
            FN++;
        // False Positive 
        if(probabilities[i] == 1 && test[i] == 0)
            FP++;
    }

    // Specificity equation 
    double specificity = (TN/(TN+FP));
    // Sensitivity equation 
    double sensitivity = (TP/(TP+FN));
    // Accuracy equation 
    double accuracy = ((TP+TN))/(TP+TN+FP+FN);

    // Printing results 
    cout << "\n" << "\t" << "RESULTS" << endl; 
    cout << "------------------------" << endl; 

    cout << "Accuracy: " << accuracy << endl;
    cout << "\nSensitivity: " << sensitivity << endl; 
    cout << "\nSpecificity: " << specificity << endl; 

    return; 
}

int main () { 

// Read in titanic.csv 
ifstream inFS; 
string header; 
string id_in, pclass_in, survived_in, sex_in, age_in; 
const int MAX_LEN = 2000; 

vector<int> pclass(MAX_LEN);
vector<int> survived(MAX_LEN);
vector<int> sex(MAX_LEN);
vector<double> age(MAX_LEN);

cout << "Opening file titanic.csv" << endl; 

// Open titanic.csv file 
inFS.open("titanic.csv");
if(!inFS.is_open()) {
    cout << "Could not oepn file titanic.csv" << endl; 
    return 1; 
}

// Read in header 
getline(inFS, header); 

// Read in observations
int numObservations = 0; 
while (inFS.good()) {
    getline(inFS, id_in, ',');
    getline(inFS, pclass_in, ',');
    getline(inFS, survived_in, ',');
    getline(inFS, sex_in, ',');
    getline(inFS, age_in, '\n');

    // String to float and double conversion 
    pclass.at(numObservations) = stof(pclass_in); 
    survived.at(numObservations) = stof(survived_in); 
    sex.at(numObservations) = stof(sex_in);
    age.at(numObservations) = stof(age_in);  

    // Count observations
    numObservations++; 
    
 } 

// Resize the container
pclass.resize(numObservations);
survived.resize(numObservations);
sex.resize(numObservations);
age.resize(numObservations);

// Close file 
cout << "Closing titanic.csv\n" << endl; 
inFS.close(); 

// Split data into testing and training data 
double train[800][4];
double test[246][4];
double trainTarget[800];
double testTarget[246];

for(int i = 0; i < 1046; i++){
    // First 800 observations for training
    if(i < 800){
        train[i][0] = 1;
        train[i][1] = age[i];
        train[i][2] = pclass[i];
        train[i][3] = sex[i];
        trainTarget[i] = survived[i];
    }
    // Rest of the data for testing 
    else{
        test[i - 800][0] = 1;
        test[i - 800][1] = age[i];
        test[i - 800][2] = pclass[i];
        test[i - 800][3] = sex[i];
        testTarget[i - 800] = survived[i];
    }

}

// Begin run time clock  
chrono::time_point<std::chrono::system_clock> begin, end; 
begin = chrono::system_clock::now(); 

// Calculate priors
    double total = 0.0;
    for(int i = 0; i < 800; i++){
        if(trainTarget[i] == 1)
            total += 1.0;
    }
    // Probability of an event before new data is collected 
    double calcPrior[] = {(800 - total) / 800.00, total / 800.00};
    double survive[] = {(800 - total), total};
    
   // Calculate the probability of pclass using training data and train target variable (survived)
    double pc[2][3];
    for(int x = 0; x < 2; x++)
        for(int p = 0; p < 3; p++){
            int nrow = 0;
            for(int i = 0; i < 800; i++){
                if(train[i][2] == p + 1 && trainTarget[i] == x)
                    nrow++;
            pc[x][p] = nrow / survive[x];
            }
        }

    // Calculate the probability of sex using training data and train target variable (survived)
    double sx[2][2];
    for(int x = 0; x < 2; x++)
        for(int p = 0; p < 2; p++){
            int nrow = 0;
            for(int i = 0; i < 800; i++){
                if(train[i][3] == p && trainTarget[i] == x)
                    nrow++;
            sx[x][p] = nrow / survive[x];
            }
        }

    // Survivabilty given pclass output 
    cout << "Survivability for pclass:  " << endl; 
    cout << "1\t\t" << "2\t\t" << "3\t" << endl; 

    // P(target|pclass)
    for(int r = 0; r < 2; r++){
        for(int c = 0; c < 3; c++)
            cout << pc[r][c] << '\t';
        cout << endl;
    }

    // Survivability given sex output 
    cout << "Survivability for sex:  " << endl; 
     cout << "0\t\t" << "1\t\t" << endl; 

    // P(target|sex)
    for(int r = 0; r < 2; r++){
        for(int c = 0; c < 2; c++)
            cout << sx[r][c] << '\t'; 
        cout << endl; 
    }

    // Calculate the mean given the training data 
    double mean[2];
    double m = 0;
    double m1 = 0;

    for(int x = 0; x < 2; x++){
        for(int i = 0; i < 800; i++)
            if(trainTarget[i] == x){
                mean[x] += train[i][1];
                if(x == 0)
                    m++;
                else
                    m1++;
            }
    }

    mean[0] /= m;
    mean[1] /= m1;

    // Calculate the variance given the training data using the mean 
    double var[2];
    for(int x = 0; x < 2; x++){
        for(int i = 0; i < 800; i++)
            if(trainTarget[i] == x){
                var[x] += ((mean[x] - train[i][1])*(mean[x] - train[i][1]));
            }
    }

    var[0] /= (m - 1);
    var[1] /= (m1 - 1);

    // Calculates the probabilites of each predictor using testing data
    double data[246];

    for(int i = 0; i < 246; i++){
    double num = test[i][1];
    int p = test[i][2] - 1; 
    int s = test[i][3]; 
   
    // Calculate probabiity of age from mean and variance 
    double a = 1 / (sqrt(2 * M_PI * var[1])) * exp(-((num-mean[1])*(num-mean[1])) / (2 * var[1]));

    // Numerator equation:
    double numerator = pc[1][p] * sx[1][s] * calcPrior[1] * a;

    // Denominator equation:
    double denominator = pc[1][p] * sx[1][s] * calcPrior[1] * a + pc[0][p] * sx[0][s] * calcPrior[0] * a;
    
    // Probability equation 
    data[i] = numerator / denominator;
    }

    // End run time clock 
    end = chrono::system_clock::now();
    chrono::duration<double> seconds = end - begin;
    cout << "\nRun Time Results: " << seconds.count() << " seconds\n";

    // Call to funciton to print results of accuracy, specificity, and sensitivity
    results(testTarget, data);

    cout << "\nValari Graham and Jeffrey Li" << endl;

    cout << "CS 4375 Machine Learning" << endl;

    return 0;
}
