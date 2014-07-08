#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <cstddef>
#include <cmath>
#include <unistd.h>

using namespace std;
using namespace boost;

class NaiveBayesClassifier {
    public:
        NaiveBayesClassifier(const vector<vector<double> >& _inst, 
                const vector<double>& _w):inst(_inst), weight(_w){
        };
        void estimateParameters();
        pair<double, double> calculateProb(const vector<double>& ); 

        int classify(const vector<double>& single) {
            pair<double, double> prob = calculateProb(single);
            if( prob.first >= prob.second )
                return 0;
            else return 1;
        };

        pair<int, double> getProb(const vector<double>& single) {
            pair<double, double> prob = calculateProb(single);
            double sum = prob.first + prob.second;
            if( fabs(sum) < 1e-10 )
            {
                cout << "WARNING:分母为零" << endl;
                sleep(10);
            }
            if( prob.first >= prob.second )
                return pair<int, double>(0, prob.first/sum);
            else return pair<int, double>(1, prob.second/sum);
        }

        void debug();
    private:
        vector<vector<double> > inst;
        vector<double> weight;
        vector<vector<double> > mean;
        vector<vector<double> > var;
};

#endif
