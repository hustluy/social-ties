#include <iostream>
#include <vector>
#include <functional>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <memory>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include "libsvm/svm.h"
#include "libsvm/svm.cpp"

using namespace std;

void Handle(ifstream& fin, int i)
{
    int u, v;
    int trainCnt = 0, testCnt = 0;
    int sameCnt = 0, diffCnt = 0;
    while( fin >> u >> v )
    {
        int flag; fin >> flag;
        if( flag )
        {
            int temp; fin >> temp;
            ++trainCnt;
        }
        else 
        {
            ++testCnt;
            double svmP0, svmP1;
            fin >> svmP0 >> svmP1;
            int svmPred = svmP0>svmP1?0:1;
            double nbP0, nbP1;
            fin >> nbP0 >> nbP1;
            int nbPred = nbP0>nbP1?0:1;
            if( svmPred == nbPred )
                ++sameCnt;
            else ++diffCnt;
        }
    }
    cout << i << "th test" << endl;
    cout << "train : " << trainCnt << " test : " << testCnt << endl;
    cout << "svm=bayes : " << sameCnt << endl;
    cout << "svm!=bayes : " << diffCnt << endl;
}

void Work()
{
    char buf[80];
    for(int i = 1; i <= 10; ++i)
    {
        sprintf(buf, "result/out%d", i);
        ifstream fin(buf);
        Handle(fin, i);
        fin.close();
    }
}

pair<int, int> ToIntPair(const string& left, const string& right)
{
    pair<int, int> ans;
    istringstream iss(left + " " + right);
    iss >> ans.first >> ans.second;
    return ans;
}

int main()
{
    Work();
    return 0;
}
