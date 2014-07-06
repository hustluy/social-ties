#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include "libsvm/svm.h"
#include "libsvm/svm.cpp"

using namespace std;

struct XX {
    int x;
    XX(int _x):x(_x){
    };
    bool operator>(const XX& rhs) const {
        return x > rhs.x;
    }
};

void Work()
{
    svm_problem svmPro;
    svmPro.f = 10;
    svmPro.l = 10;
    vector<double> weight(10, 1.0);
    svm_parameter svmPara;
    svmPara.svm_type = C_SVC;
    svmPara.kernel_type = RBF;
    svmPara.degree = 3;
    svmPara.gamma = 1.0/static_cast<double>(svmPro.f);
    svmPara.coef0 = 0;
    svmPara.nu = 0.5;
	svmPara.cache_size = 100;
	svmPara.C = 1;
    svmPara.myC = new double[svmPro.l];
    copy(weight.begin(), weight.end(), svmPara.myC);
	svmPara.eps = 1e-3;
	svmPara.p = 0.1;
	svmPara.shrinking = 1;
	svmPara.probability = 1;
	svmPara.nr_weight = 0;
	svmPara.weight_label = NULL;
	svmPara.weight = NULL;
    const char *ptr = svm_check_parameter(&svmPro, &svmPara);
    if( ptr != NULL )
        throw invalid_argument(ptr);
}

int main()
{
    Work();
    //system("ls");
    return 0;
}
