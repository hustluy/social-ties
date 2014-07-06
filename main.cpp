#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include "libsvm/svm.h"
#include "libsvm/svm.cpp"
#include "naiveBayes.h"
#include "naiveBayes.cpp"

using namespace std;
using namespace boost;

const int low = 10;
const int high = 50;

struct Pred {
    int index;
    int label;
    double prob;
    Pred(int _index, int _label, double _prob) {
        index = _index;
        label = _label;
        prob = _prob;
    };
    bool operator<(const Pred& rhs) const {
        return prob < rhs.prob;
    };
    bool operator>(const Pred& rhs) const {
        return prob > rhs.prob;
    };
    bool operator<=(const Pred& rhs) const {
        return !(*this < rhs);
    };
    bool operator>=(const Pred& rhs) const {
        return !(*this < rhs);
    };
};

void Inst2SVMNode(const vector<double>& single, svm_node*& nodePtr)
{
    int n = single.size() - 1;
    nodePtr = new svm_node[n+1];
    for(int i = 0; i < n; ++i)
    {
        nodePtr[i].index = i;
        nodePtr[i].value = single[i];
    }
    nodePtr[n].index = -1;
}

vector<Pred> TopK(const vector<Pred>& pred, double thresh)
{
    size_t i = 0;
    while(i < pred.size() && pred[i].prob >= thresh)
        ++i;
    return vector<Pred>(pred.begin(), pred.begin() + i);
}

vector<Pred> TopK(const vector<Pred>& pred, int k)
{
    int n = (pred.size() < k)?pred.size():k;
    return vector<Pred>(pred.begin(), pred.begin() + n);
}

void LoadData(const char* file, vector<vector<double> >& inst)
{
    ifstream fin(file);
    if( !fin )
    {
        throw invalid_argument("cann't open file");
    }

    string header;
    getline(fin, header);

    vector<double> minVal, maxVal;
    string line;
    while( getline(fin, line) )
    {
        vector<string> temp;
        split(temp, line, is_any_of(", \t"));
        vector<double> dval;
        for(size_t i = 2; i < temp.size(); ++i)
        {
            istringstream iss( temp.at(i) );
            double val; iss >> val;
            dval.push_back( val );
        }
        inst.push_back( dval );

        if( minVal.empty() )
        {
            minVal.assign(dval.begin(), dval.end());
            maxVal.assign(dval.begin(), dval.end());
        }
        else 
        {
            for(size_t i = 0; i < dval.size(); ++i)
            {
                minVal[i] = (minVal[i] > dval[i])?dval[i]:minVal[i];
                maxVal[i] = (maxVal[i] > dval[i])?maxVal[i]:dval[i];
            }
        }
    }
    fin.close();
    for(size_t i = 0; i < inst.size(); ++i)
    {
        for(size_t j = 0; j < inst[i].size(); ++j)
            inst[i][j] = (inst[i][j] - minVal[j])/(maxVal[j] - minVal[j]);
    }
    /*ofstream fout("data.tst");
    for(size_t i = 0; i < inst.size(); ++i)
    {
        fout << inst[i].back();
        for(size_t j = 0; j < inst[i].size() - 1; ++j)
        {
            fout << " " << j + 1 << ":" << inst[i][j];
        }
        fout << endl;
    }
    fout.close();*/

}

svm_problem InitSVMPro(const vector<vector<double> >& inst)
{
    size_t n = inst.size(), m = inst[0].size() - 1;
    svm_problem svmPro;
    svmPro.l = n;
    svmPro.f = m;
    svmPro.y = new double[n];
    svmPro.x = new svm_node*[n];
    for(size_t i = 0; i < n; ++i)
        svmPro.x[i] = new svm_node[m+1];
    for(size_t i = 0; i < n; ++i)
    {
        svmPro.y[i] = inst[i].back();
        for(size_t j = 0; j < m; ++j)
        {
            svmPro.x[i][j].index = j;
            svmPro.x[i][j].value = inst[i][j];
        }
        svmPro.x[i][m].index = -1;
    }
    return svmPro;
}


svm_parameter InitSVMPara(const svm_problem& svmPro,
        const vector<double>& weight)
{
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
    return svmPara;
}

svm_model UpdateSVM(const vector<vector<double> >& train, 
        const vector<double>& weight)
{
    svm_problem svmPro = InitSVMPro(train);
    svm_parameter svmPara = InitSVMPara(svmPro, weight);
    if( svm_check_parameter(&svmPro, &svmPara) != NULL )
        throw invalid_argument("svm parameter error");
    svm_model svmModel = *svm_train(&svmPro, &svmPara);
    return svmModel;
}

int Flatten(int low, int high, int val)
{
    if( low > high )
        return val<high?val:high;
    if( val < low )
        return low;
    if( val > high )
        return high;
}

vector<Pred> SVMMark(svm_model* svmPtr,
        const vector<vector<double> >& test, const vector<bool>& used)
{
    int numClass = svm_get_nr_class(svmPtr);
    double *probs = new double[numClass];
    vector<Pred> pred;

    for(size_t i = 0; i < test.size(); ++i)
    {
        if( !used[i] )
        {
            svm_node* x;
            Inst2SVMNode(test[i], x);
            int label = svm_predict_probability(svmPtr, x, probs);
            pred.push_back( Pred(i, label, probs[0]>probs[1]?probs[0]:probs[1]) );
        }
    }
    sort(pred.begin(), pred.end(), greater<Pred>());
    vector<Pred> ans = TopK(pred, 0.98);
    int sz = Flatten(low, high, ans.size());
    if( sz != ans.size() )
        ans = TopK(pred, sz);
    cout << "the number of instance svm marked is " << ans.size() << endl;
    return ans;
}

NaiveBayesClassifier UpdateBayes(const vector<vector<double> >& train,
        const vector<double>& weight)
{
    NaiveBayesClassifier nbc(train, weight);
    nbc.estimateParameters();
    return nbc;
}

vector<Pred> BayesMark(NaiveBayesClassifier& nbc,
        const vector<vector<double> >& test, const vector<bool>& used)
{
    vector<Pred> pred;
    for(size_t i = 0; i < test.size(); ++i)
    {
        if( !used[i] )
        {
            pair<int, double> ret = nbc.getProb(test[i]);
            pred.push_back( Pred(i, ret.first, ret.second) );
        }
    }
    sort(pred.begin(), pred.end(), greater<Pred>());
    int n = pred.size();
    vector<Pred> ans = TopK(pred, 0.9999);
    int sz = Flatten(low, high, ans.size());
    if( sz != ans.size() )
        ans = TopK(pred, sz);
    cout << "the number of instance bayes marked is " << ans.size() << endl;
    return ans;
}

void UpdateData(vector<vector<double> >& train, vector<vector<double> >& test,
        vector<bool>& used, vector<Pred>& mark)
{
    for(size_t i = 0; i < mark.size(); ++i)
    {
        int index = mark[i].index;
        vector<double> single = test.at(index);
        single.back() = mark[i].label;

        train.push_back( single );
        used[index] = true;
    }
}

pair<svm_model, NaiveBayesClassifier> CoTrain(const vector<vector<double> >& train, 
        const vector<vector<double> >& test)
{
    vector<vector<double> > nbTrain(train), svmTrain(train);
    vector<double> nbTrainW(nbTrain.size(), 1.0), svmTrainW(svmTrain.size(), 1.0);
    vector<vector<double> > nbTest(test), svmTest(test);
    vector<bool> nbUsed(test.size(), false), svmUsed(test.size(), false);
    int maxIter = 30;
    int iter = 0;
    size_t powi = 1;
    double eps = 0.001;
    while( true )
    {
        svm_model svmModel = UpdateSVM(svmTrain, svmTrainW);
        NaiveBayesClassifier nbc = UpdateBayes(nbTrain, nbTrainW);
        vector<Pred> svmMark = SVMMark(&svmModel, svmTest, nbUsed);
        vector<Pred> nbMark = BayesMark(nbc, nbTest, svmUsed);

        UpdateData(nbTrain, nbTest, nbUsed, svmMark);
        nbTrainW.insert(nbTrainW.end(), svmMark.size(), powi*eps);
        UpdateData(svmTrain, svmTest, svmUsed, nbMark);
        svmTrainW.insert(svmTrainW.end(), nbMark.size(), powi*eps);

        if( ++iter == maxIter )
        {
            return make_pair(svmModel, nbc);
        }

        svm_destroy_param(&(svmModel.param));
        svm_free_model_content(&svmModel);

        if( powi*eps < 1.0 )
            powi *= 2;
    }
}

void TestCoModel(svm_model& svmModel, NaiveBayesClassifier& nbc,
        vector<vector<double> >& test)
{
    static int count = 0;
    ofstream fout("ans", ofstream::app);
    fout << ++count << "th test" << endl;
    size_t sameCnt = 0, diffCnt = 0;
    vector<vector<double> > conf(2, vector<double>(2, 0));
    for(size_t i = 0; i < test.size(); ++i)
    {
        svm_node *x;
        Inst2SVMNode(test[i], x);
        int svmPred = svm_predict(&svmModel, x);
        int nbPred = nbc.classify(test[i]);
        if( svmPred == nbPred )
        {
            sameCnt += 1;
            conf[test[i].back()][nbPred] += 1;
        }
        else diffCnt += 1;
    }
    fout << "the number of sample(svm=bayes): " << sameCnt << endl;
    fout << "the nubmer of sample(svm!=bayes): " << diffCnt << endl;
    fout << "confusion matrix" << endl;
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
            fout << conf[i][j] << " ";
        fout << endl;
    }
    fout.close();
}

void CrossValidation(int folds)
{
    vector<vector<double> > inst;
    vector<double> minVal, maxVal;
    LoadData("data.csv", inst);
    
    // split the data into train and test
    system("rm ans");
    for(int k = 0; k < folds; ++k)
    {
        vector<vector<double> > train;
        vector<vector<double> > test;
        for(int i = 0; i < (int)inst.size(); ++i)
        {
            if( i%folds == k )
                train.push_back(inst.at(i));
            else test.push_back(inst.at(i));
        }
        pair<svm_model, NaiveBayesClassifier> coModel = CoTrain(train, test);
        TestCoModel(coModel.first, coModel.second, test);
    }
}

void Test()
{
    vector<vector<double> > inst;
    LoadData("data.csv", inst);
    //svm_node* ptr;
    //Inst2SVMNode(inst[0], ptr);
    //int i = 0;
    //while( ptr[i].index != -1 )
    //{
        //cout << ptr[i++].value << " ";
    //}
    //cout << endl;
    //copy(inst[0].begin(), inst[0].end(), ostream_iterator<double>(cout, " "));
    //cout << endl;
    svm_problem svmPro = InitSVMPro(inst);
    svm_parameter svmPara = InitSVMPara(svmPro, vector<double>(inst.size(), 1.0));
    svm_model svmModel = *svm_train(&svmPro, &svmPara);
    if( svm_check_probability_model(&svmModel) )
    {
        int n = svm_get_nr_class(&svmModel);
        double *probs = new double[n];
        double ans = svm_predict_probability(&svmModel, svmPro.x[0], probs);
        cout << "ans = " << ans << " true = " << inst[0].back() << endl;
        cout << "probability estimation" << endl;
        copy(probs, probs + 2, ostream_iterator<double>(cout, " "));
        cout << endl;
    }
}

void Work()
{
    CrossValidation(10);
}
int main()
{
    Work();
    //Test();
    return 0;
}

