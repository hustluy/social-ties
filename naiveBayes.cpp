#include "naiveBayes.h"

void NaiveBayesClassifier::estimateParameters()
{
    int n = inst.size(), m = inst[0].size() - 1;
    mean.assign(2, vector<double>(m, 0.0));
    var.assign(2, vector<double>(m, 0.0));
    int num[] = {0, 0};
    double totalWeight[] = {0.0, 0.0};
    for(int i = 0; i < n; ++i)
    {
        int c = inst[i].back();
        num[c]++;
        totalWeight[c] += weight[i];
    }
    for(int j = 0; j < m; ++j)
    {
        for(int i = 0; i < n; ++i)
        {
            int c = inst[i].back();
            mean[c][j] += weight[i]*inst[i][j];
            var[c][j] += weight[i] * inst[i][j] * inst[i][j];
        }
        for(int i = 0; i < 2; ++i)
        {
            mean[i][j] /= totalWeight[i];
            var[i][j] = (var[i][j] - 
                    totalWeight[i]*mean[i][j]*mean[i][j])/totalWeight[i];
        }
    }
    
    //for(int i = 0; i < m; ++i)
    //{
        //cout << mean[0][i] << " " << var[0][i] << " | ";
        //cout << mean[1][i] << " " << var[1][i] << endl;
    //}
}

pair<double, double> NaiveBayesClassifier::calculateProb(const vector<double>& single)
{
    double prob[] = {0.0, 0.0};
    const double PI = 3.14158926;
    for(size_t i = 0; i < single.size() - 1; ++i)
    {
        for(size_t c = 0; c < 2; ++c)
        {
            double first = -(single[i] - mean[c][i])*(single[i] - mean[c][i])/(2*var[c][i]);
            double second = -0.5*log(2*PI*var[c][i]);
            prob[c] += first + second;
        }
    }
    return pair<double, double>(exp(prob[0]), exp(prob[1]));
}

void NaiveBayesClassifier::debug() 
{
    vector<vector<double> > cm(2, vector<double>(2, 0.0));
    for(size_t i = 0; i < inst.size(); ++i)
    {
        int r = inst[i].back();
        int c = classify(inst[i]);
        cm[r][c] += 1.0;
        //if( c == 0 )
        //{
            //pair<int, double> ans = getProb(inst[i]);
            //cout << ans.first << " " << ans.second << endl;
        //}
    }

    cout << "confusion matrix" << endl;
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
            cout << cm[i][j]/*/(cm[i][0] + cm[i][1])*/ << " ";
        cout << endl;
    }
}

//void LoadData(const char* file)
//{
    //ifstream fin(file);
    //if( !fin )
    //{
        //throw invalid_argument("cann't open file");
    //}

    //string header;
    //getline(fin, header);

    //vector<vector<double> > inst;
    //vector<double> minVal, maxVal;
    //string line;
    //while( getline(fin, line) )
    //{
        //vector<string> temp;
        //split(temp, line, is_any_of(", \t"));
        //vector<double> dval;
        //for(size_t i = 2; i < temp.size(); ++i)
        //{
            //istringstream iss( temp.at(i) );
            //double val; iss >> val;
            //dval.push_back( val );
        //}
        //inst.push_back( dval );

        //if( minVal.empty() )
        //{
            //minVal.assign(dval.begin(), dval.end());
            //maxVal.assign(dval.begin(), dval.end());
        //}
        //else 
        //{
            //for(size_t i = 0; i < dval.size(); ++i)
            //{
                //minVal[i] = (minVal[i] > dval[i])?dval[i]:minVal[i];
                //maxVal[i] = (maxVal[i] > dval[i])?maxVal[i]:dval[i];
            //}
        //}
    //}
    //fin.close();
    //for(size_t i = 0; i < inst.size(); ++i)
    //{
        //for(size_t j = 0; j < inst[i].size(); ++j)
        //{
            //inst[i][j] = (inst[i][j] - minVal[j])/(maxVal[j] - minVal[j]);
        //}
    //}
    //NaiveBayesClassifier nbc(inst, vector<double>(inst.size(), 0.01));
    //nbc.estimateParameters();
    //nbc.debug();
//}


//int main()
//{
    //LoadData("data.csv");
    //return 0;
//}
