#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>

using namespace std;

struct XX {
    int x;
    XX(int _x):x(_x){
    };
    bool operator>(const XX& rhs) const {
        return x > rhs.x;
    }
};

int main()
{
    auto_ptr<int> ptr(new int(2));
    cout << *ptr << endl;
    int *iptr = ptr;
    return 0;
}
