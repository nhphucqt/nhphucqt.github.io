#include <bits/stdc++.h>

using namespace std;

const int numTest = 1000;
const bool useChecker = false;
const string name = "";

mt19937_64 ra(time(nullptr));
long long rand(long long l, long long r) {
    assert(l <= r);
    return l + ra() % (r-l+1);
}

void genTest() {
    ifstream inp(name+".inp");
    //
    // Make test here
    //
    inp.close();
}

bool checker() {
    ifstream inp(name+".inp");
    ifstream out(name+".out");
    ifstream ans(name+".ans");
    bool checkValue = false;
    // Correct output -> checkValue = true
    // Incorrect output -> checkValue = false
    //
    // Make checker here
    //
    inp.close();
    out.close();
    ans.close();
    return checkValue;
}

void strTest() {
    for (int it = 0; it < numTest; ++it) {
        cerr << "Test " << it << ":\n";
        system(("./"+name).c_str());
        system(("./"+name+"_").c_str());
        if (useChecker) {
            if (checker()) {
                cerr << "OK\n";
            }
            else {
                cerr << "Diff\n";
                return;
            }
        }
        else {
            if (system(("diff "+name+".out "+name+".ans").c_str())) {
                cerr << "Diff!!\n";
                return;
            }
            else {
                cerr << "OK\n";
            }
        }
    }
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    genTest();
    strTest();
    return 0;
}