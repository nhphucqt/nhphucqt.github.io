// suffix array O(N*logN)

#include <bits/stdc++.h>

using namespace std;

struct suffixArray {
    string s;
    int n, lim;
    vector<int> SA, Rank, LCP, tmp, cnt;
    suffixArray() {}
    suffixArray(const string &_s, int m = 256): s(_s+'#'), n(s.size()), lim(m), 
            SA(n), Rank(n), LCP(n), tmp(n), cnt(max(n,lim)) {
        buildSA();
        buildLCP();
    }
    void Sort() {
        for (int i = 0; i < lim; ++i) cnt[i] = 0;
        for (int i = 0; i < n; ++i) cnt[Rank[i]]++;
        for (int i = 1; i < lim; ++i) cnt[i] += cnt[i-1];
        for (int i = n-1; i >= 0; --i) SA[--cnt[Rank[tmp[i]]]] = tmp[i];
    }
    bool equal(int i, int j, int k) {
        return tmp[i] == tmp[j] && tmp[(i+k)%n] == tmp[(j+k)%n];
    }
    void buildSA() {
        for (int i = 0; i < n; ++i) Rank[tmp[i]=i] = s[i];
        Sort();
        for (int k = 1, i, num; k < n; k<<=1, lim = num) {
            for (tmp.swap(SA), i = 0; i < n; ++i) tmp[i] = (tmp[i]-k+n)%n;
            Sort();
            for (tmp.swap(Rank), Rank[SA[0]] = 0, i = 1, num = 1; i < n; ++i) {
                Rank[SA[i]] = !equal(SA[i], SA[i-1], k) ? num++ : num-1;
            }
        }
    }
    void buildLCP() {
        for (int i = 0, k = 0; i < n-1; ++i) {
            int j = SA[Rank[i]-1];
            while (s[i+k] == s[j+k]) k++;
            LCP[Rank[i]] = k;
            if (--k < 0) k = 0;
        }
    }
    void test() {
        for (int i = 0; i < n; ++i) {
            cerr << LCP[i] << ' ' << s.substr(SA[i]) << '\n';
        }
    }
};

int main() {
    string s;
    cin >> s;
    suffixArray sa(s);
    sa.test();
    return 0;
}
