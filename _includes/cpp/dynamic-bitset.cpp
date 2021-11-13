#include <bits/stdc++.h>

using namespace std;

class dynamic_bitset {
private:
    typedef uint64_t intType;
    const int NUMBIT = 64, NUMBITMASK = 63, LOGNUMBIT = 6;
    vector<intType> v;
    template<typename T> static bool getbit(T x, int k) { 
        return (x >> k) & T(1); 
    }
    template<typename T> static void turnon(T&x, int k) { 
        x |= T(1)<<k; 
    }
    template<typename T> static void turnoff(T&x, int k) { 
        x &= ~(T(1)<<k); 
    }
    template<typename T> static void setbit(T&x, int k, bool b) {
        x = x & ~(T(1)<<k) | (T(b)<<k);
    }
    template<typename T> static string toBin(T x, int len) {
        string s; if (x == 0) s = "0";
        while (x > 0) { s += (x&1) + '0'; x >>= 1LL; }
        assert(s.size() <= len);
        while (s.size() < len) s += '0';
        reverse(s.begin(),s.end());
        return s;
    }
public:
    int size; // number of bits
    dynamic_bitset() {
        size = 0; 
        v = vector<intType>(); 
    }
    void init(int n) { 
        v.assign((n>>LOGNUMBIT)+bool(n&NUMBITMASK), 0); 
        size = v.size()<<LOGNUMBIT; 
    }
    void assign(int s) {
        size = s<<LOGNUMBIT; 
        v.assign(s, 0);
    }
    void resize(int s) {
        size = s<<LOGNUMBIT;
        v.resize(s, 0);
    }

    bool test(int i) const { 
        return getbit(v[i>>LOGNUMBIT], i&NUMBITMASK); 
    }
    bool any() const { 
        for (intType val : v) 
            if (val) return true; 
        return false; 
    }
    bool none() const { 
        return !any(); 
    }
    bool all() const { 
        for (intType val : v) 
            if (~val) return false; 
        return true; 
    }

    void set() { 
        for (intType &val : v) val = ~0; 
    }
    void set(int i, bool b = true) { 
        setbit(v[i>>LOGNUMBIT], i&NUMBITMASK, b); 
    }
    void on(int i) { 
        turnon(v[i>>LOGNUMBIT], i&NUMBITMASK); 
    }
    void off(int i) { 
        turnoff(v[i>>LOGNUMBIT], i&NUMBITMASK); 
    }
    void reset() { 
        for (intType &val : v) val = 0; 
    }
    void flip() { 
        for (intType &val : v) val = ~val; 
    }
    
    dynamic_bitset operator & (const dynamic_bitset &b) const {
        assert(size == b.size);
        dynamic_bitset res; res.assign(v.size());
        for (int i = 0; i < v.size(); ++i) {
            res.v[i] = v[i] & b.v[i];
        } return res;
    }
    dynamic_bitset operator | (const dynamic_bitset &b) const {
        assert(size == b.size);
        dynamic_bitset res; res.assign(v.size());
        for (int i = 0; i < v.size(); ++i) {
            res.v[i] = v[i] | b.v[i];
        } return res;
    }
    dynamic_bitset operator ^ (const dynamic_bitset &b) const {
        assert(size == b.size);
        dynamic_bitset res; res.assign(v.size());
        for (int i = 0; i < v.size(); ++i) {
            res.v[i] = v[i] ^ b.v[i];
        } return res;
    }
    dynamic_bitset operator ~ () const {
        dynamic_bitset res; res.assign(v.size());
        for (int i = 0; i < v.size(); ++i) {
            res.v[i] = ~v[i];
        } return res;
    }
    dynamic_bitset operator << (int k) const {
        assert(k >= 0);
        dynamic_bitset res; res.assign(v.size());
        if (k < size) {
            int v_pos = (size-1 - k) >> LOGNUMBIT;
            int bit_pos = (size-1 - k) & NUMBITMASK;
            for (int i = (int)v.size()-1; v_pos >= 0; --i, --v_pos) {
                res.v[i] = v[v_pos] << (NUMBITMASK - bit_pos);
                if (v_pos > 0 && bit_pos < NUMBITMASK) {
                    res.v[i] |= v[v_pos-1] >> (bit_pos+1);
                }
            }
        }
        return res;
    }
    dynamic_bitset operator >> (int k) const {
        assert(k >= 0);
        dynamic_bitset res; res.assign(v.size());
        if (k < size) {
            int v_pos = k >> LOGNUMBIT;
            int bit_pos = k & NUMBITMASK;
            for (int i = 0; v_pos < v.size(); ++i, ++v_pos) {
                res.v[i] = v[v_pos] >> bit_pos;
                if (v_pos+1 < v.size() && bit_pos < NUMBITMASK) {
                    res.v[i] |= v[v_pos+1] << (NUMBIT - bit_pos);
                }
            }
        }
        return res;
    }
    bool operator == (const dynamic_bitset &b) const {
        assert(size == b.size);
        for (int i = 0; i < v.size(); ++i) {
            if (v[i] != b.v[i]) return false;
        } return true;
    }
    bool operator != (const dynamic_bitset &b) const {
        return !(*this == b);
    }

    void operator = (const dynamic_bitset &b) { 
        size = b.size; v = b.v;
    }
    void operator &= (const dynamic_bitset &b) { 
        *this = *this & b; 
    }
    void operator |= (const dynamic_bitset &b) { 
        *this = *this | b; 
    }
    void operator ^= (const dynamic_bitset &b) { 
        *this = *this ^ b; 
    }
    void operator <<= (int k) {
        *this = *this << k;
    }
    void operator >>= (int k) {
        *this = *this >> k;
    }

    void get(long long x) {
        assert((sizeof(x)<<3) <= (size));
        reset();
        for (int i = 0; i < (sizeof(x)<<3); ++i) {
            set(i, getbit(x, i));
        }
    }

    string to_string() const {
        string s;
        for (int i = (int)v.size()-1; i >= 0; --i) {
            s += toBin(v[i], NUMBIT);
        } return s;
    }

    friend ostream& operator << (ostream&os, const dynamic_bitset &b) {
        os << b.to_string(); return os;
    }
};

int main() {
    dynamic_bitset a, b;
    a.init(1);
    b.init(1);
    a.get(1394873434987LL);
    b.get(439873987323344LL);
    cout << a << '\n';
    cout << b << '\n';
    cout << (a&b) << '\n';
    cout << (a|b) << '\n';
    cout << (a^b) << '\n';
    cout << (~a) << '\n';
    cout << (a<<10) << '\n';
    cout << (a>>10) << '\n';
    return 0;
}