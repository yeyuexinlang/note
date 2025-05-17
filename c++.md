# **STL**

## **list(双向链表)**

```c++
//定义一个list 
    list<int>node; 
//为链表赋值，例如定义一个包含n个结点的链表
    for(int i = 1; i <= n; i ++){
        node.push_back(i);
    } 
//遍历链表，用it遍历链表，例如从头到尾遍历
    list<int>::iterator it= node.begin();
    while (node.size() > 1){    //list的大小由STL管理
        it++; 
        if(it == node.end()){   //循环链表，end()是1ist末端下一位置
            it = node.begin();
            break;
        }
    }
//删除一个结点
    list<int>::iterator next = ++it;
    if(next == node.end()) next = node.begin();  //循环链表
    node.erase(--it);   //删除这个结点，并将it指向前一个节点，避免再次访问it时发生错误
    it = next;   //更新it 
```
[例题（lanqiaoOJ题号1518）：](https://www.lanqiao.cn/problems/1518/learning/?page=1&first_category_id=1&problem_id=1518)
```c++
# include <bits/stdc++.h>
using namespace std;

vector<list<int>::iterator> local(1e5 + 5);
list<int> arr;
int x, y, z;
void solve(void){
    cin >> x >> y >> z;
    list<int>::iterator temp = local[y];
    if(z == 0){
        arr.insert(temp, x);
        local[x] = --temp;
    }
    if(z == 1){
        arr.insert(++temp, x);
        local[x] = --temp;
    }
}

signed main(void){
    int t; cin >> t;
    cin >> x;
    arr.emplace_back(x);
    local[x] = arr.begin();
    t --;
    while(t --){
        solve();
    }
    for(auto it : arr){
        cout << it << " ";
    }

    return 0;
}
```

---

# **字符串哈希**

## 性质

- 输入参数的可能性无线，输出的值范围相对有限
- 输入相同样本得到相同的值（没有随机机制）
- 输入不同的样本也可能得到相同的值（哈希碰撞，可能行较小）
- 输入大量不同的样本，得到大量输出值，几乎均匀的分布在整个输出域上

## 细节

1. 理解unsigned long long类型自然溢出，计算加减乘除时，自然溢出后的状态等同于对2^64次方取模的值。
2. 字符串化成base进制的数字并让其自然溢出。
3. base可以选择一些指数比如：433、299、599、1000000007；（经典值：31、131、1313、13131、131313等）
4. 转化时让每一位的值***从1开始***，不从0开始；
5. 利用数字的比较去替代字符串比较，可以大大减少复杂度  
>注：出现哈希碰撞就换base。

---

# ST算法

> ST算法通常用于求解 RMQ (Range-Minimum/Maximum Query, **区间最值问题**)  
> ST(Sparse-Table)算法：是一种用于解决 RMQ 问题的高效算法，它基于动态规划的思想，通过预处理得到一个二维数组，从而在O(1)的时间复杂度内回答  RMQ 查询。

## 模板(RMQ-max)

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5+10;
int n, m;
int a[N], dp[N][40];
void st_init(){
    for(int i = 1; i <= n; i ++) dp[i][0] = a[i]; // 初始胡区间长度为1时的值
    int p = log2(n); // 或者 p = (int)(log(double(n)) / log(2.0));
    for(int k = 1; k <= p; k ++){ // 用dp[]递推计算区间
        for(int s = 1; s + (1 << k) <= n + 1; s ++){
            dp[s][k] = max(dp[s][k - 1], dp[s + (1 << (k - 1))][k - 1]); // 最大值
        }
    }
}
int st_query(int L, int R){
    int k = log2(R - L + 1); // 或者 k = (int)(log(double(R - L + 1)) / log(2.0));
    return max(dp[L][k], dp[R - (1 << k) + 1][k]); // 最大值
}
signed main(){
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i ++) scanf("%d", &a[i]);
    st_init();
    for(int i = 1; i <= m; i ++){
        int L, R; scanf("%d%d", &L, &R);
        printf("%d\n", st_query(L, R));
    }
    return 0;
}
```

---

# 动态规划

## 区间dp

> ***石子合并问题***  
**题目描述**
设有 N(N≤300) 堆石子排成一排，其编号为 1,2,3,⋯,N。每堆石子有一定的质量mi(mi≤1000)。现在要将这 N 堆石子合并成为一堆。每次只能合并相邻的两堆，合并的代价为这两堆石子的质量之和，合并后与这两堆石子相邻的石子将和新堆相邻。合并时由于选择的顺序不同，合并的总代价也不相同。试找出一种合理的方法，使总的代价最小，并输出最小代价。  
**输入格式**
第一行，一个整数 N。
第二行，N 个整数 mi。
**输出格式**
输出仅一个整数，也就是最小代价。
**样例**
*input*
4
2 5 3 1  
*output*
22

```c++
void solve(void){
    int n; cin >> n;
    vector<int> arr(n + 1, 0);
    vector<int> pre(n + 1, 0);
    for(int i = 1; i <= n; i++){
        cin >> arr[i];
        pre[i] = pre[i - 1] + arr[i];
    } 
    vector<vector<int> > dp(n + 1, vector<int>(n + 1, INT_MAX));
    for(int i = 1; i <= n; i ++){
        dp[i][i] = 0;
    }
    for(int len = 2; len <= n; len ++){ // 枚举长度
        for(int i = 1; i + len - 1 <= n; i ++){ // 起始位置
            int ends = i + len - 1; // 结束位置
            for(int j = i; j < ends; j ++){ // 枚举每段划分可能
                dp[i][ends] = min(dp[i][ends], dp[i][j] + dp[j + 1][ends] + pre[ends] - pre[i - 1]);
            }
        }
    }
    cout << dp[1][n];
}
```

---

## 01背包

```c++
# include <bits/stdc++.h>
using namespace std;

vector<int> v; // 物品体积
vector<int> w; // 物品价值
void solve(void){
    int n, V; // 物品个数， 背包体积
    cin >> n >> V;
    // init
    v.resize(n + 1, 0); 
    w.resize(n + 1, 0);
    vector<vector<int> > dp(n + 1, vector<int>(V  + 1, 0));    
    for(int i = 1; i < i; i ++){
        cin >> v[i] >> w[i];
    }
    for(int i = 1; i <= n; i ++){ // 枚举物品
        for(int j = 1; j <= V; j ++){ // 枚举背包的体积
            dp[i][j] = dp[i - 1][j]; // 不取该物品
            if(j >= v[i]) dp[i][j] = max(dp[i][j], dp[i - 1][j - v[i]] + w[i]); // 如果剩下的体积足够，取最大值
        }
    }
    cout << dp[n][V] << endl;
}

signed main(void){
    ios::sync_with_stdio(0);
    int t = 1;
    //cin >> t;
    while(t --){
        solve();
    }
}
```

---

## 数位dp

```c++
int dp1[100][1000];
int dp2[100][1000];
int n1,n2,Max,Min;
string s1,s2;
int dfs1(int pos,int sum,int islimit,int leadzero){
    if(pos>=n1){
        return (!leadzero && sum>=Min && sum <=Max) ? 1 : 0;
        //当长度为最后一个的时候,前导不为0并且合法就+1；
    }
    //当前导不为0并且不受限时返回记忆化的值；
    if(!islimit && !leadzero && dp1[pos][sum]!=-1){
        return dp1[pos][sum];
    }
    int up = islimit ? s1[pos]-'0' : 9; //当数字受限时，上线为当前数位;
    int res=0;
    for(int i=0;i<=up;i++){
        //如果之前受限，现在到顶了就代表之后数位也将受限;
        //例如12345,执行11xxx的时候前面后面可以随便填充,但是12xxx的时候就要判断第三位受限了;
        //当前导为0并且当前数位为0,前导继续受限;
        res+= dfs1(pos+1,sum+i,islimit && i==up,leadzero && i==0);
        res%=mod;
    }
    //进行记忆化搜索；
    if(!islimit && !leadzero){
        dp1[pos][sum] = res;
    }
    return res;
}
```

---

# BFS  

> [lanqiaoOJ-迷宫](https://www.lanqiao.cn/problems/602/learning/)  

***简单方法***  
记录起点到终点的完整路径，适合小图。
```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

const int n = 30, m = 50;
vector<vector<char> > mat;
vector<vector<bool> > vis;
vector<vector<int> > dirc = {{1, 0}, {0, -1}, {0, 1}, {-1, 0}};
vector<string> mp = {"D", "L", "R", "U"};
struct node{
    int x, y;
    string path;
    node() : x(-1), y(-1){}
    node(int _x, int _y, string _path = "") : x(_x), y(_y), path(_path){}
};

void bfs(){
    queue<node> que;
    que.emplace(node(0, 0));
    vis[0][0] = true;
    node cur, nxt;
    int nxt_x, nxt_y;
    while(!que.empty()){
        cur = que.front(); que.pop();
        if(cur.x == 29 && cur.y == 49){
            cout << cur.path;
            return;
        }
        for(int i = 0; i < 4; i ++){
            nxt_x = cur.x + dirc[i][0], nxt_y = cur.y + dirc[i][1];
            if(nxt_x < 0 || nxt_y < 0 || nxt_x >= 30 || nxt_y >= 50 || mat[nxt_x][nxt_y] == '1' || vis[nxt_x][nxt_y]) continue;
            nxt = node(nxt_x, nxt_y, (cur.path + mp[i]));
            vis[nxt.x][nxt.y] = true;
            que.emplace(nxt);
        }
    }
}

void solve(void){
    mat.resize(n, vector<char> (m));
    vis.resize(n, vector<bool> (m, false));
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < m; j ++){
            cin >> mat[i][j];
        }
    }
    bfs();
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

***标准方法***

将记录所有路径改成仅记录前驱节点

```c++

```


## 双向广搜

>[跳蚱蜢](https://www.lanqiao.cn/problems/642/learning/?page=1&first_category_id=1&problem_id=642)


***单项广搜***
```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

struct node{
    string s;
    int cnt;
    node() : s(""), cnt(0){}
    node(string _s, int _cnt) : s(_s), cnt(_cnt){}
};

string s = "012345678";
unordered_map<string, bool> mp;
queue<node> que;
void solve(void){
    que.emplace(node(s, 0));
    mp[s] = true;
    node cur, nxt;
    while(!que.empty()){
        cur = que.front(); que.pop();
        if(cur.s == "087654321"){
            cout << cur.cnt;
            return;
        }
        int inx;
        for(inx = 0; inx < 9; inx ++){
            if(cur.s[inx] == '0') break;
        }
        for(int j = inx - 2; j <= inx + 2; j ++){
            int k = (j + 9) % 9;
            if(k == inx) continue;
            node nxt = cur;
            swap(nxt.s[inx], nxt.s[k]);
            if(!mp[nxt.s]){
                mp[nxt.s] = true;
            }else{
                continue;
            }
            nxt.cnt ++;
            que.emplace(nxt);
        }
    }
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}

```

***双向广搜***

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

bool isok = false;
string s1 = "012345678", s2 = "087654321";
unordered_map<string, int> mp1, mp2;
queue<string> que1, que2;
void extend(queue<string>& que, unordered_map<string, int>& mp, unordered_map<string, int>& mp2){
    string cur, nxt;
    cur = que.front(); que.pop();
    int inx;
    for(inx = 0; inx < 9; inx ++){
        if(cur[inx] == '0') break;
    }
    for(int j = inx - 2; j <= inx + 2; j ++){
        int k = (j + 9) % 9;
        if(k == inx) continue;
        nxt = cur;
        swap(nxt[inx], nxt[k]);
        if(!mp[nxt]){
            mp[nxt] = mp[cur] + 1;
        }else{
            continue;
        }
        if(mp2[nxt]){
            cout << mp[nxt] + mp2[nxt];
            isok = true;
            return;
        }
        que.emplace(nxt);
    }
}

void dfs(){
    que1.emplace(s1); mp1[s1] = 0;
    que2.emplace(s2); mp2[s2] = 0;
    while(!que1.empty() && !que2.empty()){
        if(que1.size() <= que2.size()){
            extend(que1, mp1, mp2);
        }else{
            extend(que2, mp2, mp1);
        }
        if(isok){
            return;
        }
    }

}

void solve(){
    dfs();
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

---

# 最短路径问题

## Dijkstra

### 标准

> [蓝桥王国](https://www.lanqiao.cn/problems/1122/learning/?page=1&first_category_id=1&problem_id=1122)  

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'
const ll INF = 0x3f3f3f3f3f3f3f3fLL;
const int N = 3e5 + 2;
struct edge{
    int from, to; // 起点、终点; 起点 from 没有用到，e[i] 的 i 就是 from
    ll w;   // 权值
    edge(int _from, int _to, ll _w): from(_from), to(_to), w(_w){}
};
struct node{
    int id; // 节点
    ll dis; // 节点到起点的距离
    node(int _id, ll _dis): id(_id), dis(_dis){}
    bool operator<(const node& a)const{
        return dis > a.dis;
    }
};
int n, m, s;
vector<int> pre(N, -1); // 记录前驱节点，用于生成路径
vector<vector<edge> > mat(N, vector<edge>{}); // 存图
vector<ll> dis(N, 0); // 记录所有节点到起点的距离；
void print_path(int s, int t){ // 输出从 s 到 t 的最短路径
    if(s == t){ // 输出起点
        cout << s << " ";
        return;
    }  
    print_path(s, pre[t]); // 输出前一个点
    cout << t << " "; // 后输出当前点。最后输出终点；
}
void dijkstra(){
    vector<bool> done(N, false); // true表示节点i的最短路径已经找到
    for(int i = 1; i <= n; i ++){ // init
        dis[i] = INF;
        done[i] = false;
    }
    dis[s] = 0; // 起点到自己的距离为0
    priority_queue<node> pq; // 存储节点信息
    pq.emplace(node(s, dis[s])); // 起点入队
    while(!pq.empty()){ 
        node cur = pq.top(); pq.pop(); // pop 出与起点 s 距离最小的节点 cur
        if(done[cur.id]) continue; // 丢弃已经找到最短路径的节点
        done[cur.id] = true;
        for(int i = 0; i < mat[cur.id].size(); i ++){ // 检查节点 cur 的所有邻居节点
            edge nxt = mat[cur.id][i]; // cur.id 的第 i 个邻居节点是 nxt.to
            if(done[nxt.to]) continue; // 丢弃已经找到最短路径的邻居节点
            if(dis[nxt.to] > nxt.w + cur.dis){
                dis[nxt.to] = nxt.w + cur.dis;
                pq.emplace(node(nxt.to, dis[nxt.to]));  // 扩展新的邻居节点，放到优先队列中
                pre[nxt.to] = cur.id;   // 如果有需要，就记录路径
            }
        }
    }
    //print_path(s, n); cout << endl; // 输出路径; 起点为 1，终点 n;
}

void solve(void){
    cin >> n >> m;
    s = 1; // 起点
    for(int i = 1; i <= n; i ++) mat[i].clear();
    while(m --){
        int a, b, w; cin >> a >> b >> w;
        mat[a].emplace_back(edge(a, b, w));
        // mat[b].emplace_back(edge(b, a, w)); // 双向
    }
    dijkstra();
    for(int i = 1; i <= n; i ++){
        if(dis[i] >= INF){
            cout << "-1 ";
        }else{
            cout << dis[i] << " ";
        }
    }
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

### 输出所有路径（加强版）

> **《城市间紧急救援》**
> 作为一个城市的应急救援队伍的负责人，你有一张特殊的全国地图。在地图上显示有多个分散的城市和一些连接城市的快速道路。每个城市的救援队数量和每一条连接两个城市的快速道路长度都标在地图上。当其他城市有紧急求助电话给你的时候，你的任务是带领你的救援队尽快赶往事发地，同时，一路上召集尽可能多的救援队。
**输入格式:**
输入第一行给出 4 个正整数 n、m、s、d，其中 n（2≤n≤500）是城市的个数，顺便假设城市的编号为 0 ~ (n−1)；m 是快速道路的条数；s 是出发地的城市编号；d是目的地的城市编号。
第二行给出 n 个正整数，其中第 i 个数是第 i 个城市的救援队的数目，数字间以空格分隔。随后的 m 行中，每行给出一条快速道路的信息，分别是：城市 1、城市 2、快速道路的长度，中间用空格分开，数字均为整数且不超过 500。输入保证救援可行且最优解唯一。
**输出格式:**
第一行输出最短路径的条数和能够召集的最多的救援队数量。第二行输出从 s 到 d 的路径中经过的城市编号。数字间以空格分隔，输出结尾不能有多余空格。
**输入样例:**
4 5 0 3
20 30 40 10
0 1 1
1 3 2
0 3 3
0 2 2
2 3 2
**输出样例:**
2 60
0 1 3

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n';

struct node{
    int inx, dis;
    node():inx(-1), dis(INT_MAX){}
    node(int _inx, int _dis = INT_MAX):inx(_inx), dis(_dis){}
    bool operator<(const node nxt)const{
        return this->dis > nxt.dis;
    }
};
struct edge{
    int to, val;
    edge():to(-1), val(INT_MAX){}
    edge(int _to, int _val):to(_to), val(_val){}
};

int n, m, start, target;
priority_queue<node,vector<node> > pq;
vector<vector<edge> > mat;
vector<int> val; // 记录每个节点的救援队数量
vector<vector<int> > pre; // 记录前驱节点
vector<vector<int> > path; // 记录所有路径
vector<bool> done; // 记录是否找到最短距离
vector<int> dis; // 记录节点到起点的距离

void init(void){
    cin >> n >> m >> start >> target;
    mat.assign(n, vector<edge>{});
    val.assign(n, 0);
    pre.assign(n, vector<int>{});
    done.assign(n, false);
    dis.assign(n, INT_MAX); dis[start] = 0;
    
    for(auto& it : val) cin >> it;
    for(int i = 0; i < n; i ++){
        mat[i].reserve(500);
    }
    for(int i = 0; i < m; i ++){
        int a, b, v;
        cin >> a >> b >> v;
        mat[a].emplace_back(edge(b, v));
        mat[b].emplace_back(edge(a, v));
    }
    pq.emplace(node(start, 0));
}

void get_path(int cur, vector<int> temp_path){
    temp_path.emplace_back(cur);
    if(pre[cur].empty()){
        path.emplace_back(temp_path);
        return;
    }
    for(auto& it : pre[cur]){
        get_path(it, temp_path);
    }
}

struct cmp{
    bool operator()(const vector<int>& a, const vector<int>& b)const{
        int sum_a = 0, sum_b = 0;
        for(auto it : a){
            sum_a += val[it];
        }
        for(auto it : b){
            sum_b += val[it];
        }
        return sum_a > sum_b;
    }
};

void solve(void){
    init();
    while(!pq.empty()){
        node cur = pq.top(); pq.pop();
        if(done[cur.inx] && cur.inx == target) break;
        if(done[cur.inx]) continue;
        done[cur.inx] = true;
        for(auto& nxt : mat[cur.inx]){
            if(done[nxt.to]) continue;
            if(dis[nxt.to] > cur.dis + nxt.val){
                dis[nxt.to] = cur.dis + nxt.val;
                pre[nxt.to] = {cur.inx};
                pq.emplace(node(nxt.to, dis[nxt.to]));
            }else if(dis[nxt.to] == cur.dis + nxt.val){
                pre[nxt.to].emplace_back(cur.inx);
            }
        }
    }
    get_path(target, vector<int>{});
    sort(path.begin(), path.end(), cmp()); 
    int sums = 0;
    for(auto it : path.front()){
        sums += val[it];
    }
    cout << path.size() << " " << sums << endl;
    for(int i = path.front().size() - 1; i >= 0; i --){
        if(i == 0){
            cout << path.front()[i];
            continue;
        }
        cout << path.front()[i] << " ";
    }

    
}

signed main(void){
    ios::sync_with_stdio(false);
    solve();
    return 0;
}
```


---

# 并查集

## template

```c++
int n, m; 
vector<int> arr;

void init(){
    for(int i = 0; i <= n; i ++) arr[i] = i;
}
int getFa(int a){ 
    if(a == arr[a]) return a; // 如果该节点指向自己
    //return getFa(arr[a]);
    return a = getFa(arr[a]);// 建森林
}
void Union(int x, int y){
    int X = getFa(x), Y = getFa(y); // 获取父节点
    if(X == Y) return; // 两节点在同一集合中
    arr[X] = Y; // 将两节点进行连接
}

void solve(){
    cin >> n >> m;
    arr.resize(n + 1);
    init();
    int a, b;
    for(int _ = 0; _ < m; _ ++){
        cin >> a >> b;
        Union(a, b);
    }
    //······

}
```
---

# 区间查询

## 线段树

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define ll long long
#define LfTree(x) (x << 1)
#define RtTree(x) (x << 1 | 1)

struct node{
    ll val, lf, rt;
    node():val(-1), lf(-1), rt(-1){};
    node(ll _val, ll _lf, ll _rt):val(_val), lf(_lf), rt(_rt){};
};

ll N;
vector<node> tree; // 从1开始
vector<ll> tag; // 懒惰标记,lazy_tag
vector<ll> arr; // 存入数中的值

void build(ll inx, ll lf, ll rt){ // 建线段树
    if(lf == rt){
        tree[inx] = node(arr[lf], lf, rt); // 存储叶子节点
        return;
    }
    ll mid = lf + (rt - lf) / 2;
    build(LfTree(inx), lf, mid); // 左子树
    build(RtTree(inx), mid + 1, rt); // 右子树
    tree[inx] = node(tree[LfTree(inx)].val + tree[RtTree(inx)].val, lf, rt); // 根据左右子树确定该节点
}

void add_tag(ll inx, ll val){   // 添加懒惰标记
    tag[inx] += val;
    tree[inx].val += (tree[inx].rt - tree[inx].lf + 1) * val;
} 

void emplace_tag(ll inx){ // 传递懒惰标记
    if(!tag[inx]) return; // 没有懒惰标记
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    add_tag(LfTree(inx), tag[inx]); // 传递给左子树
    add_tag(RtTree(inx), tag[inx]); // 传递给右子树
    tag[inx] = 0; // 清楚自己的标记
}

void update(ll inx, ll lf, ll rt, ll val){ // 修改数据
    if(lf <= tree[inx].lf && tree[inx].rt <= rt){ // 该节点被所求区间全覆盖，修改并添加懒惰标记
        add_tag(inx, val);
        return;
    }
    emplace_tag(inx); // 懒惰标记传递，确保左右子树为最新状态
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    if(lf <= mid){ // 更新左节点
        update(LfTree(inx), lf, rt, val);
    } 
    if(mid < rt){ // 更新右节点
        update(RtTree(inx), lf, rt, val);
    }
    tree[inx].val = tree[LfTree(inx)].val + tree[RtTree(inx)].val; //  更新自己
}

ll query(ll inx, ll lf, ll rt){ // 查询
    emplace_tag(inx); // 传递懒惰标记
    if(lf <= tree[inx].lf && tree[inx].rt <= rt){ // 该节点被所求区间全覆盖
        return tree[inx].val;
    }
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    ll ans = 0;
    if(lf <= mid){ // 左子树与所求区间有交集
        ans += query(LfTree(inx), lf, rt);
    }
    if(mid < rt){ // 右子树与所求区间有交集
        ans += query(RtTree(inx), lf, rt);
    }
    return ans;
}

void printTree(){ // 输出线段树每个节点的值
    cout << "修改后线段树每个节点的值：" << endl;
    for (ll i = 1; i < tree.size(); ++i) {
        if(tree[i].lf == -1) break;
        cout << "Node " << i << ": [" << tree[i].lf << ", " << tree[i].rt << "] = " << tree[i].val << endl;
    }
}

void init(){
    tree.reserve(N * 4);
    arr.resize(N + 1);
    tag.resize(N * 4, 0);
    for(ll i = 1; i <= N; i ++){
        cin >> arr[i];
    }
    build(1, 1, N);
}
void solve(void){
    cin >> N;
    init();
}

signed main(void){
    ios::sync_with_stdio(false);
    solve();
    return 0;
}
```

---

## 归并树

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define int long long
#define ll long long
#define LfTree(x) (x << 1)
#define RtTree(x) (x << 1 | 1)

struct node{
    vector<int> val;
    int lf, rt;
    node():lf(-1), rt(-1){};
    node(vector<int> _val, int _lf, int _rt):val(_val), lf(_lf), rt(_rt){};
};

int N;
vector<node> tree; // 从1开始
vector<int> arr; // 存入数中的值

void build(int inx, int lf, int rt){ // 建归并树
    if(lf == rt){
        tree[inx] = node({arr[lf]}, lf, rt); // 存储叶子节点
        return;
    }
    int mid = lf + (rt - lf) / 2;
    build(LfTree(inx), lf, mid); // 左子树
    build(RtTree(inx), mid + 1, rt); // 右子树
    // 将左右子树的有序数组拼接为新的有序数组
    merge(
        tree[LfTree(inx)].val.begin(), tree[LfTree(inx)].val.end(), 
        tree[RtTree(inx)].val.begin(), tree[RtTree(inx)].val.end(),
        back_inserter(tree[inx])
    );
}

int query_less(int inx, int lf, int rt, int x){ // 获取区间[lf,rt]中小于 x 的元素数量
    if(tree[inx].lf > rt || tree[inx].rt < lf) return 0;
    if(tree[inx].lf >= lf && tree[inx].rt <= rt){ // 该节点完全被所求区间包含
        auto it = lower_bound(tree[inx].val.begin(), tree[inx].val.end(), x);
        return it - tree[inx].val.begin();  // 返回该节点所表示区间中满足条件的元素数量
    }
    int mid = tree[inx].lf + ((tree[inx].rt - tree[inx].lf) >> 1);
    return query_less(LfTree(inx), lf, rt, x) + query_less(RtTree(inx), lf, rt, x); // 将左右子树的结果相加
}

int query_greater(int inx, int lf, int rt, int x){  // 获取区间[lf,rt]中大于 x 的元素数量
    if(tree[inx].lf > rt || tree[inx].rt < lf) return 0;
    if(tree[inx].lf >= lf && tree[inx].rt <= rt){   // 该节点完全被所求区间包含
        auto it = upper_bound(tree[inx].val.begin(), tree[inx].val.end(), x);
        return tree[inx].val.end() - it;    // 返回该节点所表示区间中满足条件的元素数量
    }
    int mid = tree[inx].lf + ((tree[inx].rt - tree[inx].lf) >> 1);
    return query_greater(LfTree(inx), lf, rt, x) + query_greater(RtTree(inx), lf, rt, x); // 将左右子树的结果相加
}

void init(){
    tree.reserve(N * 4);
    arr.resize(N + 1);
    for(int i = 1; i <= N; i ++){
        cin >> arr[i];
    }
    build(1, 1, N);
}
void solve(void){
    cin >> N;
    init();
}

signed main(void){
    ios::sync_with_stdio(false);
    solve();
    return 0;
}
```

---

## 树状数组

```c++
int lowbit(int x){return x & -x;} // 获取二进制最后一个1后的数值，例如 110 -> 10 也就是6 -> 2

int tree[N];
void update(int x, int d){ // 更新元素a[x] += d;
    while(x <= N){
        tree[x] += d;
        x += lowbit(x);
    }
}
int sum(int x){ // 返回前缀和
    int ans = 0;
    while(x > 0){
        ans += tree[x];
        x -= lowbit(x);
    }
    return ans;
}
```

---

# 二分

## 实数二分

> [一元三次方程求解](https://www.lanqiao.cn/problems/764/learning/?page=1&first_category_id=1&problem_id=764)

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

double a, b, c, d; // 多项式各系数
double handle(double x){ // 计算所像是所得值
    return a * pow(x, 3) + b * pow(x, 2) + c * x + d;
}

void solve(void){
    cin >> a >> b >> c >> d;
    for(int i = -100; i < 100; i ++){ // 题目给出的答案范围
        double x1 = i, x2 = i + 1;  // 假设连个解
        double y1 = handle(x1), y2 = handle(x2); // x1,x2带入得到的y值
        if(y1 == 0){ // x1为所求解之一
            cout << fixed << setprecision(2) << x1 << " ";
            continue;
        }
        if(y1 * y2 >= 0) continue;
        for(int j = 0; j < 100; j ++){ // 二分答案，不断逼近正确答案
            double mid = (x1 + x2) / 2.0;
            if(handle(mid) * handle(x2) <= 0){
                x1 = mid;
            }else{
                x2 = mid;
            }
        }
        cout << fixed << setprecision(2) << x2 << " ";
    }
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

---

# 字符串处理

## 常用函数

### find

- **str.find(s)**
    返回str中第一次出现s子串的位置。
- **str.find(s, pos)**
    返回str中从pos位置开始，第一次出现s子串的位置。
注：没有找到则返回 std::string::npos (-1).

### replace

- **str.replace(pos, len, s)**
    从 pos 位置开始将 len 长度的字符替换为 s.

### substr

- **str.substr(pos, len)**
    返回 str 从 pos 开始长度为 len 的子串。

### insert

- **str.insert(pos, s)**
    在 str 中 pos 位置添加子串 s.

## 字符串的操纵（字符串流）

> **流：** 流是一种抽象的概念，代表数据的来源和目的地。流可以是文件、控制台、内存中的字符串等。通过流，我们可以进行数据的输入（读取）和输出（写入）操作。标准 IO 库提供了一系列的流类，如 iostream、fstream 等，用于处理不同类型的流。

### 字符串流的定义

字符串流是一种特殊的流，它以字符串作为数据的来源或目的地。  
C++ 标准 IO 库提供了三个主要的字符串流类：
- **istringstream**：用于从字符串中读取数据，类似于从文件或控制台读取数据。
- **ostringstream**：用于将数据写入字符串，类似于将数据写入文件或控制台。
- **stringstream**：既可以从字符串中读取数据，也可以将数据写入字符串，兼具 istringstream 和 ostringstream 的功能。

### stringstream

> stringstream 类兼具 istringstream 和 ostringstream 的功能，既可以从字符串中读取数据，也可以将数据写入字符串。(需要包含<sstream>头文件)

```c++
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
#define endl '\n'

int main() {
    string lines = "a 25 123456";
    stringstream inps(lines); // inps 是变量名
    int age;
    string name, id;
    inps >> name >> age >> id; // 将字符串中的数据存入定义好的变量中
    cout << "name:" << name << " age:" << age << " id:" << id << endl;
    
    stringstream outs; outs.str().reserve(1024); // 初始化并预留1024字节的空间
    int year = 2022, month = 3, day = 24;
    outs.width(4); outs.fill('0'); outs << year; // 设置下次输入的宽度为4，并用‘0’预先存储，向其中存入年份
    outs.width(2); outs.fill('0'); outs << month;
    outs.width(2); outs.fill('0'); outs << day;
    cout << outs.str() << endl;
    return 0;
}
```

--- 

## 正则表达式-regex

需要包括头文件\<regex>

### std::regex

表示一个正则表达式对象。正则表达式对象可以用来存储和表示一个特定的正则表达式模式。

- **.** ：匹配任意单个字符（换行符除外）。
- **\[ ]** ：匹配方括号内的任意一个字符。例如，[abc] 匹配 'a'、'b' 或 'c'。
- **^** ：在方括号内使用时，表示取反。例如，[^abc] 匹配除 'a'、'b'、'c' 之外的任意字符。
- **\d** ：匹配任意数字，等价于 [0-9]。
- **\s** ：匹配任意空白字符，包括空格、制表符、换行符等。
- **\w** ：匹配任意字母、数字或下划线，等价于 [a-zA-Z0-9_]。

**常用量词** 
- **\*** : 匹配前面的模式零次或无数次。
- **\+** : 匹配前面的模式一次或多次。
- **？** ： 匹配前面的模式零次或一次。
- **{n}** : 匹配前面的模式恰好 n 次。
- **{n,}** : 匹配前面的模式 至少 n 次。
- **{n, m}** : 匹配前面的模式至少 n 次至多 m 次。

**常用锚点** 
- **^** : 匹配字符串的开始。
    - ^abc 可以匹配以"abc"开头的字符串。
- **$** : 匹配字符串的结束。
    - abc$ 可以匹配以"abc"结尾的字符串。
- **\b** : 匹配单词的边界。
  - \bcan\b 可以匹配单独的"can"单词。
- **\B** : 匹配非单词边界。
  - \Bcan\B 可以匹配于其他单词内部的"can"。

**分组**
分组用来将模式的匹配结果进行分组，并对每个分组进行单独的处理。用 **( )** 表示。
- (ab)+ 可以匹配 "ab" "abab" "ababab"。
- (a|b) 可以匹配 "a" 或者 "b"。 

```c++
#include <iostream>
#include <regex>
using namespace std;

int main(void){
    string str = "This is a string. 123456789";
    // 创建一个正则表达式对象，pattern为变量名。
    regex pattern("\\d+"); // 匹配一个或多个数字， 括号中也可写作 "[0-9]+"

    return 0;
}
```

### std::regex_match

用于检查整个字符串是否于表达式匹配。

```c++
// 形式 1：仅检查是否匹配
bool regex_match(const char* str, const std::regex& re);
bool regex_match(const std::string& str, const std::regex& re);

// 形式 2：检查匹配并存储匹配结果
template <class BidirIt, class Alloc, class CharT, class Traits>
bool regex_match(
                BidirIt first, 
                BidirIt last,
                std::match_results<BidirIt, Alloc>& m,
                const std::basic_regex<CharT, Traits>& e,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main(void) {
    std::string str = "abc123";
    std::regex pattern("abc\\d+");

    if (std::regex_match(str, pattern)) {
        std::cout << "整个字符串匹配正则表达式" << std::endl;
    } else {
        std::cout << "整个字符串不匹配正则表达式" << std::endl;
    }

    return 0;
}
```

### std::regex_search

用于在字符串中查找第一个与正则表达式匹配的子串。

```c++
// 形式 1：仅检查是否存在匹配的子串
bool regex_search(const char* str, const std::regex& re);
bool regex_search(const std::string& str, const std::regex& re);

// 形式 2：查找匹配并存储匹配结果
template <class BidirIt, class Alloc, class CharT, class Traits>
bool regex_search(
                BidirIt first, 
                BidirIt last,
                std::match_results<BidirIt, Alloc>& m,
                const std::basic_regex<CharT, Traits>& e,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );
```

例子：

```c++
#include <bist/stdc++.h>
using namespace std;

// 普通版
void solve_normal(void){
    string str = "This is a string. 1234567890";
    regex pattern("[0-9]+"); // 匹配一个或多个数字
    smatch matches; // 用于存储匹配的结果。
    if(regex_search(str, matches, pattern)){
        cout << "Found number: " << matches.str() << endl;
    }else{
        cout << "No match found." << endl;
    }
}
// 加强版
void solve_difficult(void){
    string str = "This is a string. 123-456-7890";
    regex pattern("(\\d{3})-(\\d{3})-{\\d{4}}");
    smatch matches;
    if(regex_search(str, matches, pattern)){
        // matches.str(0) 表示返回整个匹配的字符串。
        cout << "Found number: " << matches.str(0) << endl; 
        // matches.str(i) (i > 0) 表示返回第 i 个捕获组的匹配结果。
        for(size_t i = 1; i < matches.size(); i ++){
            cout << "捕获组" << i << ": " << matches.str(i) << endl;
        }   
    }else{
        cout << "Not find." << endl;
    }
}

int main(void){
    cout << "普通版：" << endl;
    solve_normal();

    cout << "加强版: " << endl;
    solve_difficult();

    return 0;
}
```
注：**std::smatch** 对象可以存储多个匹配结果，包括整个匹配的字符串以及各个捕获组的匹配结果。
**smatch 的常用成员函数：**
- **smatch.size()** : 返回匹配结果的数量，包括整个匹配和各个匹配组。
- **smatch.empty()** : 判断匹配结果是否为空。true 表示为空；
- **smatch.prefix()** : 返回匹配结果之前的字符串。
- **smatch.suffix()** : 返回匹配结果之后的字符串。
- **smatch.position()** : 返回匹配结果在原字符串的起始位置。
  - **smatch.position(i)** : (i>0)返回第 i 个捕获组匹配结果在原数组中的起始位置。  
- **smatch.length()** : 返回匹配结果的字符长度。

注：若可没有匹配结果，调用 position 会报错。预先判断是否有匹配结果。

### std::regex_replace

用于替换字符串中与正则表达式匹配的子串。

```c++
// 形式 1：返回替换后的字符串
template <class OutputIt, class BidirIt, class CharT, class Traits, class ST, class SA>
OutputIt regex_replace(
                OutputIt out,
                BidirIt first,
                BidirIt last,
                const std::basic_regex<CharT, Traits>& e,
                const std::basic_string<CharT, ST, SA>& fmt,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );

// 形式 2：返回替换后的字符串
template <class CharT, class Traits, class ST, class SA, class Fmt>
std::basic_string<CharT, ST, SA> regex_replace(
                const std::basic_string<CharT, ST, SA>& s,
                const std::basic_regex<CharT, Traits>& e,
                Fmt&& fmt,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );

```


```c++
#include <bits/stdc++.h>;

int main() {
    std::string str = "hello, 123 world";
    std::regex pattern("\\d+");

    std::string result = std::regex_replace(str, pattern, "###");
    std::cout << "替换后的字符串: " << result << std::endl;

    return 0;
}
```

### sregex_iterator

主要用于遍历字符串中所有与给定正则表达式匹配的子串。

```c++
std::sregex_iterator(
                const BidirectionalIterator first, 
                const BidirectionalIterator last,
                const std::basic_regex<CharT, Traits>& re,
                std::regex_constants::match_flag_type flags =  std::regex_constants::match_default
                );
/*
first,last: 表示要搜索的字符串范围，通常是字符串的起始和结束迭代器。
re: 要匹配的正则表达式。
flags: 匹配标志，用于指定匹配的行为，默认为 std::regex_constants::match_default 。
*/
```

例子(不推荐，可能出错，建议用regex_search)：

```c++
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string text = "The cat sat on the mat. The cat is cute.";
    std::regex pattern("\\b(cat)\\b"); // 匹配整个单词 "cat"

    std::sregex_iterator it(text.begin(), text.end(), pattern);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        std::cout << "匹配到的子串: " << match.str() << "，起始位置: " << match.position() << std::endl;
        ++it;
    }

    return 0;
}

```

### 应用

#### 替换指定数量个匹配子串

```c++
#include <iostream>
#include <regex>
#include <string>

// 替换原字符串中指定数量的匹配结果
std::string replaceSpecificMatches(const std::string& input, const std::regex& pattern, 
                                   const std::string& replacement, int count) {
    std::string result = input;
    int replacedCount = 0;
    std::sregex_iterator it(result.begin(), result.end(), pattern);
    std::sregex_iterator end;

    while (it != end && replacedCount < count) {
        std::smatch match = *it;
        result.replace(match.position(), match.length(), replacement);
        // 重新创建迭代器，因为字符串已被修改
        it = std::sregex_iterator(result.begin(), result.end(), pattern);
        replacedCount++;
    }
    return result;
}

int main() {
    std::string text = "The cat sat on the mat. The cat is cute.";
    std::regex pattern("\\b(cat)\\b");
    std::string replacement = "dog";
    int replaceCount = 1;

    std::string newText = replaceSpecificMatches(text, pattern, replacement, replaceCount);
    std::cout << "替换 " << replaceCount << " 次后的字符串: " << newText << std::endl;

    return 0;
}    
```

## 字符串匹配-KMP

> KMP算法是一种在任何情况下都能达到 O(n + m) 复杂度的算法。

### template

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 5;
int Next[N];
void getNext(string p){ // 计算 Next[1]~Next[plen]
    Next[0] = 0; Next[1] = 0;
    for(int i = 1; i < p.size(); i ++){ // 把 i 的增长看成后缀的逐步扩展
        int j = Next[i];    // j 的后缀：j 指向前缀阴影 w(匹配的字符串)的后一个字符
        while(j && p[i] != p[j]){   // 阴影的后一个字符不相同
            j = Next[j];    // 更新 j
        }
        if(p[i] == p[j]){
            Next[i + 1] = j + 1;
        }else{
            Next[i + 1] = 0;
        }
    }
}
int kmp(string s, string p){ // 在 s 种找 p，return 最长匹配长度
    int ans = 0;
    int slen = s.size(), plen = p.size();
    int j = 0;
    for(int i = 0; i < slen; i ++){ // 匹配 s 和 p 的每个字符
        while(j && s[i] != p[j]){   // 失配
            j = Next[j];    // j 滑动到 Next[j] 的位置
        }
        if(s[i] == p[i]){ // 当前位置的字符匹配
            j ++;   // 匹配
            ans = max(ans, j);  // 更新最长匹配长度
        }
        if(j == plen){
            // 这个匹配，在 s 种的起点是 i + 1 - plen，末尾是 i。
            // cout << "startIndex = " << endl;
            return ans; // 最长匹配长度。
        }
    }
}
signed main(void){
    string s, t; cin >> s >> t;
    getNext(t);
    cout << kmp(s, t);

    return 0;
}
```

# 二叉树

## 中序遍历与后序遍历构造二叉树

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
    int inx; // 当前为后序遍历的索引位置
    unordered_map<int, int> mp; // 中序遍历每个元素的索引
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        inx = postorder.size() - 1; // 初始化后序遍历的索引
        for(int i = 0; i <= inx; i ++){ // 初始化mp0
            mp[inorder[i]] = i;
        }
        return solve(0, inx, inorder, postorder);
    }
    TreeNode* solve(int lf, int rt, vector<int>& inorder, vector<int>& postorder){
        if(lf > rt) return nullptr; // 叶子节点的子节点为空
        int flag = postorder[inx --]; // 当前的根节点
        TreeNode* root = new TreeNode(flag); // 构建根节点
        int mid = mp[flag]; 
        root->right = solve(mid + 1, rt, inorder, postorder); // 根节点的右节点
        root->left = solve(lf, mid - 1, inorder, postorder); // 根节点的左节点
        return root; // 返回根节点
    }
};
```

## 根据二叉搜索树的前序遍历，输出后序遍历

```c++
# include <bits/stdc++.h>
using namespace std;

int n, inx = 0; // 节点数， 后序遍历目前的索引位置
vector<int> pre; // 前序遍历
vector<int> suf; // 后序遍历
bool is_mirrow, is_tree = true; // 是否是镜像， 是否是二叉搜索树

void build(int lf, int rt){
    if(!is_tree || lf > rt) return;  // 不是二叉搜索树， lf > rt
    if(lf == rt){ // 叶子结点
        suf[inx ++] = pre[lf]; // 添加到后序数组
        return;
    }
    int root = pre[lf]; // 根节点
    int i = lf; // 寻找左右子树的分界点
    bool lf_true, rt_true;
    for(i = lf; i <= rt; i ++){
        lf_true = true, rt_true = true; // 确定左右子树是否正确
        for(int j = lf + 1; j <= i; j ++){ // 检查左子树
            if(is_mirrow ? pre[j] < root : pre[j] >= root){
                lf_true = false;
                break;
            }
        }
        for(int j = i + 1; j <= rt; j ++){ // 检查右子树
            if(is_mirrow ? pre[j] >= root : pre[j] < root){
                rt_true = false;
                break;
            }
        }
        if(lf_true && rt_true) break; // 找到了合法的分界点
    }
    if(!lf_true || !rt_true){ // 没有找到合法分界点
        is_tree = false;
        return;
    }
    build(lf + 1, i); // 递归处理左子树
    build(i + 1, rt); // 递归处理右子树
    suf[inx ++] = root; // 根节点放入后序数组
}

void init(){
    cin >> n;
    pre.resize(n, 0);
    suf.resize(n, 0);
    for(auto& it : pre) cin >> it;
    if(n > 1) is_mirrow = pre[0] <= pre[1];
}


void solve(){
    init();
    if(n == 1){
        cout << "YES" << endl;
        cout << pre.front();
        return;
    }
    build(0, n - 1);
    cout << (is_tree ? "YES" : "NO") << endl;
    if(is_tree){
        for(int i = 0; i < n; i ++){
            if(i == 0){
                cout << suf[i];
                continue;
            }
            cout << " " << suf[i];
        }
    }

}

signed main(void){
    solve();

    return 0;
}
```


---

# 随记

## 全排列

### 普通全排列

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4
};

void dfs(int s, int t){ // 从第s个数开始到第k个数结束的全排列
    if(s == t){
        for(int i = 0; i <= t; i ++) cout << target[i] << " ";  // 输出一个排列
        cout << endl;
        return;
    }
    for(int i = s; i <= t; i ++){
        swap(target[s], target[i]); // 第1个数和后面的数交换
        dfs(s + 1, t);
        swap(target[s], target[i]); // 回溯
    }
}
/*
1 2 3
1 3 2 
2 1 3 
2 3 1 
3 2 1 
3 1 2
*/

void solve(void){
    int n = target.size();
    dfs(0, n - 1);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

### 从小到大输出排列

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4, 5
};
vector<bool> vis(20, false);    //第i个数是否被用过
vector<int> res(20, 0);    // 生成的一个全排列

void dfs(int s, int t){ // 从第s个数开始到第k个数结束的全排列
    if(s == t){
        for(int i = 0; i < t; i ++) cout << res[i] << " ";  // 输出一个排列
        cout << endl;
        return;
    }
    for(int i = 0; i < t; i ++){
        if(!vis[i]){
            vis[i] = true;
            res[s] = target[i];
            dfs(s + 1, t);
            vis[i] = false;
        }
    }
}

void solve(void){
    int n = target.size();
    dfs(0, n);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}          
```

---

## 组合

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4, 5
};
vector<bool> vis(20, false);    //第i个数是否被用过

void dfs(int k, int n){ // dfs到k个数
    if(k == n){
        for(int i = 0; i < n; i ++){
            if(vis[i]) cout << target[i] << '-';
        }
        cout << endl;
        return;
    }
    vis[k] = false; // 不选第k个数
    dfs(k + 1, n);
    vis[k] = true;  // 选这个数
    dfs(k + 1, n);
}

void solve(void){
    int n = target.size();
    dfs(0, n);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

---

## 高精度加法

```c++
string add(string a, string b){
    string s = "";
    int op = 0;
    for(int i = a.size() - 1, j = b.size() - 1; i >= 0 || j >= 0 || op > 0; i --, j --){
        if(i >= 0) op += a[i] - '0';
        if(j >= 0) op += b[j] - '0';
        s += to_string(op % 10);
        op /= 10;
    }
    reverse(s.begin(), s.end());
    return s;
}
```

---

## 阶乘
```c++
const int N = 1e4;
vector<int> arr;

void func(int n){
    arr.resize(N, 0); 
    arr[0] = 1;
    for(int i = 1; i <= n; i ++){
        int op = 0;
        for(int j = 0; j < N; j ++){
            arr[j] = arr[j] * i + op;
            op = arr[j] / 10;
            arr[j] = arr[j] % 10;
        }
    }
    // show;
    int last;
    for(int i = N - 1; i >= 0; i --){
        if(arr[i] != 0){
            last = i;
            break;
        }
    }
    for(int i = last; i >= 0; i --) cout << arr[i];
}
```

---

## 除法模拟

>**题目**：
这里所谓的“光棍”，并不是指单身汪啦~ 说的是全部由1组成的数字，比如1、11、111、1111等。传说任何一个光棍都能被一个不以5结尾的奇数整除。比如，111111就可以被13整除。 现在，你的程序要读入一个整数x，这个整数一定是奇数并且不以5结尾。然后，经过计算，输出两个数字：第一个数字s，表示x乘以s是一个光棍，第二个数字n是这个光棍的位数。这样的解当然不是唯一的,题目要求你输出最小的解。
**提示**：一个显然的办法是逐渐增加光棍的位数，直到可以整除x为止。但难点在于，s可能是个非常大的数 —— 比如，程序输入31，那么就输出3584229390681和15，因为31乘以3584229390681的结果是111111111111111，一共15个1。
**输入格式：**
输入在一行中给出一个不以5结尾的正奇数x（<1000）。
**输出格式：**
在一行中输出相应的最小的s和n，其间以1个空格分隔。
**输入样例：**
31
**输出样例：**
3584229390681 15

```c++
#include<stdio.h>
int main(){
    int n, r = 1, w = 1;//r表示1,11,111类型的数据，w记录位数
    scanf("%d", &n);
    while(r < n){
        r *= 10;
        r++;
        w++;
    }
    while(1){
        printf("%d", r/n);//输出商
        r %= n;//取余
        if(r == 0)//取余后，若等于0，则证明能被整除，break掉
            break;
        r = r * 10 + 1;//不等于0则在余数后一位加上1
        w++;
    }
    printf(" %d",w);
    return 0;
} 
```

other  

```c++
int a, b, e, len = 0; cin >> a >> b >> e;
string res = "";
res += to_string(a/b) + ".";
a = a % b;
a *= 10;
while(a < b){
    len ++;
    res += "0";
    a *= 10;
    if(len > e){
        res.pop_back();
        cout << res;
        return;
    }
}
while(true){
    res += to_string(a / b);
    a %= b;
    len ++;
    if(len > e){
        if(res.back() >= '5'){
            int op = 1, inx = res.size() - 2;
            res.pop_back();
            int flag;
            while(op == 1){
                if(res[inx] == '.') inx --;
                flag = res[inx] - '0' + op;
                op = flag / 10;
                res[inx] = '0' + flag % 10;
                inx --;
                if(op == 1 && inx == -1){
                    res = "1" + res;
                    break;
                }
            }
        }else{
            res.pop_back();
        }
        cout << res << endl;
        return;
    }
    a *= 10;
}
```

---

## 后缀表达式（逆波兰式）

**定义：** 指不包含括号，运算符放在两个运算对象的后面，所有的运算按运算符出现的顺序，严格从左向右进行（不考虑运算符的优先级规则）。  

**计算：** 从左往右扫描表达式，遇到数字时，将数字压入栈中，遇到运算符时，弹出栈顶的两个数，用运算符对他们做相应的计算，并将结果入栈。  

**例：** 2 3 + 4 * 5 -  ==> 15  

**注：** 由于后缀表达式忽略了括号，所以在转化为中缀表达式后主意括号的影响。

```c++
# include <bits/stdc++.h>
using namespace std;
#define endl '\n'

// 优先级， 越高优先级越高
map<char, int> priority = {
    {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'(', 0}, {')', 0}
};

// 比较运算符的优先级
bool judge(char a, char b){ 
    // a <= b;
    return priority[a] <= priority[b];
}

// 将中序表达式转换为后序表达式
string getPostfix(string infix){
    // 操作数栈和运算符栈
    stack<string> operands;
    stack<char> operators;
    string num = "";
    for(auto ch : infix){
        // 获取数字
        if(isdigit(ch)){ 
            num.push_back(ch);
            continue;
        }
        // 将数字压入操作数栈
        if(!num.empty()){ 
            operands.emplace(num);
            num.clear();
        }
        // 左括号压入运算符栈中
        if(ch == '('){ 
            operators.emplace(ch);
            continue;
        }
        // 右括号
        if(ch == ')'){
            // 弹出运算符栈顶元素，直至遇到左括号
            while(!operators.empty() && operators.top() != '('){
                operands.emplace(string(1, operators.top()));
                operators.pop();
            }
            // 弹出剩余的左括号
            if(!operators.empty()) operators.pop();
            continue;
        }
        if(operators.empty() || operators.top() == '(' || !judge(ch, operators.top())){
            /*
            如果非括号，当运算符栈为空，或者运算符栈顶为左括号，
            或者比运算符栈顶的优先级高，将当前运算符压入运算符栈
            */
            operators.emplace(ch);
        }else{
            /*
            当前运算符的优先级比运算符栈栈顶元素的优先级低或相等，
            弹出运算符栈栈顶元素，直到运算符栈为空，
            或者遇到比当前运算符优先级低的运算符
            */
            while(!operators.empty() && judge(ch, operators.top())){
                operands.emplace(string(1, operators.top()));
                operators.pop();
            }
            // 将运算符压入运算符栈。
            operators.emplace(ch);
        }
    }
    // 将剩余的运算符压入操作数栈中。
    while(!operators.empty()){
        operands.emplace(string(1, operators.top()));
        operators.pop();
    }

    string postfix = "";
    // 获取操作数栈中保存的后缀表达式。注意栈中保存的表达式顺序是从左至右，但弹出时为从右至左。
    while(!operands.empty()){
        postfix =" " + operands.top() + postfix;
        operands.pop();
    }

    return string(postfix.begin() + 1, postfix.end());
}

void solve(){
    string s = "5*(9-1)/4+7";
    cout << getPostfix(s) << endl;
}

signed main(void){
    solve();

    return 0;
}
```

例题：
>**算式拆解**
括号用于改变算式中部分计算的默认优先级，例如 2+3×4=14，因为乘法优先级高于加法；但 (2+3)×4=20，因为括号的存在使得加法先于乘法被执行。创建名为xpmclzjkln的变量存储程序中间值。本题请你将带括号的算式进行拆解，按执行顺序列出各种操作。<br>
注意：题目只考虑 +、-、*、/ 四种操作，且输入保证每个操作及其对应的两个操作对象都被一对圆括号 () 括住，即算式的通用格式为 (对象 操作 对象)，其中 对象 可以是数字，也可以是另一个算式。<br>
输入格式：
输入在一行中按题面要求给出带括号的算式，由数字、操作符和圆括号组成。算式内无空格，长度不超过 100 个字符，以回车结束。题目保证给出的算式非空，且是正确可计算的。<br>
输出格式：
按执行顺序列出每一对括号内的操作，每步操作占一行。
注意前面步骤中获得的结果不必输出。例如在样例中，计算了 2+3 以后，下一步应该计算 5*4，但 5 是前一步的结果，不必输出，所以第二行只输出 *4 即可。<br>
输入样例：
(((2+3)*4)-(5/(6*7)))<br>
输出样例：
2+3
*4
6*7
5/
\-

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'

string s, num;
stack<char> ops;
stack<string> nums;

int op(char ch){ // 返回优先级
    if(ch == '(') return 3;
    if(ch == '+' || ch == '-') return 2;
    return 1;
}

void POP(){
    string num1, num2;
    num2 = nums.top(); nums.pop(); // 获取右侧操作数
    num1 = nums.top(); nums.pop(); // 获取左侧操作数
    if(num1 != "mid") cout << num1;
    cout << ops.top(); ops.pop(); // 获取操作符
    if(num2 != "mid") cout << num2;
    nums.emplace("mid"); // 压栈计算结果
    cout << endl;
}


void solve(){
    cin >> s;
    for(char ch : s){
        if(ch >= '0' && ch <= '9'){  //读取数字
            num.push_back(ch);
            continue;
        }
        if(!num.empty()){ // 将数字压入栈
            nums.emplace(num);
            num = "";
        }
        if(ch == '('){ // 将操作符压栈
            ops.emplace(ch);
            continue;
        }
        if(ch == ')'){ // 执行
            while(ops.top() != '('){ // 执行，直到遇到第一个 (
                POP();
            }
            ops.pop(); // 将(弹出
            continue;
        }
        while(!ops.empty() && op(ops.top()) <= op(ch)){ // 已有操作符的优先级更高或相等
            POP();
        }
        ops.emplace(ch); // 该操作符压栈
    }
    while(!ops.empty()){ // 执行完剩下的操作
        POP();
    }
    return;
}

signed main(void){
	ios::sync_with_stdio(false);
	int t = 1;
//	cin >>t;
	while(t --){
		solve();
	}
	
	return 0;
}

```

---

## 快速幂

```c++
int fastPow(int a, int n){ // a^n
    int ans = 1;
    while(n){
        if(n & 1) ans *= a;
        a *= a;
        n >>= 1;
    }
    return ans;
}
```

---


## 整除分块

> 对于给定的正整数 n ,  n = k * i + r (0 <= r < i)。当 i 在一定范围内变化时，\(\lfloor\frac{n}{i}\rfloor\)(向下取整)会有很多重复情况。例如：\(n = 10\) 时，\(\lfloor\frac{10}{1}\rfloor = 10\)，\(\lfloor\frac{10}{2}\rfloor = 5\)，\(\lfloor\frac{10}{3}\rfloor=\lfloor\frac{10}{4}\rfloor = 2\)，\(\lfloor\frac{10}{5}\rfloor = 2\)，\(\lfloor\frac{10}{6}\rfloor=\lfloor\frac{10}{7}\rfloor=\lfloor\frac{10}{8}\rfloor=\lfloor\frac{10}{9}\rfloor=\lfloor\frac{10}{10}\rfloor = 1\)。可以发现，\(\lfloor\frac{n}{i}\rfloor\) 的值会呈现出块状分布的特点，相同值的 \(i\) 会形成一个块。

### 算法实现

- 假设要计算\(\sum_{i = 1}^{n}\lfloor\frac{n}{i}\rfloor\)，可以通过整除分块来优化计算。
- 对于每个块，设块的左端点为 \(l\)，右端点为 \(r\)。当 \(i = l\) 时，\(\lfloor\frac{n}{l}\rfloor\) 的值确定，而该块的右端点 \(r\) 可以通过 \(r=\lfloor\frac{n}{\lfloor\frac{n}{l}\rfloor}\rfloor\) 计算得出。这样就可以在 \(O(\sqrt{n})\) 的时间复杂度内计算出上述求和式子的值。
- 
#### 公式

\(r=\lfloor\frac{n}{\lfloor\frac{n}{l}\rfloor}\rfloor\)

### 代码示例

```cpp
#include <iostream>
#include <cmath>

using namespace std;

int main() {
    int n;
    cin >> n;
    int ans = 0;
    for (int l = 1, r; l <= n; l = r + 1) {
        // 计算当前块的右端点
        r = n / (n / l);
        // 累加当前块的值
        ans += (r - l + 1) * (n / l);
    }
    cout << ans << endl;
    return 0;
}
```

---

## 常用集合算法

### 交集 set_intersection

set_intersection 函数用于求两个**有序序列**的交集，并将交集结果存储到一个**新的容器**中。  
返回一个迭代器，指向交集结果在新容器中的**最后一个元素的下一个位置**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_intersection( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first
                );

// first1 和 last1：表示第一个有序序列的起始和结束迭代器。
// first2 和 last2：表示第二个有序序列的起始和结束迭代器。
// d_first：表示存储交集结果的目标容器的起始迭代器。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    // 输入的容器必须是有序的。
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间, 否则会报错
    result.resize(min(vec1.size(), vec2.size()));

    // 计算交集
    auto it = set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    // 调整结果容器的大小以匹配实际的交集元素数量
    result.resize(it - result.begin());

    cout << "交集: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}    
```

### 并集 set_union

set_union 函数用于求两个**有序序列**的并集，并将并集结果**存储到一个新的容器中**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_union( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first 
                );
// 参数含义与 set_intersection 类似。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间
    result.resize(vec1.size() + vec2.size());

    auto it = set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    result.resize(it - result.begin());

    cout << "并集: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}

```

### 差集 set_difference

set_difference 函数用于求两个**有序序列**的差集（即第一个序列中存在而第二个序列中不存在的元素），并将差集结果**存储到一个新的容器中**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_difference( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first 
                );
// 参数含义与前面两个函数类似。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间
    result.resize(vec1.size() + vec2.size());

    auto it = set_difference(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    result.resize(it - result.begin());

    cout << "差集 (vec1 - vec2): ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << std::endl;

    return 0;
}

```