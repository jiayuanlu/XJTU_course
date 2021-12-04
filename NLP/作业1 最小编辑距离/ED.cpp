// #include<iostream>
// #include<cstring>
// #include<cmath>
#include<bits/stdtr1c++.h>
using namespace std;
#define INF 1000000
// int ED(char a[],char b[]);
int main()
{
    int distance[1005][1005];
    char a[1005],b[1005];
    // cout<<ED(a,b);
    
// }
// int ED(char a[],char b[])
// {
    while(cin>>a>>b)
    {
    int l1=strlen(a+1);
    int l2=strlen(b+1);
    for(int i=1;i<=l1;i++)
    {
        for(int j=1;j<=l2;j++)
        {
            distance[i][j]=INF;
        }
    }
    for(int i=1;i<=l1;i++)
    {
        distance[i][0]=i;
    }
    for(int j=1;j<=l2;j++)
        distance[0][j]=j;
    distance[0][0]=0;
    for(int i=1;i<=l1;i++)
    {
        for(int j=1;j<=l2;j++)
        {
            int flag;
            if(a[i]==b[i])
                flag=0;
            else
                flag=1;
            distance[i][j]=min(distance[i-1][j]+1,min(distance[i][j-1]+1,distance[i-1][j-1]+flag));
        }
    }
    cout<<distance[l1][l2]<<endl;
    }
    return 0;
}