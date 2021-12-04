#include<iostream>
#include<cstring>
using namespace std;
int min(int a,int b,int c)
{
    if(a>=b)
    {
        if(c>=b) return b;
        else return c;
    }
    else
    {
        if(c<=a) return c;
        else return a;
    }
}

int main()
{
    char a[1001],b[1001];
    a[0]='#';
    b[0]='#';
    int mindis[1001][1001];
    int flag[1001][1001];
    int al=0,bl=0;
    int i=0,j=0;
    cout<<"Please input the source and target:"<<endl;
    while(cin>>a>>b)
    {
        al=strlen(a+1);
        bl=strlen(b+1);
        for(i=0;i<=al+1;i++)
        {
            for(j=0;j<=bl;j++)
            {
                mindis[i][j]=10000000;
            }
        }
        for(i=0;i<=al+1;i++)
        {
            for(j=0;j<=bl;j++)
            {
                flag[i][j]=0;
            }
        }
        for(i=0;i<=al+1;i++)
        {
            mindis[i][0]=i;
        }
        for(j=0;j<=bl+1;j++)
        {
            mindis[0][j]=j;
        }
        for(i=1;i<=al+1;i++)
        {
            for(j=1;j<=bl+1;j++)
            {
                if(a[i-1]==b[j-1])
                {
                    mindis[i][j]=min(mindis[i-1][j]+1,mindis[i][j-1]+1,mindis[i-1][j-1]);
                    if(mindis[i][j]==mindis[i-1][j]+1) flag[i][j]=100;
                    if(mindis[i][j]==mindis[i][j-1]+1) flag[i][j]+=20;
                    if(mindis[i][j]==mindis[i-1][j-1]) flag[i][j]+=3;
                }
                else
                {
                    mindis[i][j]=min(mindis[i-1][j]+1,mindis[i][j-1]+1,mindis[i-1][j-1]+2);
                    if(mindis[i][j]==mindis[i-1][j]+1) flag[i][j]=100;
                    if(mindis[i][j]==mindis[i][j-1]+1) flag[i][j]+=20;
                    if(mindis[i][j]==mindis[i-1][j-1]+2) flag[i][j]=3;
                }
            }
        }
        for (i=0;i<=al+1;i++)
        {
            for(j=0;j<=bl+1;j++)
            {
                cout<<mindis[i][j]<<" ";
            }
            cout<<endl;
        }
        for (i=0;i<=al+1;i++)
        {
            for(j=0;j<=bl+1;j++)
            {
                cout<<flag[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<mindis[al+1][bl+1]<<endl;
        cout<<a<<endl;
        cout<<b<<endl;
        int x=0,y=0;
        i=al+1;
        j=bl+1;
        cout<<mindis[al+1][bl+1]<<" ";
        while(i>=1&&j>=1)
        {
            if(flag[i][j]==3)
            {
                if(mindis[i][j]-mindis[i-1][j-1]==2)
                    cout<<"substitute: "<<a[i-1]<<" to "<<b[i-1]<<endl;
                i--;j--;
            }
            else if(flag[i][j]==100)
            {
                if(mindis[i][j]-mindis[i-1][j]==1)
                    cout<<"delete:"<<a[i-1]<<endl;
                i--;
            }
            else if(flag[i][j]==20)
            {
                if(mindis[i][j]-mindis[i][j-1]==1)
                    cout<<"insert:"<<b[j-1]<<endl;
                j--;
            }
            else if(flag[i][j]==23)
            {
                if(mindis[i][j-1]<=mindis[i-1][j-1])
                {
                    if(mindis[i][j]-mindis[i][j-1]==1)
                        cout<<"insert:"<<b[j-1]<<endl;
                    j--;
                }
                else
                {
                    if(mindis[i][j]-mindis[i-1][j-1]==2)
                        cout<<"substitute: "<<a[i-1]<<" to "<<b[i-1]<<endl;
                    i--;j--;
                }
            }
            else if(flag[i][j]==103)
            {
                if(mindis[i-1][j]<=mindis[i-1][j-1])
                {
                    if(mindis[i][j]-mindis[i-1][j]==1)
                        cout<<"delete:"<<a[i-1]<<endl;
                    i--;
                }
                else
                {
                    if(mindis[i][j]-mindis[i-1][j-1]==2)
                        cout<<"substitute: "<<a[i-1]<<" to "<<b[i-1]<<endl;
                    i--;j--;
                }
            }
            else if(flag[i][j]==120)
            {
                if(mindis[i-1][j]<=mindis[i][j-1])
                {
                    if(mindis[i][j]-mindis[i-1][j]==1)
                        cout<<"delete:"<<a[i-1]<<endl;
                    i--;
                }
                else
                {
                    if(mindis[i][j]-mindis[i][j-1]==1)
                        cout<<"insert:"<<b[j-1]<<endl;
                    j--;
                }
            }
            else if(flag[i][j]==123)
            {
                int k=min(mindis[i-1][j-1],mindis[i-1][j],mindis[i][j-1]);
                if(k==mindis[i-1][j-1])
                {
                    if(mindis[i][j]-mindis[i-1][j-1]==2)
                        cout<<"substitute: "<<a[i-1]<<" to "<<b[i-1]<<endl;
                    i--;j--;
                }
                else if(k==mindis[i-1][j])
                {
                    if(mindis[i][j]-mindis[i-1][j]==1)
                        cout<<"delete:"<<a[i-1]<<endl;
                    i--;
                }
                else if(k==mindis[i][j-1])
                {
                    if(mindis[i][j]-mindis[i][j-1]==1)
                        cout<<"insert:"<<b[j-1]<<endl;
                    j--;
                }
            }
            cout<<mindis[i][j]<<" ";
        }
        int m=0;
        if(i==0&&j>1)
        {
            for(m=j;m>=1;m--)
            {
                if(mindis[i][m]-mindis[i][m-1]==1)
                    cout<<"insert:"<<b[m-1]<<endl;
                cout<<mindis[i][m-1]<<" ";
            }
        }
        else if(j==0&&i>1)
        {
            for(m=i;m>=1;m--)
            {
                if(mindis[m][j]-mindis[m-1][j]==1)
                    cout<<"delete:"<<a[m-1]<<endl;
                cout<<mindis[m-1][j]<<" ";
            }
        }
    }
    return 0;
}

/*
intention
execution
*/
