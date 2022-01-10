#include <iostream>
#include <climits>

void serial_w(int *p, int *q, int *w, int numNode)
{
	for(int i=1; i<=(numNode+1); i++)
	{
		for(int j=0; j<=numNode; j++)
		{
			if(i-j < 2)
			{
				for(int k=(i-1); k<=j; k++)
				{
					w[i * (numNode + 1) + j] += q[k];
				}
				for(int k=i; k<=j; k++)
				{
					w[i * (numNode + 1) + j] += p[k];
				}
			}
		}
	}
}

void serial_s(int *q, int *w, int *s, int numNode)
{
	for(int k=-1; k<=(numNode-1); k++)
	{
		for(int i=1; i<=(numNode+1); i++)
		{
			if(i+k > numNode)
			{
				break;
			}

			int min = INT_MAX;
			if(k == -1)
			{
				s[i * (numNode+1) + i + k] = q[i + k];
			}
			else
			{
				for(int root=i; root<=i+k; root++)
				{
					s[i * (numNode+1) + i + k] = s[i * (numNode+1) + root -1] + \
						s[(root+1) * (numNode+1) + i + k] + \
						w[i * (numNode+1) + i +k];
					if(s[i * (numNode+1) + i + k] < min)
					{
						min = s[i * (numNode+1) + i + k];
					}
				}
				s[i * (numNode+1) + i + k] = min;
			}
		}
	}
}
