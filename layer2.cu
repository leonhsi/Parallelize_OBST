#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__global__ void warm_up_gpu2()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

__global__ void layer2_w(int *p, int *q, int *w, int numNode)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < (numNode+1) * (numNode+2) / 2)
	{
		int i = 1;
		for(int idx = (numNode+1); idx>0; idx--)
		{
			if(tid - idx < 0)
				break;
			
			tid -= idx;
			i++;
		}
		int j = i + tid - 1;

		for(int k=i-1; k<=j; k++)
		{
			w[i * (numNode+1) + j] += q[k];
		}
		for(int k=i; k<=j; k++)
		{
			w[i * (numNode+1) + j] += p[k];
		}
	}

	return;
}

__global__ void find_min_root(int * min_s, int *s, int k, int numNode)
{
	for(int idx=0; idx<(numNode-k); idx++)
	{
		int i = idx+1;
		int temp = INT_MAX;
		for(int root = 0; root<(k+1); root++)
		{
			int pos = (k == -1) ? 1 : (k+1);
			if(min_s[idx * pos + root] <= temp)
			{
				temp = min_s[idx * pos + root];
			}
		}
		s[i * (numNode+1) + i + k] = temp;
	}
}

__global__ void layer2_s(int *q, int *w, int *s, int k, int numNode, int *min_s)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(k == -1)
	{
		if(tid < numNode-k)
		{
			s[(tid+1) * (numNode+1) + tid] = q[tid];
		}
	}
	else if(tid < (numNode-k) * (k+1))
	{
		int i = tid / (k+1) + 1;
		int r = i + tid % (k+1);

		min_s[tid] = s[i * (numNode+1) + r -1] + s[(r+1) * (numNode+1) + i + k] + w[i * (numNode+1) + i + k];
	}

	return;
}

//layer 2 front end host function
void layer2FE(int *p, int *q, int *w, int *s, int numNode)
{
	int block_size = 32;
	int grid_size = ceil( (double)(numNode+1) * (double)(numNode+2) / 2.0 / (double)block_size );

	warm_up_gpu2<<<grid_size, block_size>>>();

	//compute w
	int *w_d;
	int *p_d;
	int *q_d;
	cudaMalloc(&w_d, (numNode+2) * (numNode+1) * sizeof(int));
	cudaMalloc(&p_d, (numNode+1) * sizeof(int));
	cudaMalloc(&q_d, (numNode+1) * sizeof(int));
	cudaMemcpy(p_d, p, (numNode+1) * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(q_d, q, (numNode+1) * sizeof(int), cudaMemcpyDefault);

	layer2_w<<<grid_size, block_size>>>(p_d, q_d, w_d, numNode);

	cudaDeviceSynchronize();

	cudaMemcpy(w, w_d, (numNode+2) * (numNode+1) * sizeof(int), cudaMemcpyDefault);

	//compute s
	int *s_d;
	cudaMalloc(&s_d, (numNode+2) * (numNode+1) * sizeof(int));

	for(int k=-1; k<=(numNode-1); k++)
	{
		int min_size = (k == -1) ? (numNode-k) : (numNode-k) * (k+1);

		grid_size = ceil(double(min_size) / block_size);

		//allocate array of each root value from an entry
		int *min_s = (int *)malloc(min_size * sizeof(int));
		memset(min_s, INT_MAX, min_size);
		int *min_s_d;
		cudaMalloc(&min_s_d, min_size * sizeof(int));
		cudaMemcpy(min_s_d, min_s, min_size * sizeof(int), cudaMemcpyDefault);

		layer2_s<<<grid_size, block_size>>>(q_d, w_d, s_d, k, numNode, min_s_d);

		cudaDeviceSynchronize();

		if(k > -1)
		{
			find_min_root<<<1, 1>>>(min_s_d, s_d, k, numNode);
		}

		cudaDeviceSynchronize();

		free(min_s);
		cudaFree(min_s_d);
	}

	cudaMemcpy(s, s_d, (numNode+2) * (numNode+1) * sizeof(int), cudaMemcpyDefault);

	cudaFree(w_d);
	cudaFree(p_d);
	cudaFree(q_d);
	cudaFree(s_d);	

}
