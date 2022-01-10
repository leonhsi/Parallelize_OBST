#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void warm_up_gpu()
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

__global__ void layer1_w(int *p, int *q, int *w, int numNode)
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

__global__ void layer1_s(int *q, int *w, int *s, int k, int numNode)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid >= (numNode-k))
		return;

	int i = tid + 1;
	if(k == -1)
		s[i * (numNode+1) + i + k] = q[i + k];
	else
	{
		int r = i;
		s[i * (numNode+1) + i + k] = s[i * (numNode+1) + r - 1] \
									 + s[(r+1) * (numNode+1) + i + k] \
									 + w[i * (numNode+1) + i + k];
		for(r=i; r<=i+k; r++)
		{
			if(s[i*(numNode+1)+r-1] + s[(r+1)*(numNode+1)+i+k] + w[i*(numNode+1)+i+k] < s[i*(numNode+1)+i+k])
			  s[i*(numNode+1)+i+k] = s[i*(numNode+1)+r-1] + s[(r+1)*(numNode+1)+i+k] + w[i*(numNode+1)+i+k];
		}
	}
	
	return;
}

//layer1 front end host function
void layer1FE(int *p, int *q, int *w, int *s, int numNode)
{
	int block_size = 32;
	int grid_size = ceil( (double)(numNode+1) * (double)(numNode+2) / 2.0 / (double)block_size );

	warm_up_gpu<<<grid_size, block_size>>>();

	//compute w
	int *w_d;
	int *p_d;
	int *q_d;
	cudaMalloc(&w_d, (numNode+2) * (numNode+1) * sizeof(int));
	cudaMalloc(&p_d, (numNode+1) * sizeof(int));
	cudaMalloc(&q_d, (numNode+1) * sizeof(int));
	cudaMemcpy(p_d, p, (numNode+1) * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(q_d, q, (numNode+1) * sizeof(int), cudaMemcpyDefault);

	layer1_w<<<grid_size, block_size>>>(p_d, q_d, w_d, numNode);

	cudaDeviceSynchronize();

	cudaMemcpy(w, w_d, (numNode+2) * (numNode+1) * sizeof(int), cudaMemcpyDefault);

	//compute s
	int *s_d;
	cudaMalloc(&s_d, (numNode+2) * (numNode+1) * sizeof(int));

	for(int k=-1; k<=(numNode-1); k++)
	{
		grid_size = ceil( (double)(numNode-k) / block_size );
		layer1_s<<<grid_size, block_size>>>(q_d, w_d, s_d, k, numNode);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(s, s_d, (numNode+2) * (numNode+1) * sizeof(int), cudaMemcpyDefault);

	cudaFree(w_d);
	cudaFree(p_d);
	cudaFree(q_d);
	cudaFree(s_d);
}
