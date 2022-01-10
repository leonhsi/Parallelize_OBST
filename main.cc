#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "CycleTimer.h"
#include "serial.h"
#include "inc/layer1.h"
#include "inc/layer2.h"
#include "inc/dp.h"

using namespace std;

void usage(const char *progname)
{
	printf("Usage: %s [options]\n", progname);
	printf("Program Options : \n");
	printf("--node <INT>\tNumber of nodes for this binary search tree\n");
	printf("--help\t\tPrint help menu\n");
}

bool verifyResult(int *target, int *serial, int numNode)
{
	for(int i=1; i<=numNode+1; i++)
	{
		for(int j=0; j<=numNode; j++)
		{
			if(abs(target[i * (numNode+1) + j] - serial[i * (numNode+1) + j]) > 0)
			{
				printf("Wrong result : [%d][%d], Expected : %d, Actual : %d\n", i, j, serial[i*(numNode+1)+j], target[i*(numNode+1)+j]);
				return 0;
			}
		}
	}
	return 1;
}

void generate_data(int *p, int *q, int numNode)
{
	srand(0);

	for(int i=0; i<=numNode; i++)
	{
		p[i] = rand() % 10;
		q[i] = rand() % 10;
	}
}

int main(int argc, char **argv)
{
	double start_time, end_time;
	int numNode = 4;
	bool enableDP = true;

	int opt;
	static struct option long_options[] = {
		{"node", 1, 0, 'n'},
		{"help", 0, 0, 'h'},
		{0, 0, 0, 0}};
	
	while((opt = getopt_long(argc, argv, "n", long_options, NULL)) != EOF )
	{
		switch(opt)
		{
			case 'n':
			{
				int node = atoi(optarg);
				if(node > 200)
				{
					enableDP = false;
				}
				numNode = node;
				break;
			}
			case 'h':
			default:
				usage(argv[0]);
				return 1;
		}
	}

	cout << "Number of nodes : " << numNode << endl;

	//allocate data
	int *w, *s;
	w = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));
	s = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));

	int *p, *q;
	p = (int *)malloc((numNode+1) * sizeof(int));
	q = (int *)malloc((numNode+1) * sizeof(int));
	generate_data(p, q, numNode);

	//serial version
	start_time = currentSeconds();

	serial_w(p, q, w, numNode);
	serial_s(q, w, s, numNode);
	
	end_time = currentSeconds();
	double serial_time = end_time - start_time;
	cout << "\n[Serial] : \t\t[" << serial_time * 1000 << "] ms\n\n";
	cout << "---------------------------------------------\n";

	//layer 1 version
	int *l1_w, *l1_s;
	l1_w = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));
	l1_s = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));

	start_time = currentSeconds();

	layer1FE(p, q, l1_w, l1_s, numNode);
	
	end_time = currentSeconds();
	double l1_time = end_time - start_time;
	cout << "\n[Layer1] : \t\t[" << l1_time * 1000 << "] ms\n\n";

	if(!verifyResult(l1_s, s, numNode))
	{
		cout << "Error : Output from layer1 dose not match serial output\n";
		return 1;
	}
	cout << "Speedup  : \t\t[" << serial_time / l1_time << "]\n\n";
	cout << "---------------------------------------------\n";
	
	//layer 2 version
	int *l2_w, *l2_s;
	l2_w = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));
	l2_s = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));

	start_time = currentSeconds();

	layer2FE(p, q, l2_w, l2_s, numNode);

	end_time = currentSeconds();
	double l2_time = end_time - start_time;
	cout << "\n[Layer2] : \t\t[" << l2_time * 1000 << "] ms\n\n";

	if(!verifyResult(l2_s, s, numNode))
	{
		cout << "Error : Output from layer2 dose not match serial output\n";
		return 1;
	}
	cout << "Speedup  : \t\t[" << serial_time / l2_time << "]\n\n";
	cout << "---------------------------------------------\n";

	//Dynamic Parallelism version
	int *dp_w, *dp_s;
	dp_w = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));
	dp_s = (int *)calloc((numNode+2) * (numNode+1), sizeof(int));

	start_time = currentSeconds();

	if(enableDP)
	{
		dpFE(p, q, dp_w, dp_s, numNode);
		
		end_time = currentSeconds();
		double dp_time = end_time - start_time;
		cout << "\n[DP] : \t\t\t[" << dp_time * 1000 << "] ms\n\n";

		if(!verifyResult(dp_s, s, numNode))
		{
			cout << "Error : Output from layer1 dose not match serial output\n";
			return 1;
		}
		cout << "Speedup  : \t\t[" << serial_time / dp_time << "]\n\n";
	}
	else
	{
		cout << "\n[DP] : \t\t\t[can't use]\n\n" ;
	}
	cout << "---------------------------------------------\n";
}
