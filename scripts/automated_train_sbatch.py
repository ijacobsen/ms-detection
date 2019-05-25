import os

hypers =[ 
         [0.3, 0.3, 512, 0.4],
         [0.3, 0.3, 512, 0.5],
         [0.3, 0.3, 512, 0.6],
         [0.3, 0.3, 512, 0.7],
         [0.3, 0.3, 512, 0.8],
        ] 

for params in hypers:

    os.system('sbatch ../auto_run.s {} {} {} {}'.format(params[0], params[1], params[2], params[3]))



