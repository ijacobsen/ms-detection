import os

hypers =[ 
         [0.3, 0.3, 64],
         [0.3, 0.03, 6],
         [0.03, 0.03, 64],
         [0.03, 0.003, 64],
         [0.003, 0.003, 64],
         [0.003, 0.0003, 64],
         [0.3, 0.3, 128],
         [0.3, 0.03, 128],
         [0.03, 0.03, 128],
         [0.03, 0.003, 128],
         [0.003, 0.003, 128],
         [0.003, 0.0003, 128],
         [0.3, 0.3, 512],
         [0.3, 0.03, 512],
         [0.03, 0.03, 512],
         [0.03, 0.003, 512],
         [0.003, 0.003, 512],
         [0.003, 0.0003, 512]
        ] 

for params in hypers:

    os.system('sh bash_test {} {} {}'.format(params[0], params[1], params[2]))



