import wandb
import matplotlib.pyplot as plt

wandb.init(project="cortex22_anna", name="test5", notes="t")

for i in range(20):

    for j in range(10):

        s = i*10+j
        wandb.log({'i':i}, step=s)
        wandb.log({'s/s': s}, step=s)
        wandb.log({'ii/iisstep':i}, step=s)
        #wandb.log({'acc':acc})
        #wandb.log({'acc':acc},step = i)


wandb.finish()