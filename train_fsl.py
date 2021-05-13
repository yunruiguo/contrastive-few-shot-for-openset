import numpy as np
import torch
import os
import json
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    print('Number device: ', torch.cuda.device_count())
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    # trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)



