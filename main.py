from datetime import datetime

from src.train.mindspore import train_mindspore
from src.datasetm import MultiDatasetM
from src.args import Args

if __name__=='__main__':
    argparse = Args(action='train')
    args = argparse.parse_args()

    for fold in range(MultiDatasetM.fold_size(args.setting)):
        with open('./output/{}/{}_folds.log'.format(args.dataset, args.sim_type), mode='a') as file:
            # 将文本写入文件
            result = train_mindspore(args, fold)
            file.write(result + '\n')
            file.write(str(datetime.now()) + ': ' + str(argparse.parse_args()) + '\n')
