import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from src.metrics import get_cindex, get_rm2
from tqdm import tqdm

from src.model.gnn import GNN
from src.dataset import MultiDataset
from src.args import Args

class Predict:
    def __init__(self, new=False, batch_size=1024):
        argparse = Args(action='train')
        self.args = argparse.parse_args()

        self._dataset = MultiDataset(
            self.args.dataset, train=False, device=self.args.device, new=new, sim_type=self.args.sim_type,
            d_threshold=self.args.d_threshold, p_threshold=self.args.p_threshold,
        )
        self._loader = DataLoader(self._dataset, batch_size=batch_size)

    def predict(self):
        with torch.no_grad():
            self._model = GNN().to(self.args.device)
            path = "./output/{}/{}_model.pt".format(self.args.dataset, self.args.sim_type)
            model_state_dict = torch.load(path, map_location=torch.device(self.args.device))
            self._model.load_state_dict(model_state_dict)
            self._model.eval()
            
        return self
    
    def model(self):
        return self._model

    def loader(self):
        return self._loader

    def dataset(self):
        return self._dataset

if __name__=='__main__':
    predict = Predict()
    predict.predict()

    preds = torch.tensor([])
    labels = torch.tensor([])

    for d_index, p_index, d_vecs, p_embeddings in tqdm(predict.loader(), leave=False):
        y_bar, _, _, _ = predict.model()(d_index, p_index, d_vecs, p_embeddings, predict.dataset())
        for i, pred in enumerate(y_bar.flatten().detach().numpy()):
            if pred > 15: print(pred, d_index[i], p_index[i])

    # preds = preds.detach().numpy()
    # labels = labels.detach().numpy()
    # np.savetxt('result/y_pre_DPI.txt', preDTI.detach().numpy(), fmt='%f')
    # test_mse = mean_squared_error(preds, labels)
    # ci = get_cindex(labels, preds)
    # rm2 = get_rm2(labels, preds)
