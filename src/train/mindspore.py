import mindspore
import mindspore.nn as mn
import mindspore.dataset as ds

from tqdm import tqdm

from src.datasetm import MultiDatasetM
from src.model.gnnm import GNNM

def train_mindspore(args, fold):
    train = MultiDatasetM(args.dataset,
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    test = MultiDatasetM(args.dataset, train=False, 
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    column_names = ['dindex', 'pindex', 'd_vec', 'p_embedding', 'y']
    trainLoader = ds.GeneratorDataset(train, shuffle=True, column_names=column_names).batch(args.batch_size)
    testLoader = ds.GeneratorDataset(test, shuffle=False, column_names=column_names).batch(args.batch_size)

    mseLoss = mn.MSELoss()
    aeMseLoss = mn.MSELoss()
    model = GNNM(args.device, args.dropout)
    optimizer = mn.Adam(model.trainable_params(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    def forward_fn(d_index, p_index, d_vecs, p_embeddings, y):
        y_bar, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
        mse = mseLoss(y, y_bar)
        loss = mse + args.lambda_1 * aeMseLoss(decoded, feature)
        return loss, mse

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(d_index, p_index, d_vecs, p_embeddings, y):
        (loss, mse), grads = grad_fn(d_index, p_index, d_vecs, p_embeddings, y)
        optimizer(grads)
        return loss, mse

    print('training fold {}...'.format(fold))
    for epoch in range(1, args.epochs + 1):
        for d_index, p_index, d_vecs, p_embeddings, y in tqdm(trainLoader, leave=False):
            (trainLoss, trainMse) = train_step(d_index, p_index, d_vecs, p_embeddings, y)
        
        if epoch % 10 != 0 and epoch != args.epochs: continue

        count, testMse = 0, 0
        for d_index, p_index, d_vecs, p_embeddings, y in testLoader:
            y_bar, _, _, = model(d_index, p_index, d_vecs, p_embeddings, test)
            testMse += mseLoss(y, y_bar)
            count += 1
        print(trainLoss.asnumpy())
        print(trainMse)
        print(testMse / count)
        result = 'Fold: {} Epoch: {} train loss: {:.6f} train mse: {:.6f} test_mse: {:.6f}'.format(fold, epoch, trainLoss.asnumpy(), trainMse.item(), testMse.item() / count)
        print(result)

    return result
