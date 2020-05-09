# encoding: utf-8
# base-line reconstruction performance

import numpy as np
from train_inversion import get_dataset
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="evaluate the base-line reconstruction loss")
    parser.add_argument("target", help="target dataset", type=str)
    parser.add_argument("source", help="auxiliary dataset", type=str)
    parser.add_argument("--repeat", help="times of repeatition", type=int, default=100)

    args = parser.parse_args()
    print("==============================")
    print(args)
    print("==============================")

    # boundary distribution
    target = get_dataset(args.target, train=True, inv=False).data.numpy()
    source = np.zeros(target.shape) #get_dataset(args.source, train=True, inv=True).data.numpy()

    # random sample from source dataset, for each feature
    np.random.seed(233)

    mse_loss = []
    t1 = time.time()
    print("please wait ...")

    for _ in range(args.repeat):
        pred = []
        indices = np.arange(len(source))
        for i in range(target.shape[1]):
            perm = np.random.choice(indices, size=len(target), replace=True)
            pred.append(source[:, i:i+1][perm])

        pred = np.hstack(pred)

        # pred = np.zeros()

        # import pandas as pd
        # pd.set_option('display.max_columns', None)
        # df = pd.DataFrame(pred.astype(np.int))
        # print(df.iloc[:5, :10])

        assert(target.shape == pred.shape)
        mse_loss.append( np.mean( np.power( target - pred, 2 ) ) )

    print(f"done! took {int(time.time() - t1)} seconds")
    print(f"MSE {np.mean(mse_loss)}")



if __name__ == "__main__":
    main()