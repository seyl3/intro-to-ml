import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, get_n_classes,onehot_to_label, label_to_onehot
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # simple train/validation split (80/20)
        n = train_features.shape[0]
        idx = np.random.permutation(n)
        split = int(0.8 * n)

        train_idx, val_idx = idx[:split], idx[split:]

        val_features = train_features[val_idx]
        val_labels_reg = train_labels_reg[val_idx]
        val_labels_classif = train_labels_classif[val_idx]
        train_features = train_features[train_idx]
        train_labels_reg = train_labels_reg[train_idx]
        train_labels_classif = train_labels_classif[train_idx]

    # Normalize features (fit on train, apply to others)
    train_features, mean, std = normalize_fn(train_features)
    test_features = (test_features - mean) / std

    if not args.test:
        val_features = (val_features - mean) / std

    # Add bias term
    train_features = append_bias_term(train_features)
    test_features = append_bias_term(test_features)

    if not args.test:
        val_features = append_bias_term(val_features)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        ### WRITE YOUR CODE HERE
        n_features = train_features.shape[1]
        n_classes = get_n_classes(train_labels_classif)

        hidden_layers = [int(x) for x in args.mlp_dim.split(',')]
        
        dimensions = (n_features, *hidden_layers, n_classes)

        act_fn = ReLU if args.activation == "relu" else Sigmoid
        
        activations = [act_fn] * (len(dimensions) - 1)

        print(f"Initialisation MLP avec dimensions: {dimensions} et activation: {args.activation}")
        method_obj = MLP(dimensions=dimensions, activations=activations)
        
        pass
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        ### WRITE YOUR CODE HERE
        if args.method == "mlp":
            Y_train_oh = label_to_onehot(train_labels_classif)

            print("MLP training in progress...")
            method_obj.fit(train_features, Y_train_oh, loss=MSE, 
                           epochs=args.max_iters, batch_size=16, learning_rate=args.lr)

            #choisit val ou test selon le flag --test
            data_to_pred = test_features if args.test else val_features
            preds_oh = method_obj.predict(data_to_pred)
            
            preds = onehot_to_label(preds_oh)

        elif args.method == "kmeans":
            print("KMeans training in progress...")
            method_obj.fit(train_features, train_labels_classif)

            data_to_pred = test_features if args.test else val_features
            preds = method_obj.predict(data_to_pred)

        gt = test_labels_classif if args.test else val_labels_classif
        split_name = "Test" if args.test else "Validation"
        print(f"{split_name} accuracy: {accuracy_fn(preds, gt):.2f}%")
        print(f"{split_name} F1: {macrof1_fn(preds, gt):.4f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument(
        "--mlp_dim",
        type=str,
        default="64,32",
        help="Dimensions des couches cachées, séparées par des virgules (par ex: 64,32)"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        choices=["relu", "sigmoid"],
        help="Fonction d'activation à utiliser pour les couches cachées"
    )
    
    args = parser.parse_args()
    main(args)
