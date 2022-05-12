import copy
import pickle
import time

import torch
import torch.nn as nn
from torchsummary import summary
from models import CasEncoder, MSLELoss

import numpy as np
from absl import app, flags
from utils.tools import divide_dataset, shuffle_two

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('epochs', 1000, 'Pre-training epochs.')
flags.DEFINE_boolean('freeze', False, 'Linear evaluation on frozen features.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_float('l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_integer('label_fraction', 100, 'Label fraction, only for 1%, 10%, and 100%.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string('name', 'RocYan', 'Name of the pre-training model.')
flags.DEFINE_string('num', '0', 'Suffix of the saved teacher model.')
flags.DEFINE_integer('patience', 20, 'Patience for early stopping.')
flags.DEFINE_string('projection_head', '4-1', 'MLP-based projection head.')

# paths
flags.DEFINE_string('input', './datasets/weibo/', 'Pre-training data path.')
flags.DEFINE_string('result_path', './results/prediction/', 'Path of model predictions.')
flags.DEFINE_string('weight_path', './results/pre_training_weight/', 'Path of saved encoder weights.')
flags.DEFINE_string('teacher_path', './results/fine_tuning_weight/', 'Path of teacher network weights.')
flags.DEFINE_string('cuda_device', '0', 'GPU in use')


def main(argv):
    start_time = time.time()
    device = torch.device("cuda:" + FLAGS.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper-params
    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    epochs = FLAGS.epochs
    freeze = FLAGS.freeze
    l2 = FLAGS.l2
    label_fraction = FLAGS.label_fraction
    lr = FLAGS.lr
    max_seq = FLAGS.max_seq
    patience = FLAGS.patience
    projection_head = FLAGS.projection_head
    # hyper-params

    # build model
    casEncoder = CasEncoder(emb_dim, gru_units)
    prediction_mlp = nn.Linear(gru_units * 2, 1)
    mlp = nn.Sequential()

    encoder = nn.Sequential()
    prediction = nn.Sequential()
    if projection_head[2] == '0':

        mlp.add_module("prediction_mlp",prediction_mlp)
        encoder.add_module("casEncoder", casEncoder)

    elif projection_head[2] == '1':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("prediction_mlp", prediction_mlp)
        encoder.add_module("casEncoder", casEncoder)
        encoder.add_module("mlp1", mlp1)
        encoder.add_module("relu1", nn.ReLU())
    elif projection_head[2] == '2':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        mlp.add_module("prediction_mlp", prediction_mlp)

        encoder.add_module("casEncoder", casEncoder)
        encoder.add_module("mlp1", mlp1)
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", mlp2)
        encoder.add_module("relu2", nn.ReLU())
    elif projection_head[2] == '3':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp3 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        mlp.add_module("mlp3", mlp3)
        mlp.add_module("relu3", nn.ReLU())
        mlp.add_module("prediction_mlp", prediction_mlp)

        encoder.add_module("casEncoder", casEncoder)
        encoder.add_module("mlp1", mlp1)
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", mlp2)
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("mlp3", mlp3)
        encoder.add_module("relu3", nn.ReLU())
    elif projection_head[2] == '4':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp3 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp4 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        mlp.add_module("mlp3", mlp3)
        mlp.add_module("relu3", nn.ReLU())
        mlp.add_module("mlp4", mlp4)
        mlp.add_module("relu4", nn.ReLU())
        mlp.add_module("prediction_mlp", prediction_mlp)

        encoder.add_module("casEncoder", casEncoder)

        encoder.add_module("mlp1", mlp1)
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", mlp2)
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("mlp3", mlp3)
        encoder.add_module("relu3", nn.ReLU())
        encoder.add_module("mlp4", mlp4)
        encoder.add_module("relu4", nn.ReLU())


    else:
        print('Wrong projection head argument, should be [0-4]-[0-4].')

    if freeze:
        encoder.eval()
    encoder.load_state_dict(torch.load(FLAGS.weight_path + FLAGS.name + '.pth')['model'])

    prediction.add_module("encoder", encoder)
    prediction.add_module("prediction_mlp", prediction_mlp)
    prediction = prediction.to(device)

    summary(prediction, input_size=(max_seq, emb_dim), batch_size=batch_size)

    # optimize
    casEncoder_optimizer = torch.optim.Adam(casEncoder.parameters(), lr=lr, weight_decay=l2)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    # criterion

    criterion = MSLELoss().to(device)

    # load data
    with open(FLAGS.input + 'train.pkl', 'rb') as f:
        train, train_y = pickle.load(f)
    with open(FLAGS.input + 'val.pkl', 'rb') as f:
        val, val_y = pickle.load(f)
    with open(FLAGS.input + 'test.pkl', 'rb') as f:
        test, test_y = pickle.load(f)

    # divide dataset
    train, train_y = divide_dataset(train, label_fraction), divide_dataset(train_y, label_fraction)

    # print data information
    dataset_info = '# fine-tuning samples {}\n' + \
                   '#  validation samples {}\n' + \
                   '#        test samples {}'
    print(dataset_info.format(len(train), len(val), len(test)))

    # linear evaluation or fine-tuning
    best_val_loss = 1000
    save_predictions = list()
    for epoch in range(epochs):
        time_start = time.time()
        train, train_y = shuffle_two(train, train_y)
        losses = []
        prediction.train()
        for i in range(len(train) // batch_size + 1):
            casEncoder_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            batch_train = copy.deepcopy(train[batch_size * i:batch_size * i + batch_size])
            batch_train_labels = train_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_train:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))

            batch_train = torch.from_numpy(np.array(batch_train)).float().to(device)
            batch_train_labels = torch.from_numpy(np.array(batch_train_labels)).float().to(device)

            predictions = prediction(batch_train).view(-1)

            loss = criterion(batch_train_labels, predictions)
            loss.backward()
            casEncoder_optimizer.step()
            mlp_optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        losses = []
        prediction.eval()
        for i in range(len(val) // batch_size + 1):
            batch_val = copy.deepcopy(val[batch_size * i:batch_size * i + batch_size])
            batch_val_labels = val_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_val:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            batch_val = torch.from_numpy(np.array(batch_val)).float().to(device)
            batch_val_labels = torch.from_numpy(np.array(batch_val_labels)).float().to(device)
            predictions = prediction(batch_val).view(-1)

            loss = criterion(batch_val_labels, predictions)
            losses.append(loss.item())
        val_loss = np.mean(losses)

        losses = []
        prediction.eval()
        pred = list()
        for i in range(len(test) // batch_size + 1):
            batch_test = copy.deepcopy(test[batch_size * i:batch_size * i + batch_size])
            batch_test_labels = test_y[batch_size * i:batch_size * i + batch_size]

            for batch_cascade in batch_test:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))

            batch_test = torch.from_numpy(np.array(batch_test)).float().to(device)
            batch_test_labels = torch.from_numpy(np.array(batch_test_labels)).float().to(device)
            predictions = prediction(batch_test).view(-1)
            loss = criterion(batch_test_labels, predictions)
            pred.extend(predictions)
            losses.append(loss.item())
        test_loss = np.mean(losses)
        pred = [float(pre) for pre in pred]

        report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in pred])) -
                                        np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

        template = '{}: Fine-tuning Epoch {:3}, Time: {:.3f}s, Train Loss: {:.3f}, Val Loss: {:.3f}, ' \
                   'Test Loss: {:.3f}, LOG2 MSLE: {:.3f}'
        print(template.format(FLAGS.name, epoch + 1, time.time() - time_start,
                              train_loss, val_loss, test_loss, report_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_predictions = pred
            patience = FLAGS.patience

            # save model
            torch.save({'model': prediction.state_dict()}, FLAGS.teacher_path + FLAGS.name + '.pth')
            print('Model saved!')

        if patience == 0:
            report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in save_predictions])) -
                                            np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

            print('Predictions saved! Best Test MSLE: {}'.format(report_loss))
            break
        else:
            patience -= 1

    print('Finished! Time used: {:.3f}min'.format((time.time() - start_time) / 60))


if __name__ == '__main__':
    app.run(main)
