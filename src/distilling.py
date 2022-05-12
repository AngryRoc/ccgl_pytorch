import copy
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models import CasEncoder, MSLELoss

import numpy as np
from absl import app, flags

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_boolean('distill_with_unlabel', True, 'Distill with label.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('epochs', 1000, 'Distillation epochs.')
flags.DEFINE_float('l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string('name', 'RocYan', 'Name of the teacher model.')
flags.DEFINE_string('num', '0', 'Suffix of the student model name.')
flags.DEFINE_integer('patience', 20, 'Patience for early stopping.')
flags.DEFINE_string('projection_head', '4-1', 'MLP-based projection head.')
flags.DEFINE_boolean('self_distill', True, 'Self distillation.')

# paths
flags.DEFINE_string('input', './datasets/weibo/', 'Distillation data path.')
flags.DEFINE_string('result_path', './results/prediction/', 'Path of model predictions.')
flags.DEFINE_string('teacher_path', './results/fine_tuning_weight/', 'Path of saved teacher weigths.')
flags.DEFINE_string('student_path', './results/student_weight/', 'Path of saved student weigths.')
flags.DEFINE_string('cuda_device', '0', 'GPU in use')


def main(argv):
    start_time = time.time()
    device = torch.device("cuda:" + FLAGS.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper-params
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    max_seq = FLAGS.max_seq
    emb_dim = FLAGS.emb_dim
    patience = FLAGS.patience
    l2 = FLAGS.l2
    lr = FLAGS.lr
    projection_head = FLAGS.projection_head
    # hyper-params

    # load data
    with open(FLAGS.input + 'train.pkl', 'rb') as f:
        train, _ = pickle.load(f)
    with open(FLAGS.input + 'val.pkl', 'rb') as f:
        val, val_y = pickle.load(f)
    with open(FLAGS.input + 'test.pkl', 'rb') as f:
        test, test_y = pickle.load(f)

    if FLAGS.distill_with_unlabel:
        with open(FLAGS.input + 'data_unlabel_aug_1.pkl', 'rb') as f:
            unlabel_aug_1 = pickle.load(f)
        with open(FLAGS.input + 'data_unlabel_aug_2.pkl', 'rb') as f:
            unlabel_aug_2 = pickle.load(f)
        unlabel_aug_1.extend(unlabel_aug_2)
        train.extend(unlabel_aug_1)


    # build model

    teacher_casEncoder = CasEncoder(emb_dim, gru_units)
    student_casEncoder = CasEncoder(emb_dim, gru_units)
    student_mlp = nn.Sequential()

    encoder = nn.Sequential()
    teacher = nn.Sequential()
    student = nn.Sequential()

    if projection_head[2] == '0':
        encoder.add_module("casEncoder", teacher_casEncoder)
    elif projection_head[2] == '1':
        encoder.add_module("casEncoder", teacher_casEncoder)
        encoder.add_module("mlp1", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu1", nn.ReLU())
    elif projection_head[2] == '2':
        encoder.add_module("casEncoder", teacher_casEncoder)
        encoder.add_module("mlp1", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu2", nn.ReLU())
    elif projection_head[2] == '3':
        encoder.add_module("casEncoder", teacher_casEncoder)
        encoder.add_module("mlp1", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("mlp3", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu3", nn.ReLU())
    elif projection_head[2] == '4':
        encoder.add_module("casEncoder", teacher_casEncoder)
        encoder.add_module("mlp1", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("mlp2", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("mlp3", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu3", nn.ReLU())
        encoder.add_module("mlp4", nn.Linear(gru_units * 2, gru_units * 2))
        encoder.add_module("relu4", nn.ReLU())

    else:
        print('Wrong projection head argument, should be [0-4]-[0-4].')

    teacher.add_module("encoder", encoder)
    teacher.add_module("prediction_mlp", nn.Linear(gru_units * 2, 1))
    teacher.load_state_dict(torch.load(FLAGS.teacher_path + FLAGS.name + '.pth')['model'])
    teacher = teacher.to(device)
    teacher.eval()

    # build a self-distilled student or a dense student
    if FLAGS.self_distill:
        if projection_head[2] == '0':
            student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))
            student.add_module("student_casEncoder", student_casEncoder)
            student.add_module("student_mlp", student_mlp)
        elif projection_head[2] == '1':
            student_mlp.add_module("student_mlp1", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu1", nn.ReLU())
            student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))

            student.add_module("student_casEncoder", student_casEncoder)
            student.add_module("student_mlp", student_mlp)
        elif projection_head[2] == '2':
            student_mlp.add_module("student_mlp1", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu1", nn.ReLU())
            student_mlp.add_module("student_mlp2", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu2", nn.ReLU())
            student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))

            student.add_module("student_casEncoder", student_casEncoder)
            student.add_module("student_mlp", student_mlp)
        elif projection_head[2] == '3':
            student_mlp.add_module("student_mlp1", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu1", nn.ReLU())
            student_mlp.add_module("student_mlp2", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu2", nn.ReLU())
            student_mlp.add_module("student_mlp3", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu3", nn.ReLU())
            student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))

            student.add_module("student_casEncoder", student_casEncoder)
            student.add_module("student_mlp", student_mlp)
        elif projection_head[2] == '4':

            student_mlp.add_module("student_mlp1", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu1", nn.ReLU())
            student_mlp.add_module("student_mlp2", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu2", nn.ReLU())
            student_mlp.add_module("student_mlp3", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu3", nn.ReLU())
            student_mlp.add_module("student_mlp4", nn.Linear(gru_units * 2, gru_units * 2))
            student_mlp.add_module("student_relu4", nn.ReLU())
            student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))

            student.add_module("student_casEncoder", student_casEncoder)
            student.add_module("student_mlp", student_mlp)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')

    else:
        student_mlp.add_module("student_mlp", nn.Linear(gru_units * 2, gru_units * 2))
        student_mlp.add_module("student_relu", nn.ReLU())
        student_mlp.add_module("student_prediction_mlp", nn.Linear(gru_units * 2, 1))
        student.add_module("student_casEncoder",student_casEncoder)
        student.add_module("student_mlp", student_mlp)

    student = student.to(device)

    # summary
    summary(teacher, input_size=(max_seq, emb_dim), batch_size=batch_size)
    summary(student, input_size=(max_seq, emb_dim), batch_size=batch_size)

    # optimizer
    casEncoder_optimizer = torch.optim.Adam(student_casEncoder.parameters(), lr=lr, weight_decay=l2)
    mlp_optimizer = torch.optim.Adam(student_mlp.parameters(), lr=lr)
    # criterion

    criterion = MSLELoss().to(device)

    # print data information
    dataset_info = '#   unlabeled samples {}\n' + \
                   '#  validation samples {}\n' + \
                   '#        test samples {}'
    print(dataset_info.format(len(train), len(val), len(test)))

    # distilling
    best_val_loss = 1000
    save_predictions = list()
    for epoch in range(epochs):
        time_start = time.time()
        random.shuffle(train)
        losses = []
        student.train()
        for i in range(len(train) // batch_size + 1):
            casEncoder_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            batch_train = copy.deepcopy(train[batch_size * i:batch_size * i + batch_size])
            for batch_cascade in batch_train:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))

            batch_train = torch.from_numpy(np.array(batch_train)).float().to(device)
            predictions = student(batch_train).view(-1)
            labels = teacher(batch_train).view(-1).detach()
            loss = criterion(predictions, labels)
            loss.backward()
            casEncoder_optimizer.step()
            mlp_optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        losses = []
        student.eval()
        for i in range(len(val) // batch_size + 1):
            batch_val = copy.deepcopy(val[batch_size * i:batch_size * i + batch_size])
            batch_val_labels = val_y[batch_size * i:batch_size * i + batch_size]
            for batch_cascade in batch_val:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            batch_val = torch.from_numpy(np.array(batch_val)).float().to(device)
            batch_val_labels = torch.from_numpy(np.array(batch_val_labels)).float().view(-1).to(device)
            predictions = student(batch_val).view(-1)
            loss = criterion(batch_val_labels, predictions)
            losses.append(loss.item())
        val_loss = np.mean(losses)

        losses = []
        student.eval()
        pred = list()
        for i in range(len(test) // batch_size + 1):
            batch_test = copy.deepcopy(test[batch_size * i:batch_size * i + batch_size])
            batch_test_labels = test_y[batch_size * i:batch_size * i + batch_size]

            for batch_cascade in batch_test:
                while len(batch_cascade) < max_seq:
                    batch_cascade.append(np.zeros(emb_dim))
            batch_test = torch.from_numpy(np.array(batch_test)).float().to(device)
            batch_test_labels = torch.from_numpy(np.array(batch_test_labels)).float().view(-1).to(device)
            predictions = student(batch_test).view(-1)
            loss = criterion(batch_test_labels, predictions)
            pred.extend(predictions)
            losses.append(loss.item())
        test_loss = np.mean(losses)

        pred = [float(pre) for pre in pred]
        report_loss = np.mean(np.square(np.log2(np.array([pre if pre >= 1 else 1 for pre in pred])) -
                                        np.log2(np.array([tru if tru >= 1 else 1 for tru in list(test_y)]))))

        template = 'Epoch {:2}, Time: {:.3f}s, Train Loss: {:.3f}, Val Loss: {:.3f}, ' \
                   'Test Loss: {:.3f}, LOG2 MSLE: {:.3f}'
        print(template.format(epoch + 1, time.time() - time_start,
                              train_loss, val_loss, test_loss, report_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_predictions = pred
            patience = FLAGS.patience

            # save model
            torch.save({'model': student.state_dict()}, FLAGS.student_path + FLAGS.name + '.pth')
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
