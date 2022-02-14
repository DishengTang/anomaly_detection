import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score

from model.GCL.eval import BaseEvaluator


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_logits = classifier(x[split['test']]).detach().cpu().numpy()
                    y_pred = y_logits.argmax(axis=1)

                    acc = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred, average="macro")

                    auc = roc_auc_score(y_test, y_logits[:, 1])  #
                    ap = average_precision_score(y_test, y_logits[:, 1])

                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_val_logits = classifier(x[split['valid']]).detach().cpu().numpy()
                    y_val_pred = y_val_logits.argmax(axis=1)

                    val_micro = f1_score(y_val, y_val_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_test_acc = acc
                        best_test_recall = recall
                        best_test_auc = auc
                        best_test_ap = ap
                        best_epoch = epoch

                    pbar.set_postfix({'best test acc':acc, 'recall':recall, 'auc':auc, 'ap': ap,
                                      'F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'acc': best_test_acc,
            'recall': best_test_recall,
            'auc': best_test_auc,
            'ap': best_test_ap,
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }