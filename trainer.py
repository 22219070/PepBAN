import os
import numpy as np
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
import torch, gc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import matthews_corrcoef, roc_curve, auc
gc.collect()
torch.cuda.empty_cache()
class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, test_dataloader, opt_da=None, discriminator=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        if self.is_da:
            self.optim_da = opt_da
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
            if torch.cuda.is_available():
                self.random_layer.cuda()
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.best_model = None
        self.best_epoch = None
        self.best_loss = float('inf')

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "Sensitivity", "Specificity", "Accuracy",
                              "Test_loss","precision", "mcc","f1","recall"]
        if self.is_da:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss"]
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact
    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            if train_loss <= self.best_loss and self.current_epoch >= self.da_init_epoch:
                if self.model.state_dict():
                    checkpoint = {'model': self.model.state_dict()}
                    self.best_loss = train_loss
                    self.best_epoch = self.current_epoch
        self.best_model=self.model
        self.best_model.load_state_dict(checkpoint['model'])
        auroc, auprc, sensitivity, specificity, accuracy, test_loss, precision, mcc, f1, recall = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, sensitivity, specificity,
                                                                            accuracy, test_loss, precision, mcc, f1, recall]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " precision " + str(precision) + "mcc " + str(mcc) +"f1 " +str(f1) +"recall "+str(recall))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["Precision"] = precision
        self.test_metrics["mcc"] = mcc
        self.test_metrics["f1"] = f1
        self.test_metrics["recall"] = recall
        self.save_result()
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w
    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch
    def train_da_epoch(self):
        torch.cuda.empty_cache()
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].float().to(
                self.device)
            v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score= self.model(v_d, v_p)
            labels = labels.long()
            n, model_loss = cross_entropy_logits(score, labels)
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score= self.model(v_d_t, v_p_t)
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score,
                                                                torch.ones(self.batch_size).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score,
                                                                torch.zeros(self.batch_size).to(self.device),
                                                                tgt_weight)
                    da_loss = abs(loss_cdan_src - loss_cdan_tgt)
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"): 
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                v_d, v_p, f, score = self.best_model(v_d, v_p)
             
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, _= roc_curve(y_label, y_pred)
            auroc = auc(fpr, tpr)
            prec, recall, thresholds = precision_recall_curve(y_label, y_pred, pos_label=1)
            auprc = auc(recall, prec)
            f1 = 2 * prec * recall / (recall + prec + 0.00001)
            thred_optim = thresholds[1:][np.argmax(f1[1:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision = precision_score(y_label, y_pred_s, average='binary')
            recall = recall_score(y_label, y_pred_s, average='binary')
            f1 = 2. * precision * recall / (precision + recall)
            mcc = matthews_corrcoef(y_label, y_pred_s)
            return auroc, auprc, sensitivity, specificity, accuracy, test_loss, precision, mcc, f1, recall
        else:
            fpr, tpr, _ = roc_curve(y_label, y_pred)
            auroc = auc(fpr, tpr)
            prec, recall, _ = precision_recall_curve(y_label, y_pred, pos_label=1)
            auprc = auc(recall, prec)
            return auroc, auprc, test_loss
        
