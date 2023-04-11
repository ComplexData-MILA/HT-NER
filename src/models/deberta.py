import torch
from torch import nn
import numpy as np
from sklearn import metrics
from transformers import DebertaModel, DebertaPreTrainedModel

from .layers.crf import CRF
from losses.lossfn import crossentropy_loss
from ark_nlp.factory.loss_function import GlobalPointerCrossEntropy
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer


class DebertaBaseModel(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    base_model = "deberta"

    def __init__(self, config):
        super(DebertaBaseModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()


class DebertaTokenClassification(DebertaBaseModel):
    def __init__(self, config):
        super(DebertaTokenClassification, self).__init__(config)
        self.loss = crossentropy_loss
        if config.lstm:
            self.lstm = nn.LSTM(
                self.config.hidden_size,
                self.config.hidden_size // 2,
                2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True,
            )
            self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.use_lstm = config.lstm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )[0]
        if self.use_lstm:
            outputs = self.lstm(outputs)[0]
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return self.model_output(logits, labels, attention_mask)

    def model_output(self, logits, targets, mask):
        loss = 0
        if targets is not None:
            loss = self.loss(logits, targets, attention_mask=mask)
            f1 = self.monitor_metrics(logits, targets, attention_mask=mask)
            return loss, logits, f1
        prob = torch.softmax(logits, dim=-1)
        return prob, loss

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}


class DebertaGlobalPointer(DebertaTokenClassification):
    def __init__(self, config, head_size=128, efficient=False):
        super(DebertaGlobalPointer, self).__init__(config)
        if efficient:
            self.global_pointer = EfficientGlobalPointer(
                config.num_labels,
                head_size=head_size,
                hidden_size=config.hidden_size * (2 if self.use_lstm else 1),
            )
        else:
            self.global_pointer = GlobalPointer(
                config.num_labels,
                head_size=head_size,
                hidden_size=config.hidden_size * (2 if self.use_lstm else 1),
            )
        del self.classifier
        del self.dropout
        self.init_weights()

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        Return: tensor with shape [bc, batch_max_length]
        """
        outputs = getattr(self, self.base_model)(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )[0]
        if self.use_lstm:
            outputs = self.lstm(outputs)[0]
        logits = self.global_pointer(outputs, mask=attention_mask)
        return self.model_output(logits, labels)

    def model_output(self, logits, targets, threshold=0.0):
        loss = 0
        if targets is not None and targets._nnz() != 0:
            # print(self.num_labels,targets.shape,logits.shape)
            loss_fct = GlobalPointerCrossEntropy()
            loss = loss_fct(logits, targets)
            f1 = self.monitor_metrics(logits, targets)
            # print(logits, loss, f1)
            return loss, logits, f1

        logits = logits.cpu().numpy()
        logits[:, :, [0, -1]] -= np.inf
        logits[:, :, :, [0, -1]] -= np.inf

        batch_results = []  # [[1,2,3,3,4,3,23],[2,3,3,3,3,]]
        for i in range(logits.shape[0]):
            cur_logtis = logits[i]
            confidences = []
            for category, start, end in zip(*np.where(cur_logtis > threshold)):
                confidences.append(
                    (cur_logtis[category, start, end], category, start, end)
                )
            confidences = sorted(confidences, key=lambda x: x[0])
            output_ids = [1] * cur_logtis.shape[-1]
            for conf, cat, s, e in confidences:
                output_ids[s:e] = [cat] * (e - s)
            batch_results.append(output_ids)

        return batch_results, None

    def monitor_metrics(self, outputs, targets, attention_mask=None):
        def global_pointer_f1_score(y_true, y_pred):
            y_pred = torch.gt(y_pred, 0)
            return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()

        numerate, denominator = global_pointer_f1_score(
            targets.to_dense().cpu(), outputs.cpu()
        )
        return {"f1": 2 * numerate / denominator}


class DebertaCRF(DebertaTokenClassification):
    def __init__(self, config):
        super(DebertaCRF, self).__init__(config)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.init_weights()

    def model_output(self, logits, targets, mask):
        loss = 0
        if targets is not None:
            # cross_loss = self.loss(logits, targets, attention_mask=mask)
            loss = -self.crf(
                emissions=torch.nn.functional.log_softmax(logits, 2),
                tags=targets,
                mask=mask,
            )
            f1 = self.monitor_metrics(logits, targets, attention_mask=mask)
            return loss, logits, f1
        # logits = self.crf.decode(logits, mask=mask, nbest=3).squeeze()
        # logits = torch.nn.functional.one_hot(logits, self.num_labels)
        nbest_weight = [0.6, 0.3, 0.1]
        logits = self.crf.decode(
            logits, mask=mask, nbest=len(nbest_weight)
        )  # (nbest, bc, seqlen)
        final = torch.zeros(logits.shape[1:] + (self.num_labels,)).to(
            "cuda"
        )  # a = a.to(b.device)
        for i, thr in enumerate(nbest_weight):
            final += torch.nn.functional.one_hot(logits[i], self.num_labels) * thr
        return final, loss
