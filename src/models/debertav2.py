import torch
from torch import nn
import numpy as np
from sklearn import metrics
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from .layers.crf import CRF
from ark_nlp.factory.loss_function import GlobalPointerCrossEntropy
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer


from torch.nn.modules.loss import CrossEntropyLoss
def crossentropy_loss(logits, labels, attention_mask, num_labels):
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels),
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


class DebertaV2BaseModel(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    base_model = "deberta"

    def __init__(self, config):
        super(DebertaV2BaseModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()


class DebertaV2TokenClassification(DebertaV2BaseModel):
    def __init__(self, config):
        super(DebertaV2TokenClassification, self).__init__(config)
        self.loss = crossentropy_loss
        if config.BiLSTM:
            self.lstm = nn.LSTM(
                self.config.hidden_size,
                self.config.hidden_size,
                2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True,
            )
            self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        if self.config.BiLSTM:
            outputs = self.lstm(outputs)[0]
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return self.model_output(logits, labels, attention_mask)
    
    def model_output(self, logits, targets, mask):
        # loss = 0
        # if targets is not None:
        #     loss = self.loss(logits, targets, attention_mask=mask, num_labels=self.num_labels)
        #     # f1 = self.monitor_metrics(logits, targets, attention_mask=mask)
        #     return loss, logits
        # return logits
        
        loss = None
        if targets is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), targets.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)
    

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}


class DebertaV2GlobalPointer(DebertaV2TokenClassification):
    def __init__(self, config, head_size=64, efficient=False):
        super(DebertaV2GlobalPointer, self).__init__(config)
        if efficient:
            self.global_pointer = EfficientGlobalPointer(
                config.num_labels + 1, head_size=head_size, 
                hidden_size=config.hidden_size*(2 if self.config.BiLSTM else 1)
            )
        else:
            self.global_pointer = GlobalPointer(
                config.num_labels + 1, head_size=head_size, 
                hidden_size=config.hidden_size*(2 if self.config.BiLSTM else 1)
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
        if self.config.BiLSTM:
            outputs = self.lstm(outputs)[0]
        logits = self.global_pointer(outputs, mask=attention_mask)
        return self.model_output(logits, labels)

    def model_output(self, logits, targets, threshold=0.0):
        loss = 0.0
        if targets is not None and targets._nnz() != 0 and logits.requires_grad:
            loss_fct = GlobalPointerCrossEntropy()
            loss = loss_fct(logits, targets)
            # f1 = self.monitor_metrics(logits, targets)
            # print(logits, loss, f1)
            # return logits
            return TokenClassifierOutput(loss=loss, logits=logits)
        
        if targets is not None:
            return torch.tensor(0.0), self.decode(logits), \
                np.argmax(self.decode(targets.to_dense().type(torch.DoubleTensor)), axis=2)
        else:
            return torch.tensor(0.0), self.decode(logits), None
        
    def decode(self, logits):
        logits = logits.cpu().numpy()
        logits[:, :, [0, -1]] -= np.inf
        logits[:, :, :, [0, -1]] -= np.inf

        batch_results = []  # [[1,2,3,3,4,3,2],[2,3,3,3,3,]]
        for i in range(logits.shape[0]):
            cur_logtis = logits[i]
            confidences = []
            for category, start, end in zip(*np.where(cur_logtis > 0.)):
                confidences.append(
                    (cur_logtis[category, start, end], category, start, end)
                )
            confidences = sorted(confidences, key=lambda x: x[0])
            output_ids = [1] * cur_logtis.shape[-1]
            for conf, cat, s, e in confidences:
                output_ids[s:e+1] = [cat] * (e - s+1)
            output_ids = torch.nn.functional.one_hot(torch.tensor(output_ids), self.num_labels + 1) # modified
            batch_results.append(output_ids)

        return torch.stack(batch_results, dim=0)[:, :, 1:]
    
    @staticmethod
    def monitor_metrics(outputs, targets, attention_mask=None):
        def global_pointer_f1_score(y_true, y_pred):
            y_pred = torch.gt(y_pred, 0)
            return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()

        numerate, denominator = global_pointer_f1_score(
            targets.to_dense().cpu(), outputs.cpu()
        )
        return {"f1": 2 * numerate / denominator}

from transformers import DataCollatorForTokenClassification

class DebertaV2GlobalPointerDataCollator(DataCollatorForTokenClassification):
    # label_pad_token_id: int = 0 
    # return_tensors: str = "pt"
    # num_labels: int = 5
    # max_length = 300
    # padding = 'max_length'
    
    def torch_call(self, features):
        # print(features)
        def tmpf(x):
            x[0]=0
            x[-1]=0
            return x
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [tmpf(feature[label_name]) for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=True, #'max_length', #self.padding,
            max_length=self.max_length, # self.max_length,
            pad_to_multiple_of=True, #self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items() if k != label_name}
        batch[label_name] = torch.stack([self.padding_sparse(label, sequence_length) for label in labels])
        batch['attention_mask'][:,0] = batch['attention_mask'][:,1]
        return batch
    
    def padding_sparse(self, label, batch_max=300):
        if label != []:
            # print(label) => [label, s, e]
            groups = [[-2,-1,-1],]
            for i, l in enumerate(label):
                if l == groups[-1][0]:
                    groups[-1][2] = i
                elif l != groups[-1][0]:
                    groups.append([l, i, i])
            groups.pop(0)
            
            label = np.array(groups).T
            label[0] += 1
            label = torch.sparse_coo_tensor(
                label, [1] * len(label[0]), (self.num_labels + 1, batch_max, batch_max)
            )
        else:
            label = torch.sparse_coo_tensor(size=(self.num_labels + 1, batch_max, batch_max))
        return label


class DebertaV2CRF(DebertaV2TokenClassification):
    def __init__(self, config):
        config.num_labels += 2
        super(DebertaV2CRF, self).__init__(config)
        from allennlp.modules.conditional_random_field.conditional_random_field import ConditionalRandomField
        self.crf = ConditionalRandomField(num_tags=self.num_labels, include_start_end_transitions=True)#, batch_first=True)
        self.init_weights()
        # <start>: self.num_labels, <end>: self.num_labels+1
    def model_output(self, logits, targets, mask):
        loss = 0.
        if targets is not None and logits.requires_grad:
            # cross_loss = self.loss(logits, targets, attention_mask=mask)
            loss = -self.crf(
                inputs=logits,
                # emissions=torch.nn.functional.log_softmax(logits, 2),
                tags=targets,
                mask=mask,
            )
            return TokenClassifierOutput(loss=loss, logits=logits)
        # logits = self.crf.decode(logits, mask=mask, nbest=3).squeeze()
        # logits = torch.nn.functional.one_hot(logits, self.num_labels)
        nbest_weight = [0.4, 0.35, 0.25]
        results = self.crf.viterbi_tags(
            logits, mask=mask, top_k=3#len(nbest_weight)
        )  # (nbest, bc, seqlen)
        # print(logits)
        # bc, best, tag, score
        padding = lambda x: x + [0]*(512-len(x))
        # print([sample for sample in results[:2]])
        final = torch.zeros((logits.shape[0], 512, self.num_labels-2,), dtype=torch.float32)
        for i, thr in enumerate(nbest_weight):
            tmp = torch.tensor([padding(sample[i][0]) for sample in results])
            tmp[tmp>=self.num_labels-2] = 0
            final += torch.nn.functional.one_hot(tmp, self.num_labels-2) * thr
        # print(final)
        # final = torch.zeros(logits.shape[1:] + (self.num_labels,)).to("cuda")  # a = a.to(b.device)
        # for i, thr in enumerate(nbest_weight):
        #     final += torch.nn.functional.one_hot(logits[i], self.num_labels) * thr
        return torch.tensor(0.0), final, targets
        # return TokenClassifierOutput(loss=loss, logits=final, labels=targets)
        


class DebertaV2CRFDataCollator(DataCollatorForTokenClassification):
    def torch_call(self, features):
        def tmpf(x):
            x[0] = self.num_labels 
            x[-1] = self.num_labels + 1
            return x
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [tmpf(feature[label_name]) for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=True, #'max_length', #self.padding,
            max_length=300, # self.max_length,
            pad_to_multiple_of=True, #self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )
        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        batch[label_name] = [
            list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
        ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch[label_name][batch[label_name]==-100] = 0
        return batch
    
    
"""
huggingface -> trasformers trainer_pt_utils.py:115
        if tensors.shape[-1] == tensors.shape[-2] and \
            new_tensors.shape[-1] == new_tensors.shape[-2] and \
            len(tensors.shape) == len(new_tensors.shape) == 4:
                # design for sparse
                tensors = tensors.to('cpu').to_dense()
                new_tensors = new_tensors.to('cpu').to_dense()
                dtype = tensors.dtype
                device = tensors.device
                shapea = tensors.shape[-1]
                shapeb = new_tensors.shape[-1]
                
                if shapea == shapeb:
                    pass
                elif shapea < shapeb:
                    diff = shapeb - shapea
                    comshape = tensors.shape[:2]
                    tensors = torch.cat([tensors, torch.zeros(size=comshape + (diff, shapea), 
                                                                          dtype=dtype, device=device)], axis=-2)
                    tensors = torch.cat([tensors, torch.zeros(size=comshape + (shapeb, diff),
                                                                          dtype=dtype, device=device)], axis=-1)
                else:
                    diff = shapea - shapeb
                    comshape = new_tensors.shape[:2]
                    new_tensors = torch.cat([new_tensors, torch.zeros(size=comshape + (diff, shapeb), 
                                                                                  dtype=dtype, device=device)], axis=-2)
                    new_tensors = torch.cat([new_tensors, torch.zeros(size=comshape + (shapea, diff),
                                                                                  dtype=dtype, device=device)], axis=-1)
                    
                return torch.cat([tensors, new_tensors],axis=0)
            

"""