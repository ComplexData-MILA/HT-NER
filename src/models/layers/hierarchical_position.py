from transformers.models.bert.modeling_bert import BertEmbeddings
import torch


class BertEmbeddingsHierarchicalPosition(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings. HierarchicalPosition
    Base on transformers.models.bert.BertEmbeddings
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L166
    https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L772
    """

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[
                0, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            #### original
            position_embeddings = self.position_embeddings(position_ids)

            #### new one
            alpha = 0.4
            position_embeddings = position_embeddings - alpha * position_embeddings[:1]
            position_embeddings = position_embeddings / (1 - alpha)
            position_embeddings = (
                alpha * position_embeddings[position_ids // 512]
                + (1 - alpha) * position_embeddings[position_ids % 512]
            )
            # print("using hex now!!")
            # print(position_embeddings.shape, embeddings.shape)
            #### original
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
