"""The embedding block of BERT: token, position and segment tables summed, then LayerNorm and dropout.

A standalone copy of the `transformers` `BertEmbeddings` layer with the same parameter names, so a BERT
checkpoint loads into it key for key; `SimplifiedMLM` uses it as its only layer before the output head.
"""

from typing import Optional
import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """Token + position + segment embeddings with LayerNorm and dropout, as in BERT.

    Input ids are `(batch, seq_len)`; the output is `(batch, seq_len, hidden)`. Sized from the constructor
    arguments: the vocabulary, the hidden width, the longest sequence and the number of segment types. The
    dropout rate is fixed at 0.1.

    Args:
        vocab_size: Number of token ids.
        hidden_size: Width of every embedding table and of the output.
        max_position_embeddings: Longest sequence the position table covers.
        type_vocab_size: Number of segment (token type) ids.
        layer_norm_eps: Epsilon of the LayerNorm.
        pad_token_id: Token id whose embedding row stays zero and receives no gradient.
        position_embedding_type: Position embeddings are added only when this is `"absolute"`.
    """

    def __init__(
        self,
        vocab_size: int = 5093,
        hidden_size: int = 4,
        max_position_embeddings: int = 64,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings: nn.Embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings: nn.Embedding = nn.Embedding(type_vocab_size, hidden_size)

        # `LayerNorm` keeps the capitalised name of the transformers layer so that state-dict keys match a
        # BERT checkpoint.
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

        # Non-persistent buffers: they follow the module to its device but are derived from the constructor
        # sizes, so they are not part of a checkpoint.
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """Embed a batch of token ids (or precomputed token embeddings).

        Args:
            input_ids: Token ids, `(batch, seq_len)`; may be omitted when `inputs_embeds` is given.
            token_type_ids: Segment ids, `(batch, seq_len)`; all zeros when omitted.
            position_ids: Positions, `(1, seq_len)` or `(batch, seq_len)`; `0..seq_len-1` shifted by
                `past_key_values_length` when omitted.
            inputs_embeds: Token embeddings already looked up, `(batch, seq_len, hidden)`; replaces the
                `word_embeddings` lookup of `input_ids`.
            past_key_values_length: Offset of the first position, for a continued sequence.

        Returns:
            The summed, normalized and dropout-applied embeddings, `(batch, seq_len, hidden)`.
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # The zero buffer, sliced and expanded, stands in for absent segment ids: it already sits on the
        # module's device and keeps the module traceable without that argument.
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
