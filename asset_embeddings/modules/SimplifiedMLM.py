"""The SimplifiedMLM baseline: a BERT embedding block followed directly by a vocabulary head, no encoder.

It takes token ids `(batch, seq_len)` and returns per-position logits over the vocabulary (plus the
masked-LM loss when labels are given) in the `MaskedLMOutput` form the MP trainer consumes.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import transformers.modeling_outputs

from .BertEmbeddings import BertEmbeddings


class SimplifiedMLM(nn.Module):
    """Masked-LM model with no attention: each position is predicted from its own embedding alone.

    The only layers are a `BertEmbeddings` block and a linear head `hidden -> vocab`; the logits at a
    position depend on that position's token, position and segment embeddings and on nothing else in the
    sequence, which is what makes it the baseline of the MP task. Token ids come in as `(batch, seq_len)`,
    logits go out as `(batch, seq_len, vocab)`. Sized from the constructor arguments; the preparer then
    loads a checkpoint (`strict=False`) or copies an embedding CSV into `embeddings.word_embeddings`.

    Args:
        vocab_size: Number of token ids, also the width of the head.
        hidden_size: Embedding width.
        max_position_embeddings: Longest sequence the position table covers.
        type_vocab_size: Number of segment (token type) ids.
        layer_norm_eps: Epsilon of the embedding LayerNorm.
        pad_token_id: Token id whose embedding row stays zero.
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
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
        )

        self.head = nn.Linear(hidden_size, vocab_size)  # raw logits: CrossEntropyLoss applies the softmax

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """The token embedding table, in the accessor form `transformers` models expose."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the token embedding table.

        Args:
            value: The new table; its width must equal `hidden_size`.
        """
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple | transformers.modeling_outputs.MaskedLMOutput:
        """Score every position against the vocabulary and, given labels, compute the masked-LM loss.

        Args:
            input_ids: Token ids, `(batch, seq_len)`.
            attention_mask: Accepted for call-site compatibility with the BERT trainers and ignored; there is
                no attention to mask.
            token_type_ids: Segment ids, `(batch, seq_len)`; zeros when omitted.
            labels: Target ids, `(batch, seq_len)`, with `-100` at positions that carry no loss.
            position_ids: Positions, `(1, seq_len)` or `(batch, seq_len)`; `0..seq_len-1` when omitted.
            return_dict: Return a `MaskedLMOutput` (default) instead of a tuple.

        Returns:
            A `MaskedLMOutput` with `logits` `(batch, seq_len, vocab)` and `loss` (a scalar, or `None` without
            labels), or the tuple `(logits, loss)` / `(logits,)` when `return_dict` is false.
        """
        batch_size, seq_length = input_ids.shape

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        logits = self.head(embedding_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + (loss,) if loss is not None else (logits,)
            return output

        return transformers.modeling_outputs.MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
