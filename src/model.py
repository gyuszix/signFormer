"""
src/model.py

Two model variants:

  SignTransformer   — CTC output (T, B, C+1). Use for future continuous/streaming signing.
  SignClassifier    — CrossEntropy output (B, C) via mean pooling. Better for isolated
                      word recognition — this is what we train for the current experiments.

Both share the same encoder architecture and accept the same (B, T, input_dim) input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — injects temporal position information
    into the frame embeddings before the transformer encoder.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignTransformer(nn.Module):
    """
    Compact transformer encoder for isolated ASL sign recognition.

    Architecture:
        Linear projection (126 → d_model)
        → Positional encoding
        → N × TransformerEncoderLayer (self-attention + FFN)
        → Linear classifier (d_model → n_classes + 1)

    The output is shaped (T, B, C) for PyTorch's nn.CTCLoss.

    Args:
        d_model:        feature dimension (default 128 for student, 512 for teacher)
        nhead:          number of attention heads
        n_layers:       number of transformer encoder layers
        dim_feedforward: FFN hidden dimension
        n_classes:      vocabulary size (number of ASL signs)
        dropout:        dropout rate
        input_dim:      keypoint feature dimension (126 for hands-only)
    """

    def __init__(
        self,
        d_model:        int = 128,
        nhead:          int = 4,
        n_layers:       int = 3,
        dim_feedforward:int = 256,
        n_classes:      int = 100,
        dropout:        float = 0.1,
        input_dim:      int = 126,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,   # expects (B, T, C)
            norm_first     = True,   # pre-norm — more stable training
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes + 1)  # +1 for CTC blank token

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (B, T, 126) keypoint sequences
            src_key_padding_mask: (B, T) bool mask — True = padded frame to ignore

        Returns:
            log_probs: (T, B, n_classes + 1) — log-softmax for CTCLoss
        """
        x = self.input_proj(x)                              # (B, T, d_model)
        x = self.pos_enc(x)                                 # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        x = self.classifier(x)                              # (B, T, C)
        x = x.permute(1, 0, 2)                              # (T, B, C) for CTCLoss
        return F.log_softmax(x, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNBaseline(nn.Module):
    """
    1D CNN baseline for ASL sign recognition.

    Architecture:
        Linear projection (126 → d_model)
        → 4 × Conv1d residual blocks (kernel=3, causal padding)
        → Linear classifier

    Captures local motion patterns (e.g. handshape transitions over 3–7 frames)
    but has no mechanism for long-range temporal dependencies. Faster than both
    LSTM and Transformer on CPU — serves as the lower bound in the comparison.

    Supports both CTC and CE loss modes:
    - CTC mode: outputs (T, B, n_classes + 1) for CTCLoss
    - CE mode: outputs (B, n_classes) for CrossEntropyLoss
    """

    def __init__(
        self,
        d_model:   int = 128,
        n_layers:  int = 4,
        n_classes: int = 300,
        dropout:   float = 0.1,
        input_dim: int = 126,
        use_ctc:   bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_ctc = use_ctc
        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.norms      = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        out_features = n_classes + 1 if use_ctc else n_classes
        self.classifier = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 126) keypoint sequences
        Returns:
            If use_ctc=True: log_probs (T, B, n_classes + 1) for CTCLoss
            If use_ctc=False: logits (B, n_classes) for CrossEntropyLoss
        """
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)

        for block, norm in zip(self.blocks, self.norms):
            residual = x
            x = block(x)
            x = x + residual
            x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x = self.classifier(x)

        if self.use_ctc:
            x = x.permute(1, 0, 2)
            return F.log_softmax(x, dim=-1)
        else:
            x = x.mean(dim=1)
            return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMBaseline(nn.Module):
    """
    Bidirectional LSTM baseline for ASL sign recognition.

    Architecture:
        2 × stacked BiLSTM (hidden=128, bidirectional → 256-dim output)
        → Dropout
        → Linear classifier

    Supports both CTC and CE loss modes:
    - CTC mode: outputs (T, B, n_classes + 1) for CTCLoss
    - CE mode: outputs (B, n_classes) for CrossEntropyLoss

    Args:
        hidden_size: LSTM hidden size per direction (default 128 → 256 bidirectional)
        n_layers:    number of stacked LSTM layers
        n_classes:   vocabulary size
        dropout:     dropout between LSTM layers and before classifier
        input_dim:   keypoint feature dimension (126)
        use_ctc:     if True, use CTC loss mode; if False, use CE loss mode
    """

    def __init__(
        self,
        hidden_size: int = 128,
        n_layers:    int = 2,
        n_classes:   int = 300,
        dropout:     float = 0.3,
        input_dim:   int = 126,
        use_ctc:     bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_ctc = use_ctc

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        out_features = n_classes + 1 if use_ctc else n_classes
        self.classifier = nn.Linear(hidden_size * 2, out_features)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (B, T, 126) keypoint sequences
            src_key_padding_mask: ignored (LSTM handles variable lengths implicitly)

        Returns:
            If use_ctc=True: log_probs (T, B, n_classes + 1) for CTCLoss
            If use_ctc=False: logits (B, n_classes) for CrossEntropyLoss
        """
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.classifier(out)

        if self.use_ctc:
            out = out.permute(1, 0, 2)
            return F.log_softmax(out, dim=-1)
        else:
            out = out.mean(dim=1)
            return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SignClassifier(nn.Module):
    """
    Transformer encoder + global average pooling for cross-entropy classification.

    Better suited for isolated word recognition than CTC:
    - One label per clip → CE is the natural loss
    - Mean pooling aggregates all frames into a single representation
    - ~10–30% higher Top-1 than CTC on this task empirically

    Output: (B, n_classes) logits — for nn.CrossEntropyLoss
    """

    def __init__(
        self,
        d_model:         int = 128,
        nhead:           int = 4,
        n_layers:        int = 3,
        dim_feedforward: int = 256,
        n_classes:       int = 300,
        dropout:         float = 0.1,
        input_dim:       int = 126,
    ):
        super().__init__()
        self.n_classes  = n_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (B, T, input_dim) keypoint sequences
            src_key_padding_mask: (B, T) bool — True = padded frame

        Returns:
            logits: (B, n_classes) — raw scores for CrossEntropyLoss
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling — ignore padded frames
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x    = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)   # (B, C)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_student_model(n_classes: int = 300) -> SignTransformer:
    """CPU-deployable student transformer (~672K params)."""
    return SignTransformer(
        d_model=128, nhead=4, n_layers=3,
        dim_feedforward=256, n_classes=n_classes, dropout=0.1)


def build_teacher_model(n_classes: int = 300) -> SignTransformer:
    """Larger teacher transformer (GPU recommended)."""
    return SignTransformer(
        d_model=512, nhead=8, n_layers=6,
        dim_feedforward=1024, n_classes=n_classes, dropout=0.1)


def build_cnn_baseline(n_classes: int = 300, loss: str = "ce") -> CNNBaseline:
    """1D CNN baseline — local temporal patterns only, fastest CPU inference."""
    use_ctc = (loss == "ctc")
    return CNNBaseline(
        d_model=128, n_layers=4, n_classes=n_classes, dropout=0.1, use_ctc=use_ctc)


def build_student_classifier(n_classes: int = 300, input_dim: int = 126,
                              d_model: int = 128, n_layers: int = 3,
                              dropout: float = 0.1) -> SignClassifier:
    """Student classifier with CE loss. Default ~451K params (d=128, l=3)."""
    nhead           = 8 if d_model >= 256 else 4
    dim_feedforward = d_model * 2
    return SignClassifier(
        d_model=d_model, nhead=nhead, n_layers=n_layers,
        dim_feedforward=dim_feedforward, n_classes=n_classes,
        dropout=dropout, input_dim=input_dim)


def build_lstm_baseline(n_classes: int = 300, loss: str = "ce") -> BiLSTMBaseline:
    """BiLSTM baseline — sequential hidden state, middle ground."""
    use_ctc = (loss == "ctc")
    return BiLSTMBaseline(
        hidden_size=128, n_layers=2, n_classes=n_classes, dropout=0.3, use_ctc=use_ctc)


def make_padding_mask(input_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a boolean padding mask for the transformer.
    True = padded (ignore), False = real data.

    Args:
        input_lengths: (B,) actual sequence lengths
        max_len:       padded sequence length T

    Returns:
        mask: (B, T) bool tensor
    """
    batch_size = input_lengths.size(0)
    mask = torch.arange(max_len, device=input_lengths.device).unsqueeze(0)
    mask = mask.expand(batch_size, -1) >= input_lengths.unsqueeze(1)
    return mask  # True where padded


if __name__ == "__main__":
    # Quick sanity check
    model = build_student_model(n_classes=100)
    print(f"Student parameters: {model.count_parameters():,}")

    x    = torch.randn(4, 80, 126)   # (B=4, T=80, keypoints=126)
    lens = torch.tensor([80, 60, 45, 30])
    mask = make_padding_mask(lens, max_len=80)
    out  = model(x, src_key_padding_mask=mask)
    print(f"Output shape: {out.shape}")  # expected: (80, 4, 101)
