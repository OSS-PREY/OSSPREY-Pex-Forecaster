"""
    @brief defines the model class for interfacing with a model's info
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @acknowledgements Nafiz I. Khan, Dr. Likang Yin
    @creation-date January 2024
    @version 0.1.0
"""

# --- Environment Setup --- #
# external modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import shap
import lime
import torchviz
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
from torchviz import make_dot
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

## built-in modules
import json
import re
import copy
import math
from typing import Any, Optional
from dataclasses import dataclass, field

## DECAL modules
from decalfc.utils import *
from decalfc.abstractions.netdata import *
from decalfc.abstractions.perfdata import *


# Constants
params_dict = load_params()
weights_dir = Path(params_dict["weights-dir"])


# Model Architectures
# interface [abstract class] for models
"""
# Since the language isn't type strict, we can pretend to polymorphically call 
# __init__ and forward
class forecast_model(ABC):
    @abstractmethod
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        pass

    @abstractmethod
    def forward(self, x):
        pass
"""

## Bidirectional LSTM
"""## -- deprecated version of BRNN -- #
# class BRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(BRNN, self).__init__()
#         self.hidden_size = hidden_size 
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
#                             batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size*2, num_classes)
#         self.softmax = nn.Softmax(dim=1)
            
#     def forward(self, x):
#         if torch.isnan(x).any():
#             # set NaN to zero
#             x[x != x] = 0
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         out = x
#         # out = self.softmax(x)
#         return out

#     def predict(self, x):
#         return self.softmax(self(x))
"""
        
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.4, **kwargs):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # layer normalization layers
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        # activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x):
        # handle NaN values
        x = torch.nan_to_num(x, nan=0.0)
        
        # LSTM layer
        lstm_output, _ = self.lstm(x)
        
        # attention mechanism
        attn_output = self.attention_net(lstm_output)
        
        # fully connected layers with residual connections and layer normalization
        out = self.fc1(attn_output)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        residual = out
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual[:, :out.size(1)]  # Residual connection
        
        out = self.fc3(out)
        return out

    def predict(self, x):
        return F.softmax(self(x), dim=1)


## Bidirectional GRU
class BGNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads=1, dropout_rate=0.4, **kwargs):
        super(BGNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Bidirectional GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Multi-headed attention mechanism
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_heads)
        ])
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2 * num_heads, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Layer normalization layers
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def attention_net(self, gru_output):
        attention_outputs = []
        for attention_head in self.attention:
            attn_weights = attention_head(gru_output).squeeze(2)
            soft_attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(gru_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
            attention_outputs.append(context)
        return torch.cat(attention_outputs, dim=1)

    def forward(self, x):
        # handle NaN values
        x = torch.nan_to_num(x, nan=0.0)
        
        # GRU layer
        gru_output, _ = self.gru(x)
        
        # multi-headed attention mechanism
        attn_output = self.attention_net(gru_output)
        
        # fully connected layers with residual connections and layer normalization
        out = self.fc1(attn_output)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        residual = out
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual[:, :out.size(1)]  # Residual connection
        
        out = self.fc3(out)
        return out

    def predict(self, x):
        return F.softmax(self(x), dim=1)


## Dilated LSTM
class DRNNComponent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dilations, **kwargs):
        super(DRNNComponent, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dilations = dilations

        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden_states = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        cell_states = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            for layer in range(self.num_layers):
                dilation = self.dilations[layer]
                if t % dilation == 0:
                    if layer == 0:
                        input_t = x[:, t, :]
                    else:
                        input_t = hidden_states[layer - 1]
                    
                    hidden_states[layer], cell_states[layer] = self.cells[layer](
                        input_t, (hidden_states[layer], cell_states[layer])
                    )
            
            outputs.append(hidden_states[-1])

        return torch.stack(outputs, dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out(context)

        return output

class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5, num_heads=1, **kwargs):
        super(DRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        
        # Dilated LSTM layers with increased dilation
        self.dilation = [8 ** i for i in range(num_layers)]
        self.forward_lstm = DRNNComponent(input_size, hidden_size, num_layers, self.dilation)
        self.backward_lstm = DRNNComponent(input_size, hidden_size, num_layers, self.dilation)
        
        # multi-head attention mechanism with increased number of heads
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads * 2)
        
        # global max pooling (changed from average pooling)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # fully connected layers with increased capacity
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # layer normalization layers
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # activation and dropout
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, x):
        # handle NaN values
        x = torch.nan_to_num(x, nan=0.0)
        
        # dilated LSTM layers
        forward_output = self.forward_lstm(x)
        backward_output = self.backward_lstm(torch.flip(x, [1]))
        backward_output = torch.flip(backward_output, [1])
        
        # concatenate forward and backward outputs
        lstm_output = torch.cat((forward_output, backward_output), dim=2)
        
        # multi-head attention mechanism
        attn_output = self.attention(lstm_output)
        
        # global max pooling
        pooled_output = self.global_max_pool(attn_output.transpose(1, 2)).squeeze(2)
        
        # fully connected layers with residual connections and layer normalization
        out = self.fc1(pooled_output)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        residual = out
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = out + residual[:, :out.size(1)]  # Residual connection
        
        out = self.fc3(out)
        return out

    def predict(self, x):
        return torch.sigmoid(self(x))


## One-directional LSTM
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
           
    def forward(self, x):
        if torch.isnan(x).any():
            # set NaN to zero
            x[x != x] = 0
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        out = x
        # out = self.softmax(x)
        return out

    def predict(self, x):
        return self.softmax(self(x))


## Bidrectional LSTM w/ Sigmoid Output
class S_BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        super(S_BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.sigmoid = nn.Sigmoid()
          
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        # out = self.sigmoid(x)
        out = x
        return out

    def predict(self, x):
        return self.sigmoid(self(x))


## Bidirectional LSTM w/ Batch Normalization
class BN_BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        super(BN_BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:,-1,:])
        # out = self.softmax(x)
        out = x
        return out
    
    def predict(self, x):
        return self.softmax(self(x))


## Transformer
class PositionalEncoding(nn.Module):
    """
    Implements the standard positional encoding (sinusoidal) as used in the Transformer.
    """
    def __init__(self, d_model, dropout: float=0.1, max_len: int=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # pe matrix dependent on woul
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # exponential decay
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        
        # register as a buffer to save with model but avoid training
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        
        # unpack constants
        seq_len = x.size(1)

        # add encoding to the embeddings
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class TNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int=128, num_heads: int=2,
        num_layers: int=1, dropout_rate: float=0.1, num_classes: int=2, **kwargs
    ):
        """
        Args:
            input_dim (int): Number of features per timestep.
            model_dim (int, optional): Dimension to project each timestep into. 
                Defaults to 64.
            num_heads (int, optional): Number of attention heads in each 
                transformer encoder layer. Defaults to 4.
            num_layers (int, optional): Number of transformer encoder layers. 
                Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        
        # initialize model
        super().__init__()
        
        # project inputs onto the hidden-size dimensional space
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate)
        
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # learnable CLS parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # output head
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor of shape (batch_size,) representing probability outputs.
        """
        
        # unpack
        batch_size, seq_len, _ = x.size()
        
        # linear projection
        x = self.input_projection(x)  # shape: (batch_size, seq_len, model_dim)
        
        # prepend the CLS token, one per batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, model_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # new sequence length is seq_len + 1
        
        x = self.pos_encoder(x)  # shape: (batch_size, seq_len+1, model_dim)
        x = self.transformer_encoder(x)  # shape: (batch_size, seq_len+1, model_dim)
        
        # extract the CLS output
        cls_representation = x[:, 0, :]  # shape: (batch_size, model_dim)
        
        # pass through classification head
        out = self.fc(cls_representation)  # shape: (batch_size, 1)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)  # shape: (batch_size,)

    def predict(self, x):
        return F.softmax(self(x), dim=1)

## Regressor (from Nafiz)
class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, **kwargs):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Class
@dataclass
class TimeSeriesModel:
    # data
    ## user input
    model_arch: str = field(default="BLSTM")                                    # model architecture
    hyperparams: dict[str, Any] = field(default=None)                           # hyperparameters
    bias_weights: bool = field(default=False)                                   # control weights for the loss fn
    
    ## inferred
    strategy: str = field(default="")                                           # strategy used in training model
    report_name: str = field(init=False)                                        # export report as this name
    is_interval: dict[str, bool] = field(
        default_factory=lambda: dict()
    )                                                                           # info on whether or not the tensors are intervaled
    device: Any = field(init=False, repr=False)                                 # device to use for training
    model: Any = field(init=False, repr=False)                                  # model
    loss_fc: Any = field(init=False, repr=False)                                # loss function
    optimizer: Any = field(init=False, repr=False)                              # optimizer function
    scheduler: Any = field(default=None, repr=False)                            # scheduler for the learning rate
    
    preds: Any = field(init=False, repr=False)                                  # predictions
    targets: Any = field(init=False, repr=False)                                # targets

    # internal utility
    def gen_model(self) -> None:
        """
            Generates the model architecture based on the options given.
        """

        # router
        match self.model_arch:
            case "BLSTM":
                log("Model Chosen :: Bidirectional LSTM", "new")
                self.model = BRNN(
                    **self.hyperparams
                ).to(self.device)
            
            case "BGRU":
                log("Model Chosen :: Bidirectional Gated Recurrent Unit [GRU]", "new")
                self.model = BGNN(
                    **self.hyperparams
                ).to(self.device)

            case "DLSTM":
                log("Model Chosen :: Dilated LSTM", "new")
                self.model = DRNN(
                    **self.hyperparams
                ).to(self.device)
                
            case "LSTM":
                log("Model Chosen :: One-Directional LSTM", "new")
                self.model = RNN(
                    **self.hyperparams
                ).to(self.device)
            
            case "S_BLSTM":
                log("Model Chosen :: Sigmoid Bidirectional LSTM", "new")
                self.model = S_BRNN(
                    **self.hyperparams
                ).to(self.device)
            
            case "BN_BLSTM":
                log("Model Chosen :: Batch Normalized Bidirectional LSTM", "new")
                self.model = BN_BRNN(
                    **self.hyperparams
                ).to(self.device)

            case "Transformer":
                log("Model Chosen :: Transformer", "new")
                self.model = TNN(
                    **self.hyperparams
                ).to(self.device)
            
            case "Regressor":
                log("Model Chose :: Regressor", "new")
                self.model = Regressor(
                    self.hyperparams["input_size"],
                    32,
                    16
                ).to(self.device)
                self.hyperparams["num_epochs"] = 120
                self.hyperparams["learning_rate"] = 1e-5
                self.hyperparams["batch_size"] = 32

            case _:
                log(f"model architecture `{self.model_arch}` undefined", "error")
                exit(1)


    def save(self, strategy: str, metrics: dict[str, float]) -> None:
        """
            Saves the model weights for later loading if needed.
            
            @param strategy: transfer strategy
            @param metrics: lookup of performance (accuracy, f1-score, precision,
                recall); note these are all macro-avg for the `all` months
        """
        
        # format metrics
        metrics = {m: f"{v:.4f}" if m != "accuracy" else v for m, v in metrics.items()}
        metrics["accuracy"] = f"{metrics['accuracy'] * 100:.2f}"
        
        # ensure directory
        cleaned_strat = re.sub(r"\s*\+\s*", "_", strategy)                      # cleaned strategy string
        dir = weights_dir / f"{cleaned_strat}/"
        check_dir(dir)
        
        # generate path
        path = Path(params_dict["model-weights-format"].format(
            cleaned_strat,
            self.model_arch,
            metrics["accuracy"],
            metrics["f1-score"],
            metrics["precision"],
            metrics["recall"]
        ))
        
        # save to path
        check_dir(path.parent)
        torch.save(self.model.state_dict(), path)
        log(f"saved model weights to \"{path}\"")
        
    
    def load(self, strategy: str, *args, **kwargs) -> bool:
        """
            Loads the best model without having to train; the assumption here is
            that some method already exists for pruning the saved model weights
            to ensure only high quality trials remain. How this is determined 
            will depend on the version.
            
            Initially, trials will have to be manually pruned. In future 
            versions, a small-scale database may be implemented to keep track of 
            trials and their performances, allowing easy querying.
            
            @param strategy: strategy for transfer
            
            @returns succesful loading or not
        """
        
        # check saves exist
        cleaned_strat = re.sub(r"\s*\+\s*", "_", strategy)
        dir = weights_dir / f"{cleaned_strat}/"
        
        if not os.path.exists(dir):
            log("strategy has no saved weights", "warning")
            return False
            
        # check model has prior weights
        saved_models = list(os.listdir(dir))
        matched_weights = ([
            weight_path for weight_path in saved_models \
            if self.model_arch in Path(weight_path).stem
        ])

        if len(matched_weights) < 1:
            log(f"model architecture has no saved weights for <{cleaned_strat}>", "warning")
            return False
        
        # load best model weight, prioritize perf in order of metric listing
        matched_weights.sort()
        best_weights = Path().cwd().parent / "model-weights" / cleaned_strat / matched_weights[-1]
        
        log(f"using <{cleaned_strat}  {best_weights.stem}> for the model")
        self.model.load_state_dict(torch.load(best_weights))
        self.model.eval()
        return True


    def check_device(self) -> None:
        """
            Attempts to use CUDA enabled hardware if possible.
        """

        # check & report
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using ***{self.device}*** for training...")


    def gen_report_name(self) -> None:
        """
            Generates a unique identifier for the model for later retrieval of 
            info. Uses:
                - model_arch
                - date & time
        """

        import time
        self.report_name = f"{self.model_arch}-{time.time()}"


    def __post_init__(self):
        # generate model archs
        log("Model Setup", "new")
        self.hyperparams = load_hyperparams(self.hyperparams)
        self.check_device()
        self.gen_model()

        # optimizers
        # class_weights = compute_class_weight("balanced", np.unique(y), y.numpy())
        class_weights = None if not self.bias_weights else \
            torch.tensor([1 - 10 / 441, 1 - 431/441], dtype=torch.float).to(self.device)
        
        if "Regressor" in self.model_arch:
            self.loss_fc = nn.MSELoss()
        else:
            self.loss_fc = nn.BCEWithLogitsLoss(
                weight=class_weights
            )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"]
        )
        
        if "scheduler" in self.hyperparams:
            # match scheduler formula
            match self.hyperparams["scheduler"]:
                case "plateau":
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            
            # intialize
            self.scheduler = self.scheduler(optimizer=self.optimizer)


    def test_reg(self, X, y, raw_prob: bool=False) -> dict[str, list[Any]]:
        """
            Regular trials.
        """

        # setup
        preds_list = np.empty(0)
        targets_list = np.empty(0)
        missed_projects = []

        # generate prediction
        for i, (data, target) in enumerate(zip(X, y)):
            # transform data to use
            data = data.to(self.device)# .squeeze(1)
            data = data.reshape(1, data.shape[0], -1)

            targets = target.to(self.device)

            if raw_prob:
                preds = self.model.predict(data)[:, 1].to(self.device)  # grab probability of success
            else:
                preds = torch.argmax(self.model.predict(data), dim=1).to(self.device)
                
                # mismatched
                if targets.cpu().detach().numpy()[0] != preds.cpu().detach().numpy()[0]:
                    missed_projects.append(i)

            # concatenate lists
            preds_list = np.concatenate((preds_list, preds.cpu().detach().numpy()))
            targets_list = np.concatenate((targets_list, targets.cpu().detach().numpy()))

        # reshape
        preds_list.flatten()
        targets_list.flatten()

        return {"preds": preds_list, "targets": targets_list, "missed-projects": missed_projects}


    def test_intervaled(self, X_dict, y_dict, raw_prob: bool=False) -> dict[str, dict[str, list[Any]]]:
        """
            Defines the testing strategy specifically for the intervaled 
            trials, returning information for each month.
        """

        # setup
        preds_dict = dict()
        targets_dict = dict()

        # generate prediction by month
        for month in tqdm(X_dict):
            # unpack
            X = X_dict[month]
            y = y_dict[month]

            # gen preds
            month_results = self.test_reg(X, y, raw_prob)
            preds_dict[month] = month_results["preds"]
            targets_dict[month] = month_results["targets"]

        # export
        return {"preds": preds_dict, "targets": targets_dict}


    # external utility
    def train(
        self, md, save_epochs: bool=False, validation_loss: bool=True, 
        attempt_load: bool=False
    ) -> None:
        """
            Trains the model on the necessary data.

            @param md: ModelData Object to train on
            @param soft_prob_model: **FUTURE** will introduce soft probabilities 
                for training on intervaled data ::: deprecated in future 
                versions, will instead use modeldata
            @param save_epochs: Whether to save the best model during training
            @param validation_loss: Whether to compute validation loss
            @param attempt_load: Whether to attempt to load an existing model or 
                not
        """

        # strategy
        self.strategy = md.transfer_strategy
        
        # attempt to greedy load if possible
        if attempt_load and self.load(strategy=self.strategy):
            return
        
        # track losses
        losses = {}
        test_losses = {}
        best_loss = float("inf")
        best_epoch = 0
        patience = 10
        TOLERANCE = 1e-4

        # interval training
        self.is_interval = md.is_interval
        
        ## check if training on intervaled and if any targets are soft probs by 
        ## just using the first entry's targets; heuristic, but should work
        is_soft_prob = (self.is_interval["train"]) and (
            any(
                ((not np.isclose(prob.cpu(), 0.0)) and (not np.isclose(prob.cpu(), 1.0))) 
                for prob in md.tensors["train"]["y"]
            )
        )

        if (self.is_interval["train"]) and is_soft_prob:
            log("found soft probabilities for training")
        elif (self.is_interval["train"]) and (not is_soft_prob):
            log("training directly on intervals without soft probabilities")

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # training
        for epoch in range(self.hyperparams["num_epochs"]):
            # setup
            self.model.train()
            losses[epoch] = []

            ## iterate batches
            for data, target in tqdm(list(zip(md.tensors["train"]["x"], md.tensors["train"]["y"]))):
                # transform data for training
                data = data.to(self.device)
                data = data.reshape(1, data.shape[0], -1)
                target = target.to(self.device).to(torch.float32)
                
                # forward; grab the probability of success
                pred = self.model.predict(data)
                pred = pred[..., 1].to(torch.float32)

                # backward
                loss = self.loss_fc(pred, target)   
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # optimizer step
                self.optimizer.step()
                
                # loss
                losses[epoch].append(loss.item())
                        
            # validation loss & loss tracking
            X_test = md.tensors["test"]["x"]
            y_test = md.tensors["test"]["y"]
            
            # only gauge fit on full projects
            if md.is_interval["test"]:
                X_val = X_test["all"]
                y_val = y_test["all"]
            else:
                X_val = X_test
                y_val = y_test

            if validation_loss:
                self.model.eval()
                test_losses[epoch] = []

                with torch.no_grad():                    
                    # for every test tensor; treated as validation here
                    for data, target in list(zip(X_val, y_val)):
                        # transform data
                        data = data.to(self.device)
                        data = data.reshape(1, data.shape[0], -1)
                        target = target.to(self.device).to(torch.float32)

                        # prediction
                        pred = self.model(data)[..., 1].to(torch.float32)

                        # track loss
                        test_losses[epoch].append(self.loss_fc(pred, target).item())
            
            losses[epoch] = np.mean(losses[epoch])
            test_losses[epoch] = np.mean(test_losses[epoch])

            current_lr = self.optimizer.param_groups[0]["lr"]
            log(f"Epoch [{epoch + 1}/{self.hyperparams['num_epochs']}] | "
                      f"Loss: {losses[epoch]:.4f}, Test Loss: {test_losses[epoch]:.4f}, "
                      f"LR: {current_lr:.6f}", "log")

            # scheduler step
            if self.scheduler is not None:
                if not validation_loss:
                    log("Unable to schedule step without validation loss", "error")
                else:
                    self.scheduler.step(test_losses[epoch])

            # early stopping; avg the test and train perf since we don't have enough for validation
            # avg_loss = (test_losses[epoch] + losses[epoch]) / 2
            avg_loss = losses[epoch]

            if avg_loss < best_loss - TOLERANCE:
                best_loss = avg_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())      
                patience = 10
                best_epoch = epoch
                if save_epochs:
                    torch.save(self.model.state_dict(), f"best_model_epoch_{epoch}.pth")
            else:
                patience -= 1

                if patience == 0:
                    log("Early stopping triggered. Loading best model weights.", "log")
                    self.model.load_state_dict(best_model_weights)
                    break

        if np.isnan(sorted(losses.items(), reverse=True)[0][1]) or np.isinf(sorted(losses.items(), reverse=True)[0][1]):
            log("NaN or Inf loss generated, i.e. failed to converge: ignoring and exiting", "error")
            return

        log("Training completed.", "log")

        print(f"Model Name: {self.model_arch}")
        print(f"Input size: {self.hyperparams['input_size']}")
        print(f"Hidden size: {self.hyperparams['hidden_size']}")
        print(f"Number of layers: {self.hyperparams['num_layers']}")

        # x = torch.randn(1, input_size)
        # y = model(x)
        # dot = make_dot(y, params=dict(model.named_parameters()))
        # dot.render("model_architecture", format="png")

        # visualize loss
        dir = "../model-reports/loss-visualization/"
        check_dir(dir)

        df = pd.DataFrame(list(losses.items()), columns=["Epoch", "Loss"])
        test_df = pd.DataFrame(list(test_losses.items()), columns=["Epoch", "Loss"])

        # sns.set_style("darkgrid")
        # sns.lineplot(x="Epoch", y="Loss", data=df, color="darkred", label="train_loss")
        # sns.lineplot(x="Epoch", y="Loss", data=test_df, color="lightblue", label="test_loss")
        # plt.axvline(best_epoch, color="purple", label="chosen_weights")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title(f"Loss per Epoch for {md.transfer_strategy}")
        # plt.text(1, 1, "\n".join(f"{k}: {v}" for k, v in self.hyperparams.items()),
        #          horizontalalignment="right", verticalalignment="top", 
        #          transform=plt.gca().transAxes)

        # # plt.savefig(f"{dir}[{md.transfer_strategy}].png")
        # plt.clf()


    def test(self, md, raw_prob: bool=False) -> None:
        """
            Runs the model on the test data and returns the results. NOTE for 
            intervaled testing we assume all testing incubators have been 
            intervaled; while this likely won't error and work as expected since 
            `all` months is a valid entry, it is NOT explicitly handled.

            @param md: ModelData object that contains the data to train on
        """

        # router
        self.is_interval = md.is_interval
        
        if self.is_interval["test"]:
            test_results = self.test_intervaled(md.tensors["test"]["x"], md.tensors["test"]["y"], raw_prob=raw_prob)
        else:
            test_results = self.test_reg(md.tensors["test"]["x"], md.tensors["test"]["y"], raw_prob=raw_prob)
        
        self.preds = test_results["preds"]
        self.targets = test_results["targets"]
        self.case_studies = test_results.get("missed-projects", list())


    def soft_probs(self, interval_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
            Generates soft probabilities for every interval in the given 
            dictionary.

            @param interval_data (dict[str, torch.Tensor]): lookup of interval 
                period to the data that reflects the given interval period.

            @returns dict[str, torch.Tensor]: lookup of interval period to the 
                raw probability prediction of that interval period.
        """
        
        # setup
        soft_probs = dict()

        # generate prediction
        for interval_step, data in interval_data.items():
            # transform data to use
            data = data.to(self.device)
            data = data.reshape(1, data.shape[0], -1)
            preds = self.model.predict(data)[:, 1].to(self.device)  # grab probability of success

            # concatenate lists
            soft_probs[interval_step] = torch.tensor(
                preds.cpu().detach().numpy()
            ).to(self.device)

        # return the interval period : soft prob lookup
        return soft_probs


    def report(self, display=False, save=True) -> str:
        """
            Generates a performance report of the model.

            @param display: True if report should be printed
            @param save: True if report should be saved
        """

        # classification report
        if self.is_interval.get("test", False):
            l = list(range(2))
            report = ""

            # for m, p in self.preds.items():
            #     report = f"\n\n{classification_report(self.targets[m], p, labels=l)}"
            report = "\n\n".join(
                [classification_report(
                    self.targets[m],
                    p,
                    labels=l,
                    zero_division=0.0
                ) for m, p in self.preds.items()]
            )
            print_report = "\n\n".join(
                [classification_report(
                    self.targets[m],
                    p,
                    labels=l,
                    zero_division=0.0
                    ) for m, p in self.preds.items() \
                      if m == "all" or int(m.split("-")[0]) % 25 == 0]
            )
            
            # cache model
            report_dict = classification_report(
                self.targets["all"], self.preds["all"], labels=list(range(2)), 
                zero_division=0.0, output_dict=True
            )
            self.save(self.strategy, metrics={
                "accuracy": report_dict["accuracy"],
                "f1-score": report_dict["macro avg"]["f1-score"],
                "precision": report_dict["macro avg"]["precision"],
                "recall": report_dict["macro avg"]["recall"]
            })
        else:
            # generate report
            report = classification_report(self.targets, self.preds, 
                                           labels=list(range(2)), 
                                           zero_division=0.0)
            report_dict = classification_report(
                self.targets, self.preds, labels=list(range(2)), 
                zero_division=0.0, output_dict=True
            )
            
            # save model
            self.save(self.strategy, metrics={
                "accuracy": report_dict["accuracy"],
                "f1-score": report_dict["macro avg"]["f1-score"],
                "precision": report_dict["macro avg"]["precision"],
                "recall": report_dict["macro avg"]["recall"]
            })
            
            # export report
            print_report = report

        if save:
            with open(f"../model-reports/trials/{self.report_name}.txt", "w") as f:
                f.write(report)
        if display:
            print(print_report)

    
    def save_weights(self) -> None:
        """
            Saves the model weights for later retrieval.
        """

        # export weights
        log("DEPRECATED save method, not saving w/ perf info", "warning")
        torch.save(
            self.model.state_dict(),
            f"../model-reports/transfer-weights/{self.report_name}.pt"
        )

    
    def interpret(self, md, strategy: str="SHAP") -> dict[str, float]:
        """
            Generates an interpretation of the model via SHAP (or some model 
            agnostic program) and pushed out a dictionary of the importance for 
            each feature.
            
            @param md (ModelData): instance that contains the data needed for 
                interpretation
            @param strategy (str): one of "SHAP" or "LIME"
        """
        
        # routing
        if strategy.upper() == "SHAP":
            ## wrapper class to ensure eval mode
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    was_training = self.model.training
                    self.model.train()
                    with torch.set_grad_enabled(True):
                        output = self.model(x)
                    if not was_training:
                        self.model.eval()
                    return output

            return self.shap_analysis(md, ModelWrapper)

        if strategy.upper() == "LIME":
            return self.lime_analysis(md)

        raise ValueError("didn't specify valid interpret strategy")


    def lime_analysis(self, md):
        X_train = md.tensors["train"]["x"]
        X_test = md.tensors["test"]["x"]
        print(f"Shape of X_train: {X_train[2].shape}")
        
        # Determine the number of features and maximum sequence length
        n_features = X_train[0].shape[-1]  # Should be 14
        max_seq_length = max(len(seq) for seq in X_train)  # Should be 530
        
        log(f"Number of features: {n_features}, Max sequence length: {max_seq_length}", "log")
        feature_names = [f"f{i}_t{t}" for t in range(max_seq_length) for i in range(n_features)]

        # Pad and flatten the training data
        X_train_padded = []
        for sequence in X_train:
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.cpu().numpy()
            padded_seq = np.pad(sequence, ((0, max_seq_length - len(sequence)), (0, 0)), mode='constant')
            X_train_padded.append(padded_seq.flatten())
        X_train_flat = np.array(X_train_padded)

        log(f"Shape of flattened training data: {X_train_flat.shape}", "log")

        # Create a LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_flat,
            feature_names=feature_names,
            class_names=["graduated", "retired"],
            mode="classification"
        )

        # Function to get model prediction for a single sample
        def predict_proba(x):
            log(f"Input shape to predict_proba: {x.shape}", "log")
            
            # Reshape x to match the expected input of the model
            x = x.reshape(-1, max_seq_length, n_features)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            log(f"Reshaped tensor shape: {x_tensor.shape}", "log")
            
            with torch.no_grad():
                output = self.model(x_tensor)
            
            # Ensure output is a proper probability distribution
            output = torch.nn.functional.softmax(output, dim=-1)
            return output.cpu().numpy()
        
        total_importance = {f"feature_{i}": 0 for i in range(n_features)}
        print("total importance: ", total_importance)
        

        for i in range(0, len(X_test)):   
            X_test_sample = X_test[i]

            if isinstance(X_test_sample, torch.Tensor):
                X_test_sample = X_test_sample.cpu().numpy()
            X_test_padded = np.pad(X_test_sample, ((0, max_seq_length - len(X_test_sample)), (0, 0)), mode='constant')
            X_test_flat = X_test_padded.flatten()

            log(f"Shape of flattened test sample: {X_test_flat.shape}", "log")

            print("Features to consider", X_test_flat.shape[0])
            FeaturesToConsider = int(X_test_flat.shape[0])

            # Explain the entire sequence at once
            explanation = explainer.explain_instance(
                X_test_flat,
                predict_proba,
                num_features=FeaturesToConsider,
                num_samples=1000
            )

            available_labels = explanation.available_labels()
            print(f"Available labels: {available_labels}")
            
            if(explanation.available_labels()[0] == 0):
                continue

            for feature, importance in explanation.as_list(label=explanation.available_labels()[0]):
                # print(extract_feature_name(feature), importance)
                # feature_name = extract_feature_name(feature)
                # feature_name = feature_name.split('_')[0][1:]  # Extract feature number
                total_importance[f"feature_{feature}"] += (importance)


        log("Aggregated Feature Importance (LIME):", "log")
        for feature, importance in total_importance.items():
            log(f"{feature}: {importance:.4f}", "log")

        log("LIME analysis completed.", "log")


    def shap_analysis(self, md, ModelWrapper):
        X_train = md.tensors["train"]["x"]
        X_test = md.tensors["test"]["x"]
        if md.is_interval["train"]:
            X_train = X_train["all"]
        if md.is_interval["test"]:
            X_test = X_test["all"]
                
        print(f"Shape of X_train: {X_train[2].shape}")

        # determine the number of features and maximum sequence length
        n_features = X_train[0].shape[-1]  # Should be 13/14 depending on if feature subset was taken or not
        max_seq_length = max(len(seq) for seq in X_train)  # Should be 530
        log(f"Number of features: {n_features}, Max sequence length: {max_seq_length}", "log")
        feature_names = [f"f{i}_t{t}" for t in range(max_seq_length) for i in range(n_features)]

        # pad the training data
        X_train_padded = []
        for sequence in X_train:
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.cpu().numpy()
            padded_seq = np.pad(sequence, ((0, max_seq_length - len(sequence)), (0, 0)), mode='constant')
            X_train_padded.append(padded_seq)
        X_train_padded = np.array(X_train_padded)

        log(f"Shape of padded training data: {X_train_padded.shape}", "log")

        def compute_shap_values(model_wrapper, inputs):
            inputs = inputs.detach().requires_grad_(True)
            outputs = model_wrapper(inputs)
                
            if outputs.shape[1] > 1:  # Multi-class
                shap_values = []
                for target_class in range(outputs.shape[1]):
                    outputs[:, target_class].sum().backward(retain_graph=True)
                    shap_values.append(inputs.grad.cpu().numpy().copy())
                    inputs.grad.zero_()
            else:  # Binary classification
                outputs.sum().backward()
                shap_values = [inputs.grad.cpu().numpy().copy()]
                
            return shap_values

        total_importance = {f"feature_{i}": 0 for i in range(n_features)}
        print("total importance: ", total_importance)

        # Wrap the model
        model_wrapper = ModelWrapper(self.model)

        for i in range(0, len(X_test)):
            X_test_sample = X_test[i]

            if isinstance(X_test_sample, torch.Tensor):
                X_test_sample = X_test_sample.cpu().numpy()
            X_test_padded = np.pad(X_test_sample, ((0, max_seq_length - len(X_test_sample)), (0, 0)), mode='constant')

            log(f"Shape of padded test sample: {X_test_padded.shape}", "log")

            # Convert to tensor
            X_test_tensor = torch.tensor(X_test_padded.reshape(1, max_seq_length, n_features), dtype=torch.float32).to(self.device)

            # Compute SHAP values
            shap_values = compute_shap_values(model_wrapper, X_test_tensor)

            # Assuming binary classification, we'll use the positive class (index 0)
            # print("Shap Values - before", shap_values)
            shap_values = shap_values[1]
            # print("Shap Values - after", shap_values)

            # Aggregate importance across time steps
            for feature in range(n_features):
                importance = np.sum(shap_values[0, :, feature])
                total_importance[f"feature_{feature}"] += importance

        log("Aggregated Feature Importance (SHAP-like):", "log")
        for feature, importance in total_importance.items():
            log(f"{feature}: {importance:.4f}", "log")

        log("SHAP-like analysis completed.", "log")
        
    
    def visualize_model(self, data_shape):
        # dummy input
        dummy_input = torch.zeros(1, *data_shape)
        dummy_input = dummy_input.to(next(self.model.parameters()).device)

        # computational graph from the model and dummy input
        output = self.model(dummy_input)
        dot = make_dot(output, params=dict(self.model.named_parameters()))

        # save model graph
        output_dir = "../model-reports/model-visualizations/"
        check_dir(output_dir)
        dot.render(f"{output_dir}{self.gen_report_name()}", format="png")
    
    ## class utility
    def clean_weights(dir: Path | str=None) -> dict[str, int]:
        """
            Script to only keep the best model weights. Assumes a directory 
            structure of dir > {transfer_strategy} > model weights. Also assumes
            all weights follow the performance graded format, i.e. must have the 
            performance values in the name itself.

        Args:
            dir (Path | str, optional): directory to clean. Defaults to 
                weights dir.

        Returns:
            dict[str, int]: statistics, e.g. num_removed (number of files 
                removed, int), num_strategies (number of strategies 
                inspected, int), mem_saved (memory in Mb saved, float)
        """
        
        # aux fn
        def clean_trial(t_dir: Path, stats: dict[str, int]) -> None:
            # organize weights
            weights = list(t_dir.iterdir())
            weights = [weight.stem for weight in weights]
            stats["num_strategies"] += 1
            
            # the only trials that we may want to keep are either 100 accuracy 
            # or the highest accuracy trial that exists; so let's unpack the 
            # values
            def extraction(weight_str):
                weight_items = weight_str.split("-")
                return weight_items[0], *[float(weight_items[i][2:-1]) for i in range(1, 5)]
            
            ## lookup creation
            lookup = {
                "arch": list(),
                "acc": list(),
                "f1": list(),
                "prec": list(),
                "rec": list(),
            }
            
            for weight_str in weights:
                arch, a, f, p, r = extraction(weight_str)
                
                lookup["arch"].append(arch)
                lookup["acc"].append(a)
                lookup["f1"].append(f)
                lookup["prec"].append(p)
                lookup["rec"].append(r)
            
            lookup = pd.DataFrame(lookup)
            
            # get best entries
            best_weights = lookup.sort_values(
                by=["acc", "f1", "prec", "rec"], ascending=False
            ).groupby("arch").first().reset_index()
            
            best_entries = {
                f"{row['arch']}-[a{row['acc']:.2f}]-[f{row['f1']:.4f}]-[p{row['prec']:.4f}]-[r{row['rec']:.4f}]"
                for _, row in best_weights.iterrows()
            }
            
            # eliminate any non-best entry
            for weight_file in t_dir.iterdir():
                ## skip dirs
                if not weight_file.is_file():
                    continue
                    
                ## remove file if not the best
                if weight_file.stem in best_entries:
                    continue
                
                stats["num_removed"] += 1
                stats["mem_saved"] += weight_file.stat().st_size
                weight_file.unlink()
        
        # ensure arguments
        if dir is None:
            dir = Path().cwd().parent / "model-weights"
        dir = Path(dir)
        
        # track statistics
        stats = {stat: 0 for stat in ["num_removed", "num_strategies", "mem_saved"]}
        
        # walk through each directory
        for trial_dir in dir.iterdir():
            ## skip any files; shouldn't exist but for safety
            if not trial_dir.is_dir():
                continue
            
            ## for every directory, keep only the best trials
            clean_trial(trial_dir, stats)
        
        # report
        stats["mem_saved"] /= 1e6
        
        log("", "summary")
        for stat, val in stats.items():
            if stat == "mem_saved":
                log(f"{stat.replace('_', ' ').title()}: {val:.4f} Mb")
            else:
                log(f"{stat.replace('_', ' ').title()}: {val}")
    
# Utility
def load_hyperparams(new_hp: dict[str, Any]) -> dict[str, Any]:
    """
        Overwrites any default hyperparams with any specific hyperparams defined.
    """

    # default
    hyperparams = {
        "input_size": 14,
        "hidden_size": 512,
        "num_classes": 2,
        # # "num_heads": 1,   # has to be specified to override default value since nheads varies across model archs, gets too confusing
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        # "batch_size": 16,
        "num_epochs": 1,
        "num_layers": 3,
        "scheduler": "plateau"
    }

    # overwrite
    if new_hp != None:
        for k, v in new_hp.items():
            hyperparams[k] = v
    
    # export
    return hyperparams
    


# Testing
if __name__ == "__main__":
    TimeSeriesModel.clean_weights()
