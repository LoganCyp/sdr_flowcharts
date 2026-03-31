import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from gnuradio import gr

SEQ_LEN   = 128
INPUT_DIM = 2

class RadioLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=11):
        super().__init__()
        self.lstm1      = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1   = nn.Dropout(0.3)
        self.lstm2      = nn.LSTM(256, 128, batch_first=True)
        self.batchnorm  = nn.BatchNorm1d(256)
        self.fc         = nn.Linear(256, 128)
        self.dropout2   = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x     = self.dropout1(x)
        x, _ = self.lstm2(x)
        x_mean = x.mean(dim=1)
        x_max  = x.max(dim=1).values
        x      = torch.cat([x_mean, x_max], dim=1)
        x      = self.batchnorm(x)
        return self.classifier(self.dropout2(F.relu(self.fc(x))))


class blk(gr.sync_block):
    """
    RadioLSTM Modulation Classifier
    Tap point: post-AGC, pre-FLL.
    Pass-through: complex IQ flows unchanged to the rest of the chain.
    Place model.pt / scaler.joblib / class_labels.joblib in ./artifacts/
    """
    def __init__(self, artifacts_dir="artifacts", print_every=50):
        gr.sync_block.__init__(self,
            name="RadioLSTM_AMC",
            in_sig=[np.complex64],
            out_sig=[np.complex64])

        self.print_every = int(print_every)
        self.buf         = np.zeros(SEQ_LEN, dtype=np.complex64)
        self.buf_idx     = 0
        self.call_count  = 0
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.classes = joblib.load(os.path.join(artifacts_dir, "class_labels.joblib"))
            self.scaler  = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
            self.model   = RadioLSTM(input_dim=INPUT_DIM, num_classes=len(self.classes))
            self.model.load_state_dict(
                torch.load(os.path.join(artifacts_dir, "model.pt"),
                           map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"[AMC] Loaded | classes={self.classes} | device={self.device}")
            self._ready = True
        except Exception as e:
            print(f"[AMC] WARNING - could not load model: {e}")
            self._ready = False

    def _infer(self):
        iq = np.stack([self.buf.real, self.buf.imag], axis=1).astype(np.float32)
        iq = self.scaler.transform(iq)
        x  = torch.tensor(iq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            idx   = probs.argmax().item()
            conf  = probs[0, idx].item()
        return self.classes[idx], conf

    def work(self, input_items, output_items):
        inp = input_items[0]
        output_items[0][:] = inp  # pass-through

        if not self._ready:
            return len(inp)

        i = 0
        while i < len(inp):
            space = SEQ_LEN - self.buf_idx
            take  = min(space, len(inp) - i)
            self.buf[self.buf_idx:self.buf_idx + take] = inp[i:i + take]
            self.buf_idx += take
            i            += take
            if self.buf_idx == SEQ_LEN:
                self.call_count += 1
                if self.call_count % self.print_every == 0:
                    label, conf = self._infer()
                    print(f"[AMC] #{self.call_count:06d}  pred={label:<12} conf={conf:.2%}")
                self.buf_idx = 0

        return len(inp)