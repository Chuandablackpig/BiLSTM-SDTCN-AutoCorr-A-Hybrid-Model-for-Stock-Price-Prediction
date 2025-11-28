# BiLSTM-SDTCN-AutoCorr: A Hybrid Model for Stock Price Prediction Integrating Sequence Decomposition and Autocorrelation Attention

## 📌 Description

* The BiLSTM-SDTCN-AutoCorr model uses a dual-path encoder-decoder architecture. Input data is first processed with positional encoding and a 3-layer BiLSTM (64 units per layer, with residual connections) for initial feature extraction. A sequence decomposition module then splits these features into trend and seasonal components. The Transformer encoder employs FFT-based autocorrelation attention to efficiently capture long-term periodic patterns. In the decoder stage, multiple TCN layers and fully-connected layers (with Tanh activation) replace the standard Transformer decoder to emphasize causal temporal dependencies. Multi-level residual connections and layer normalization are applied throughout the network to stabilize training.

* Tested on five major Chinese stock indices, the model outperforms baseline methods (e.g., LSTM, GRU, Transformer) in **MSE**, **MAE**, **RMSE**, and **R²**.

---

## 🧠 Model Architecture Diagram

The following diagram illustrates the overall structure of the **BiLSTM-SDTCN-AutoCorr** model, including the BiLSTM encoder, sequence decomposition module, Auto-Correlation attention, and TCN decoder:

![Model Architecture](images/model_architecture.png)

---

## 🚀 Features

- 📉 **Sequence Decomposition**: Splits the input series into trend and seasonal components to reduce noise.
- 🔁 **Autocorrelation Attention**: Uses FFT-based self-attention to capture long-term periodic dependencies.
- 🧠 **TCN Decoder**: Replaces the Transformer decoder with Temporal Convolutional Network layers for enhanced local sequence modeling.
- 🧬 **Hybrid Architecture**: Combines BiLSTM, Transformer encoder, and TCN advantages for powerful sequence modeling.
- 📈 **Superior Performance**: Achieves significantly lower MSE/MAE/RMSE and higher R² compared to baselines.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/BiLSTM-SDTCN-AutoCorr.git
cd BiLSTM-SDTCN-AutoCorr
pip install -r requirements.txt
```

---

## 🧪 Usage

### 📁 Data Preparation

1. Place your raw stock data in the `data/` directory.
2. The system will automatically perform:
   - **Z-score normalization**
   - **Feature engineering**, including:
     - Price change
     - Percentage change
     - 5-day and 10-day moving averages
   - **Sliding window sequence generation**:
     - 20 days of historical data → predict the 21st day

Data preprocessing can be executed via:

```bash
python data_preprocessing.py
```

### 🚦 Model Training

1. Train the model with default hyperparameters:

- Epochs: 400
- window_size: 20 (sequence length)
- Batch size: 36
- Learning rate: 1e-4
- dropout: 0.2
- Optimizer: Adam
- BiLSTM: 3 layers of 64 hidden units.
- Transformer encoder: 6 layers, 8 attention heads.

2. Run with default hyperparameters:

```bash
python train.py
```

3. Adjust training settings via config_manager.py:

- window_size, batch_size, epochs, learning_rate, dropout_rate, etc.

---

## 🗂️ Project Structure

```bash
BiLSTM-SDTCN-AutoCorr/
├── core/                    # Configuration and logging
│   ├── config_manager.py
│   └── logger.py
├── data/                    # Raw and preprocessed datasets
├── models/                  # Model definitions
│   └── bilstm_mtran_tcn.py
├── modules/                 # Utility modules
│   ├── series_decomposition.py
│   └── output_layer.py
├── get_stock_data/         # Data acquisition scripts
│   ├── akshare.py
│   └── yfinance.py
├── results/                # Output results
├── train.py                # Training script
├── data_preprocessing.py   # Data preprocessing
└── requirements.txt
```

---

## 📚 Reference to Autoformer

This project incorporates ideas and components inspired by the following work:

**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**  
by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long  
NeurIPS 2021.

Paper link: https://arxiv.org/abs/2106.13008

Key techniques adapted in this project:
- Series decomposition (trend + seasonal components)
- Auto-Correlation attention mechanism

We acknowledge and appreciate the authors' contributions to the field.

---


