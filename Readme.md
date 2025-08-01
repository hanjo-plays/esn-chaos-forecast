# Chaotic Time Series Prediction Using Reservoir Computing

##  Project Overview
- **Goal:** Benchmark three neural forecasting models (FFNN, LSTM, ESN) on truly chaotic data (Lorenz system).  
- **Why it matters:** Chaotic dynamics appear in weather, finance, physiology… tiny changes → vastly different outcomes (“butterfly effect”).  
- **Core contributions:**  
  - Data generator for the Lorenz system in its chaotic regime  
  - Implementation of FFNN, auto-regressive LSTM, and Echo State Network  
  - Multi-step forecasting under two input formats (last state vs. last 10 states)  
  - Quantitative comparison (MSE/RMSE, prediction horizon, training cost, model size)