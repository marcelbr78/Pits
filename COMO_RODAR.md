## COMO RODAR O PITS — GUIA COMPLETO

### PASSO 1 — Instalar dependências novas (uma vez só)

```bash
pip install fastapi uvicorn xgboost scikit-learn joblib pyarrow pandas numpy
# Opcional — para LSTM e GNN (Fase 5):
pip install torch
pip install torch-geometric   # só se quiser GNN real
```

---

### PASSO 2 — Organizar arquivos

Estrutura esperada dentro da pasta `Pits/`:

```
Pits/
├── brain/
│   ├── orchestrator.py          ← Fase 1 (original)
│   ├── orchestrator_v2.py       ← Fase 2 (novo)
│   └── orchestrator_final.py   ← Fases 3-5 (novo)
├── feature_engine/
│   ├── ofi_calculator.py        ← Fase 1
│   ├── obi_calculator.py        ← Fase 2 (novo)
│   ├── trade_flow.py            ← Fase 2 (novo)
│   ├── lag_features.py          ← Fase 2 (novo)
│   ├── vwap_deviation.py        ← Fase 2 (novo)
│   ├── feature_pipeline_v2.py  ← Fase 2 (novo)
│   ├── atr_calculator.py        ← Fase 3 (novo)
│   └── advanced_features.py    ← Fase 4 (novo)
├── market_intelligence/
│   ├── regime_detector.py       ← Fase 1
│   ├── volatility_regime.py     ← Fase 1
│   ├── intelligence_pipeline_v2.py ← Fase 2 (novo)
│   ├── economic_calendar.py    ← Fase 2 (novo)
│   ├── macro_regime.py          ← Fase 2 (novo)
│   ├── pattern_library.py       ← Fase 3 (novo)
│   └── anomaly_detector.py     ← Fase 4 (novo)
├── ml_engine/
│   ├── bayesian_model.py        ← Fase 1
│   ├── xgboost_model.py         ← Fase 1
│   ├── ensemble_model.py        ← Fase 1
│   ├── ml_pipeline.py           ← Fase 1
│   ├── dataset_builder_v2.py   ← Fase 4 (novo)
│   ├── ml_pipeline_v2.py        ← Fase 4 (novo)
│   ├── lstm_model.py            ← Fase 5 (novo)
│   ├── gnn_model.py             ← Fase 5 (novo)
│   └── ensemble_v2.py           ← Fase 5 (novo)
├── risk_engine/
│   ├── manager.py               ← Fase 1
│   ├── manager_v2.py            ← Fase 4 (novo)
│   ├── portfolio_risk.py        ← Fase 4 (novo)
│   └── monte_carlo.py           ← Fase 4 (novo)
├── paper_trading/
│   ├── paper_trading_engine.py  ← Fase 1
│   └── paper_trading_engine_v2.py ← Fase 3 (novo)
├── execution_engine/
│   ├── execution_pipeline.py    ← Fase 1
│   └── execution_pipeline_v2.py ← Fase 4 (novo)
├── dashboard/
│   └── index.html               ← Dashboard novo (substituído)
├── models/                      ← criado automaticamente ao treinar
│   ├── bayesian_model.pkl
│   ├── xgboost_model.json
│   └── scaler_v2.pkl
├── data/ticks/                  ← dados históricos coletados
├── run_system_test.py           ← Fase 1 original
├── run_system_test_v2.py        ← Fase 2
├── run_pits_final.py            ← Fases 3-5 ← USE ESSE
├── train_pits_model.py          ← treino Fase 1
└── train_pits_model_v2.py       ← treino Fases 4-5 ← USE ESSE
```

---

### PASSO 3 — Treinar os modelos

**Primeira vez** (sem dados históricos ainda → modelos ficam com fallback 0.5):
```bash
cd Pits
python train_pits_model_v2.py --skip-lstm --skip-gnn
```

**Com dados históricos** (depois de coletar pelo menos 500 ticks):
```bash
python train_pits_model_v2.py
# Com GPU disponível:
python train_pits_model_v2.py  # treina LSTM também
# Sem GPU (mais rápido):
python train_pits_model_v2.py --skip-lstm --skip-gnn
```

---

### PASSO 4 — Iniciar o sistema

**Paper Trading (recomendado sempre primeiro):**
```bash
cd Pits
python run_pits_final.py
```

**Live Trading (capital real — cuidado):**
```bash
python run_pits_final.py --live
# Vai pedir confirmação: digite CONFIRMAR
```

---

### PASSO 5 — Abrir o dashboard

No navegador:
```
http://localhost:8001/dashboard
```

Ou se usar ngrok:
```
https://<url-ngrok>/dashboard
```

---

### O QUE APARECE NO DASHBOARD

**Barra de evento (topo amarela):**
- Mostra próximo EIA/Fed/OPEP e quantos minutos faltam
- Fica laranja quando < 30min → threshold cai para 65%
- Fica vermelha quando < 5min → sistema pausa automaticamente

**Barra de métricas (linha horizontal):**
- Win Rate, Sharpe, Drawdown, Profit Factor → calculados em tempo real
- Posições abertas, Total Ticks processados
- Modelos ativos (0/4 no início, 2/4 sem LSTM/GNN, 4/4 completo)
- Confiança do ensemble (0% = modelos discordam, 100% = unanimidade)

**Painel esquerdo — Sinais:**
- Barra colorida: verde ≥75% (opera), amarelo 60-74%, vermelho <60%
- OBI: positivo = pressão compradora no book
- TFT: >60% = compradores dominantes
- Regime macro de cada ativo

**Painel esquerdo — Regime:**
- 3 camadas: Macro (WAR/RISK_ON/OFF/INFLATION/CRISIS/NEUTRAL)
- Volatilidade: PANIC/HIGH_VOL/NORMAL_VOL/LOW_VOL
- Micro: TRENDING/RANGING

**Padrão histórico:**
- Mostra qual crise passada mais se assemelha ao momento atual
- Ajusta TP/SL automaticamente baseado no histórico

**Painel central — Features USOILm:**
- OBI 10 níveis: -1 a +1 (pressão)
- Trade Flow: % de ticks comprador
- ATR: se > $1.50 aparece vermelho → sistema pausa
- VWAP Dev%: quanto o preço desviou do preço justo
- Entropy: se > 0.55 → mercado caótico → não opera

**Painel central — Ensemble:**
- XGBoost e Bayesian: sempre verdes (Fase 1, sem treino necessário)
- LSTM e GNN: cinza até ter dados suficientes para treinar
- Barra de confiança: quanto os modelos concordam entre si

**Painel direito — Posições:**
- Posições abertas com entrada e PnL atual
- Últimos trades com resultado

**Log:**
- Eventos do sistema em tempo real
- Azul = operações normais
- Amarelo = avisos (ATR alto, pré-evento)
- Vermelho = anomalias, erros, pausas

---

### BOTÕES DE CONTROLE

- **▶ Resumir** → continua após pausa manual
- **⏸ Pausar** → para de processar sinais (coleta continua)
- **↺ Retreinar** → força retreino imediato dos modelos
- **⚡ Live Mode** → ativa trading real (pede confirmação)

---

### SEQUÊNCIA RECOMENDADA (primeiro uso)

1. `python run_pits_final.py` — inicia em paper trading
2. Aguarda MT5 conectar (badge verde no canto)
3. Aguarda acumular 500+ ticks (contador no dashboard)
4. `python train_pits_model_v2.py --skip-lstm --skip-gnn` — treina modelos
5. Sistema já usa novos modelos automaticamente (sem reiniciar)
6. Monitora Win Rate por 2-3 dias em paper
7. Se Win Rate > 55% consistente → considera live

---

### PROBLEMAS COMUNS

**MT5 não conecta:**
```
Verificar: MT5 está aberto? Conta logada? Símbolo USOILm visível no Market Watch?
```

**Dashboard mostra zeros:**
```
Verificar: python está rodando? curl http://localhost:8001/status
```

**ATR > 1.50 → sistema pausado:**
```
Normal durante notícias fortes. Aguarda estabilizar.
```

**LSTM/GNN mostram "Aguardando treino":**
```
Normal no início. Rodar: python train_pits_model_v2.py
Precisa de dados em data/ticks/ primeiro.
```
