import logging
import MetaTrader5 as mt5
from typing import Optional, Dict, Any

class OBICalculator:
    """
    Order Book Imbalance (OBI) — Fase 2.
    Substitui o OFI simples com leitura de 10 níveis do order book.
    
    Formula:
      OBI = Σ bid_vol(i)/i  -  Σ ask_vol(i)/i   (i=1..10)
    
    Níveis mais próximos do preço têm peso maior (dividido por i).
    Detecta pressão escondida onde grandes players se acumulam.
    """

    def __init__(self, levels: int = 10):
        self.logger = logging.getLogger("OBICalculator")
        self.levels = levels
        self.last_obi: float = 0.0
        self.dom_available: Optional[bool] = None  # None = ainda não testado

    def calculate(self, symbol: str) -> float:
        """
        Lê o order book do MT5 e calcula OBI ponderado.
        Retorna float entre -1.0 (pressão sell forte) e +1.0 (pressão buy forte).
        Retorna 0.0 se DOM não disponível.
        """
        try:
            book = mt5.market_book_get(symbol)
        except Exception:
            book = None

        if book is None or len(book) == 0:
            if self.dom_available is None:
                self.logger.warning(
                    f"[{symbol}] DOM não disponível — usando OFI fallback. "
                    "Para ativar: MT5 > View > Market Watch > clique direito > Book."
                )
                self.dom_available = False
            return self.last_obi

        self.dom_available = True

        bids = sorted(
            [e for e in book if e.type == mt5.BOOK_TYPE_BUY],
            key=lambda x: x.price, reverse=True
        )
        asks = sorted(
            [e for e in book if e.type == mt5.BOOK_TYPE_SELL],
            key=lambda x: x.price
        )

        bid_pressure = 0.0
        for i, level in enumerate(bids[:self.levels], start=1):
            bid_pressure += level.volume / i

        ask_pressure = 0.0
        for i, level in enumerate(asks[:self.levels], start=1):
            ask_pressure += level.volume / i

        total = bid_pressure + ask_pressure
        if total == 0:
            return 0.0

        # Normalizado entre -1 e +1
        obi = (bid_pressure - ask_pressure) / total
        self.last_obi = round(obi, 4)
        return self.last_obi

    def is_dom_available(self) -> bool:
        return self.dom_available is True
