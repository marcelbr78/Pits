import logging
import datetime
from typing import Dict, Any, List, Optional

class EconomicCalendar:
    """
    Calendário Econômico — Fase 2.
    
    Eventos de alto impacto para WTI/Oil:
      EIA Crude Oil Inventory — toda quarta 14:30 UTC
      OPEP Meeting — datas fixas anuais
      Fed Rate Decision — 8x por ano
      NFP (Non-Farm Payroll) — primeira sexta do mês 13:30 UTC
    
    Comportamento do sistema pré-evento:
      > 30 min antes : threshold reduzido para 65% (menos exigente)
      < 5 min antes  : sistema pausa (muito imprevisível)
      < 5 min depois : threshold normal volta (75%)
    """

    EVENTS_2026: List[Dict[str, Any]] = [
        # EIA toda quarta 14:30 UTC
        *[{
            'name': 'EIA Crude Oil Inventory',
            'impact': 'HIGH',
            'weekday': 2,      # quarta = 2
            'hour': 14,
            'minute': 30,
            'recurring': 'weekly'
        }],
        # Fed meetings 2026 (aproximados)
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-01-29', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-03-19', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-05-07', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-06-18', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-07-30', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-09-17', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-11-05', 'hour': 19, 'minute': 0},
        {'name': 'Fed Rate Decision', 'impact': 'HIGH', 'date': '2026-12-17', 'hour': 19, 'minute': 0},
        # OPEP meetings 2026
        {'name': 'OPEP Meeting', 'impact': 'VERY_HIGH', 'date': '2026-02-03', 'hour': 12, 'minute': 0},
        {'name': 'OPEP Meeting', 'impact': 'VERY_HIGH', 'date': '2026-06-01', 'hour': 12, 'minute': 0},
        {'name': 'OPEP Meeting', 'impact': 'VERY_HIGH', 'date': '2026-12-07', 'hour': 12, 'minute': 0},
    ]

    def __init__(self):
        self.logger = logging.getLogger("EconomicCalendar")

    def get_next_event(self, now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """
        Retorna o próximo evento e minutos restantes.
        """
        if now is None:
            now = datetime.datetime.utcnow()

        closest: Optional[Dict[str, Any]] = None
        min_diff = float('inf')

        for event in self.EVENTS_2026:
            event_dt = self._get_next_occurrence(event, now)
            if event_dt is None:
                continue

            diff_minutes = (event_dt - now).total_seconds() / 60
            if 0 <= diff_minutes < min_diff:
                min_diff = diff_minutes
                closest = {
                    'name': event['name'],
                    'impact': event['impact'],
                    'minutes_remaining': round(diff_minutes, 1),
                    'datetime': event_dt.strftime('%Y-%m-%d %H:%M UTC'),
                }

        if closest is None:
            return {
                'name': 'Nenhum evento próximo',
                'impact': 'LOW',
                'minutes_remaining': 9999,
                'datetime': 'N/A'
            }

        return closest

    def get_trading_modifier(self, now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """
        Retorna ajuste de threshold baseado na proximidade do evento.
        
        Returns:
          prob_threshold : float — threshold mínimo para operar
          should_pause   : bool  — True = pausa total
          pre_event_flag : int   — 1 se dentro de 30 min de evento
        """
        event = self.get_next_event(now)
        mins = event['minutes_remaining']

        if mins <= 5:
            return {
                'prob_threshold': 0.99,   # Efetivamente pausa
                'should_pause': True,
                'pre_event_flag': 1,
                'reason': f"Muito próximo de {event['name']} ({mins:.1f}min)"
            }
        elif mins <= 30:
            return {
                'prob_threshold': 0.65,   # Menos exigente — pode ser catalysador
                'should_pause': False,
                'pre_event_flag': 1,
                'reason': f"Pré-evento {event['name']} ({mins:.1f}min)"
            }
        else:
            return {
                'prob_threshold': 0.75,   # Normal
                'should_pause': False,
                'pre_event_flag': 0,
                'reason': 'Normal'
            }

    def _get_next_occurrence(
        self, event: Dict[str, Any], now: datetime.datetime
    ) -> Optional[datetime.datetime]:
        """Calcula próxima ocorrência do evento a partir de agora."""
        try:
            if event.get('recurring') == 'weekly':
                weekday = event['weekday']
                h, m = event['hour'], event['minute']

                days_ahead = (weekday - now.weekday()) % 7
                candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
                candidate += datetime.timedelta(days=days_ahead)

                if candidate <= now:
                    candidate += datetime.timedelta(weeks=1)

                return candidate

            elif 'date' in event:
                dt = datetime.datetime.strptime(event['date'], '%Y-%m-%d')
                dt = dt.replace(hour=event.get('hour', 0), minute=event.get('minute', 0))
                if dt > now:
                    return dt

        except Exception as e:
            self.logger.warning(f"Erro ao calcular evento {event}: {e}")

        return None
