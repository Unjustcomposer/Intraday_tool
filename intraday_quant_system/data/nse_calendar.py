"""
NSE Holiday Calendar and Trading Day Utilities
===============================================
Prevents trading on NSE holidays and handles weekend gap risk.
"""

from datetime import date, datetime, timedelta
from typing import List

# NSE holidays for 2026 (update annually)
# Source: https://www.nseindia.com/resources/exchange-communication-holidays
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Maha Shivaratri
    date(2026, 3, 17),   # Holi
    date(2026, 3, 30),   # Id-ul-Fitr (Eid)
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 25),   # Buddha Purnima
    date(2026, 6, 5),    # Eid ul-Adha (Bakri Id)
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 10, 21),  # Dussehra (2nd day)
    date(2026, 11, 9),   # Diwali (Laxmi Puja)
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 30),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

# Keep previous year for lookbacks
NSE_HOLIDAYS_2025 = {
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Maha Shivaratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Eid-ul-Fitr
    date(2025, 4, 10),   # Ram Navami
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 5, 12),   # Buddha Purnima
    date(2025, 6, 7),    # Eid ul-Adha
    date(2025, 7, 6),    # Muharram
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 20),  # Dussehra
    date(2025, 10, 21),  # Diwali (Laxmi Puja)
    date(2025, 10, 22),  # Diwali (Balipratipada)
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
}

ALL_HOLIDAYS = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


def is_trading_day(d: date = None) -> bool:
    """Check if the given date is an NSE trading day"""
    d = d or date.today()
    # Weekend check
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Holiday check
    if d in ALL_HOLIDAYS:
        return False
    return True


def get_previous_trading_day(d: date = None) -> date:
    """Get the most recent trading day before the given date"""
    d = d or date.today()
    d = d - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def days_since_last_session(d: date = None) -> int:
    """Number of calendar days since the last trading session.
    Useful for gap risk assessment — gaps after long weekends/holidays are wider."""
    d = d or date.today()
    prev = get_previous_trading_day(d)
    return (d - prev).days


def get_trading_days(start: date, end: date) -> List[date]:
    """Get list of trading days in a date range"""
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def gap_risk_multiplier(d: date = None) -> float:
    """
    Returns a multiplier for gap risk based on non-trading days since last session.
    
    - Normal overnight (1 day): 1.0x
    - Weekend (2-3 days): 1.5x  
    - Long weekend/holiday (4+ days): 2.0x
    
    Use this to widen stops on the first bar after extended closures.
    """
    gap_days = days_since_last_session(d)
    if gap_days <= 1:
        return 1.0
    elif gap_days <= 3:
        return 1.5
    else:
        return 2.0


class NSECalendar:
    """Class wrapper for NSE calendar functions to support OOP usage"""
    @staticmethod
    def is_trading_day(d: date = None) -> bool:
        return is_trading_day(d)

    @staticmethod
    def get_previous_trading_day(d: date = None) -> date:
        return get_previous_trading_day(d)

    @staticmethod
    def days_since_last_session(d: date = None) -> int:
        return days_since_last_session(d)

    @staticmethod
    def get_trading_days(start: date, end: date) -> List[date]:
        return get_trading_days(start, end)

    @staticmethod
    def gap_risk_multiplier(d: date = None) -> float:
        return gap_risk_multiplier(d)

