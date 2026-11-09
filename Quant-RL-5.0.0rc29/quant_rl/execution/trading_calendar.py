"""Conservative weekly FX entry schedule in the broker's configured IANA zone."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def weekly_fx_entry_blocked(
    utc_timestamp: int | float,
    *,
    server_timezone: str,
    friday_cutoff_hour: int = 23,
    sunday_open_hour: int = 21,
) -> bool:
    """Return whether new FX entries are blocked by the generic weekend schedule.

    ``utc_timestamp`` is an actual UTC epoch timestamp from the application.
    The broker's wall-clock schedule is derived with an explicitly configured
    IANA timezone, so DST transitions are handled by the timezone database.
    No timestamp returned by MT5 is shifted to imitate server-local time.
    """
    if not server_timezone:
        raise ValueError("MT5 server IANA timezone must be explicitly configured before LIVE")
    if not 0 <= int(friday_cutoff_hour) <= 23:
        raise ValueError("friday_cutoff_hour must be between 0 and 23")
    if not 0 <= int(sunday_open_hour) <= 23:
        raise ValueError("sunday_open_hour must be between 0 and 23")
    try:
        zone = ZoneInfo(server_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown IANA MT5 server timezone: {server_timezone!r}") from exc
    server_time = datetime.fromtimestamp(float(utc_timestamp), timezone.utc).astimezone(zone)
    if server_time.weekday() == 5:
        return True
    if server_time.weekday() == 6:
        return server_time.hour < int(sunday_open_hour)
    if server_time.weekday() == 4:
        return server_time.hour >= int(friday_cutoff_hour)
    return False
